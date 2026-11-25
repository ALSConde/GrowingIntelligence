"""Runner de experimentos: executa N seeds (padrão: 10) de UM método escolhido e salva,
por seed, os parâmetros antes/depois de cada etapa, a acurácia ao longo do treino e a
acurácia por classe; ao fim agrega mean ± std.

Exemplos (a partir de src/):
    # seu método e as baselines com as mesmas condições do artigo (10 seeds cada)
    python run_experiment.py --arch we_att --strategy lwf
    python run_experiment.py --arch cnn    --strategy lwf
    python run_experiment.py --arch we     --strategy naive
    python run_experiment.py --arch cnn    --strategy er
    python run_experiment.py --strategy der --prune
    python run_experiment.py --strategy orth_der --prune
    # congelamento suave: a cada expansão a LR dos pesos já existentes é multiplicada por 0.98
    python run_experiment.py --arch rwe_att --strategy lwf --lr_decay 0.98
    # controle sem expansão: mesmo decaimento por etapa num modelo de largura fixa
    python run_experiment.py --arch rcnn_att --strategy lwf --lr_decay 0.98 --lr_decay_on step
    # teste rápido do pipeline, sem baixar o CIFAR-100 (dados sintéticos)
    python run_experiment.py --arch we --strategy lwf --smoke
    # comparar resultados
    python summarize.py results/we_att_lwf results/cnn_lwf --baseline results/cnn_lwf

Valores padrão = artigo: Split CIFAR-100 em 10 etapas de 10 classes, Adam lr 1e-3, batch
256, 30 épocas por etapa, EWC/SI λ=1e9, LwF/LwM T=2 β=1, memória de 100 amostras por classe.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from experiment.builders import (
    DER_FAMILY,
    MEMORY_FREE,
    MLP_ARCHS,
    RES_ARCHS,
    WE2_ARCHS,
    build_model,
    build_plugins,
    build_strategy,
    make_benchmark,
    method_tag,
)
from experiment.summary import fmt, write_aggregate
from experiment.tracking import EvalTracker, count_parameters

PRESETS = {
    "cifar100": dict(
        n_exp=10,
        epochs=30,
        batch_size=256,
        eval_batch_size=128,
        agem_ref_batch=256,
        d_att=256,
        att_mem=64,
        arch="rcnn",
        backbone="res",
    ),
    "mnist": dict(
        n_exp=5,
        epochs=10,
        batch_size=128,
        eval_batch_size=64,
        agem_ref_batch=128,
        d_att=128,
        att_mem=32,
        arch="mlp",
        backbone="mlp",
    ),
}


def resolve_args(args, arch_given):
    pre = PRESETS[args.dataset]
    for k, v in pre.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.dataset == "mnist":
        if args.arch not in MLP_ARCHS:
            raise SystemExit(
                f"MNIST só aceita MLP sem CNN: --arch {sorted(MLP_ARCHS)} (recebido {args.arch})"
            )
        if args.strategy in DER_FAMILY and args.backbone != "mlp":
            raise SystemExit("no MNIST, der/orth_der usam --backbone mlp")
    else:
        if args.arch in MLP_ARCHS:
            raise SystemExit(
                "as arquiteturas MLP (mlp, mlp_att, mwe, mwe_att) são do --dataset mnist"
            )
        if args.backbone == "mlp":
            raise SystemExit("--backbone mlp é do --dataset mnist")
    if args.lr_decay < 1.0:
        if args.strategy in DER_FAMILY:
            raise SystemExit(
                "--lr_decay não se aplica a der/orth_der (cada extrator já é congelado)"
            )
        if args.strategy == "joint":
            raise SystemExit("--lr_decay não faz sentido com joint (uma única etapa)")
        if args.lr_decay_on == "expansion" and args.arch not in WE2_ARCHS:
            raise SystemExit(
                "--lr_decay_on expansion exige expansão por orçamento (rwe, rwe_att, rwe_fc, "
                "rwe_fc_att, we2*, mwe*). Para modelos de largura fixa (controle) use "
                "--lr_decay_on step"
            )
    # ---- receita ---------------------------------------------------------------------------
    v2 = args.recipe == "v2"
    is_der = args.strategy in DER_FAMILY
    is_res = (not is_der) and args.arch in RES_ARCHS
    att = is_res and args.arch.endswith("_att")
    recipe = {
        "lr_schedule": "cosine" if v2 else "const",
        "weight_decay": 0.01 if v2 else 0.0,
        "label_smoothing": 0.0,
        # partes de arquitetura: só onde existem
        "pool": (
            "gap"
            if (v2 and (is_res or (is_der and args.backbone == "res")))
            else "flatten"
        ),
        "emb_dim": 128 if (v2 and is_res) else 0,
        "classifier": "ncm" if (v2 and is_res) else "linear",
        "mem_mode": "slots" if (v2 and att) else "append",
        "mem_guard": "both" if (v2 and att and args.strategy != "joint") else "none",
    }
    for k, v in recipe.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.num_workers is None:
        args.num_workers = 4 if (v2 and not args.smoke) else 0
    if args.pool == "gap" and not (is_res or (is_der and args.backbone == "res")):
        raise SystemExit(
            "--pool gap vale para as arquiteturas residuais (rcnn/rwe...) e DER res"
        )
    if args.emb_dim > 0 and not is_res:
        raise SystemExit(
            "--emb_dim > 0 vale só para as arquiteturas residuais (rcnn/rwe...)"
        )
    if args.classifier == "ncm" and args.emb_dim <= 0:
        raise SystemExit(
            "--classifier ncm exige --emb_dim > 0 (os protótipos vivem no embedding)"
        )
    if args.mem_mode == "slots" and not att:
        raise SystemExit(
            "--mem_mode slots vale só para arquiteturas residuais com atenção (*_att)"
        )
    if args.mem_guard != "none" and args.mem_mode != "slots":
        raise SystemExit("--mem_guard exige --mem_mode slots")
    if args.strategy in DER_FAMILY and arch_given:
        print(
            "[aviso] --arch é ignorado para der/orth_der (usam o DERNet; escolha com --backbone)"
        )
    return args


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        choices=["cifar100", "mnist"],
        default="cifar100",
        help="mnist = Split MNIST (5 etapas), só com MLP; cifar100 = Split CIFAR-100 (10 etapas)",
    )
    p.add_argument(
        "--arch",
        choices=[
            "cnn",
            "we",
            "cnn_att",
            "we_att",
            "ecnn",
            "we2",
            "we2_fc",
            "ecnn_att",
            "we2_att",
            "we2_fc_att",
            "rcnn",
            "rwe",
            "rwe_fc",
            "rcnn_att",
            "rwe_att",
            "rwe_fc_att",
            "mlp",
            "mlp_att",
            "mwe",
            "mwe_att",
        ],
        default=None,
        help="cnn=baseline, we=Width Expansion, cnn_att=só atenção, we_att=WE+Atenção; "
        "ecnn=CNN de largura ajustável (--width_mult), we2=expansão com orçamento "
        "(conv+FC), we2_fc=expansão com orçamento só nas FC; sufixo _att = com atenção "
        "(ecnn_att, we2_att, we2_fc_att); r* = versões com conexões de salto "
        "(rcnn fixa, rwe expande blocos+FC, rwe_fc só FC); MNIST: mlp, mlp_att (fixas), "
        "mwe, mwe_att (expansão por orçamento). Padrão: rcnn (cifar100) / mlp (mnist)",
    )
    p.add_argument(
        "--strategy",
        default="naive",
        choices=[
            "naive",
            "ewc",
            "si",
            "lwf",
            "lwm",
            "er",
            "agem",
            "der",
            "orth_der",
            "joint",
        ],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--n_exp", type=int, default=None)
    p.add_argument(
        "--class_order",
        choices=["shift", "fixed", "random"],
        default="shift",
        help="shift = deslocamento afim ordem[i]=(offset+i*stride) mod N, com offset e stride sorteados "
        "por seed (quebra a continuidade dos ids); fixed = ordem natural, igual em todas as seeds; "
        "random = permutação aleatória por seed. Em todos os modos, métodos rodados com a mesma "
        "seed recebem exatamente a mesma ordem",
    )
    p.add_argument(
        "--class_stride",
        type=int,
        default=None,
        help="força o stride do modo shift (coprimo de N; 1 = rotação)",
    )
    p.add_argument(
        "--class_offset", type=int, default=None, help="força o offset do modo shift"
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="avalia no teste a cada N épocas dentro de cada etapa (0 desliga)",
    )
    p.add_argument("--out_dir", default=None, help="padrão: results/<método>")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--smoke", action="store_true", help="dados sintéticos (teste rápido)"
    )
    p.add_argument(
        "--quiet", action="store_true", help="sem logger interativo do Avalanche"
    )
    p.add_argument("--no_resume", action="store_true", help="refaz seeds já salvas")
    p.add_argument(
        "--no_audit",
        action="store_true",
        help="não anexa o ExemplarFreeAudit nas estratégias sem memória",
    )
    p.add_argument(
        "--legacy_stats",
        action="store_true",
        help="reproduz o main.py antigo: a avaliação de fim de etapa alimenta as "
        "estatísticas dos WDLayer (ver README)",
    )
    # estratégias do artigo
    p.add_argument("--ewc_lambda", type=float, default=1e3)
    p.add_argument("--si_lambda", type=float, default=1e3)
    p.add_argument("--lwf_beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--mem_per_class", type=int, default=100)
    p.add_argument("--agem_ref_batch", type=int, default=None)
    p.add_argument(
        "--d_att",
        type=int,
        default=None,
        help="dimensão de atenção (padrão: 256 cifar100, 128 mnist)",
    )
    p.add_argument(
        "--att_mem",
        type=int,
        default=None,
        help="vetores de memória da atenção (padrão: 64 cifar100, 32 mnist)",
    )
    # expansão (WE)
    p.add_argument("--growth_factor", type=float, default=0.25)
    p.add_argument("--growth_threshold", type=float, default=0.5)
    p.add_argument("--loss_threshold", type=float, default=0.5)
    p.add_argument("--w_loss", type=float, default=0.7)
    p.add_argument("--w_local", type=float, default=0.3)
    p.add_argument("--limited_expansion", action="store_true")
    p.add_argument("--capacity_limit", type=int, default=400)
    # expansão com orçamento (we2 / we2_fc) e baseline de capacidade igualada (ecnn)
    p.add_argument(
        "--width_mult",
        type=float,
        default=1.0,
        help="escala as larguras iniciais (ecnn/we2)",
    )
    p.add_argument(
        "--inner_ratio",
        type=float,
        default=0.5,
        help="canais internos do bloco residual / largura do fluxo (arquiteturas r*)",
    )
    p.add_argument(
        "--budget_per_step",
        type=float,
        default=0.05,
        help="fração dos parâmetros iniciais do extrator que pode ser acrescentada por etapa",
    )
    p.add_argument(
        "--need_signal", choices=["rank", "usage", "uniform"], default="rank"
    )
    p.add_argument("--max_growth_frac", type=float, default=0.25)
    p.add_argument("--max_width_factor", type=float, default=2.0)
    p.add_argument("--stats_images", type=int, default=2048)
    # congelamento suave por decaimento de LR (pesos antigos treinam cada vez mais devagar)
    p.add_argument(
        "--lr_decay",
        type=float,
        default=1.0,
        help="fator por geração: a cada expansão a LR dos pesos já existentes é multiplicada por "
        "este valor (ex.: 0.98); 1.0 desliga. Pesos criados na expansão j treinam com "
        "lr*decay**(k-j) após k expansões",
    )
    p.add_argument(
        "--lr_decay_on",
        choices=["expansion", "step"],
        default="expansion",
        help="expansion = a geração avança a cada expansão efetiva (arquiteturas com expansão); "
        "step = avança a cada etapa (controle para modelos de largura fixa)",
    )
    p.add_argument(
        "--lr_decay_exclude",
        nargs="*",
        default=[],
        help="prefixos de nomes de parâmetros que NÃO sofrem decaimento (ex.: classifier)",
    )
    p.add_argument(
        "--lr_decay_min",
        type=float,
        default=0.0,
        help="piso da escala de LR (0 = sem piso)",
    )
    # receita de treino/arquitetura. v2 = cosseno + pooling global + embedding fixo + protótipos +
    # memória de slots protegida; legacy = comportamento anterior. Cada opção abaixo, quando
    # informada, vale sobre a receita (para ablações); None = usa o valor da receita.
    p.add_argument("--recipe", choices=["v2", "legacy"], default="v2")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="processos de carregamento/augmentation do treino (padrão: 4 na receita v2 com "
        "GPU; 0 em legacy e em --smoke). Mudar altera o fluxo aleatório da augmentation",
    )
    p.add_argument(
        "--no_aggregate",
        action="store_true",
        help="não agrega ao fim (usado pelos processos paralelos do run_matrix --parallel)",
    )
    p.add_argument(
        "--tag_suffix", default="", help="sufixo livre no nome da pasta de resultados"
    )
    p.add_argument("--lr_schedule", choices=["const", "cosine"], default=None)
    p.add_argument("--warmup_epochs", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=None, help="AdamW se > 0")
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument(
        "--pool",
        choices=["flatten", "gap"],
        default=None,
        help="gap = pooling global antes do fc1 (arquiteturas residuais e DER)",
    )
    p.add_argument(
        "--emb_dim",
        type=int,
        default=None,
        help="largura fixa do embedding antes do classificador (0 = sem; só arquiteturas rwe/rcnn)",
    )
    p.add_argument(
        "--classifier",
        choices=["linear", "ncm"],
        default=None,
        help="cabeça reportada como principal; as duas são sempre registradas quando há embedding",
    )
    p.add_argument(
        "--no_proto_sdc", action="store_true", help="sem compensação de deriva"
    )
    p.add_argument("--proto_sigma", type=float, default=0.3)
    p.add_argument(
        "--mem_mode",
        choices=["append", "slots"],
        default=None,
        help="slots = memória lida por pesos positivos w_m(q), protegível (arquiteturas *_att)",
    )
    p.add_argument(
        "--mem_guard",
        choices=["none", "importance", "projection", "both"],
        default=None,
        help="proteção de k_mem/v_mem: importância por slot e/ou projeção do passo",
    )
    p.add_argument(
        "--att_mem_gain",
        type=float,
        default=1.0,
        help="ganho na inicialização de k_mem (mem_mode slots): >1 dá mais massa à memória",
    )
    p.add_argument("--guard_energy", type=float, default=0.95)
    p.add_argument("--guard_beta", type=float, default=4.0)
    p.add_argument("--guard_max_rank", type=float, default=0.9)
    # DER / Orth-DER
    p.add_argument(
        "--backbone",
        choices=["plain", "res", "mlp"],
        default=None,
        help="extrator por etapa do der/orth_der: res = CNN com saltos (padrão cifar100), "
        "plain = CNN do artigo, mlp = MLP (padrão mnist)",
    )
    p.add_argument(
        "--prune", action="store_true", help="poda por máscara de canais do DER"
    )
    p.add_argument(
        "--prune_lambda", type=float, default=1.0, help="λ_s da perda de esparsidade"
    )
    p.add_argument("--smax", type=float, default=400.0)
    p.add_argument("--aux_weight", type=float, default=1.0)
    p.add_argument("--lambda_s", type=float, default=0.14)
    p.add_argument("--lambda_i", type=float, default=0.14)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument(
        "--orth_reduce",
        choices=["sum", "mean", "norm"],
        default="sum",
        help="sum (padrão, fiel às Eqs. 3-5 de Dong et al., 2026): soma sobre todos os pares de "
        "classes do lote; norm (adaptação, NÃO é do artigo): L_in e L_io pela média dos pares e "
        "L_s = média sobre as classes novas do cos² ponderado pela TRM, cada termo em [0,1], "
        "para que a penalidade não cresça com o nº de classes antigas no lote; mean: média global",
    )
    p.add_argument("--keep_aux", action="store_true")
    p.add_argument("--no_trm", action="store_true")
    p.add_argument("--ft_epochs", type=int, default=30)
    p.add_argument("--ft_lr", type=float, default=0.1)
    p.add_argument("--ft_temp", type=float, default=5.0)
    return p.parse_args(argv)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seed(args, seed, device):
    set_seeds(seed)
    benchmark = make_benchmark(args, seed)
    n_classes = int(benchmark.n_classes)
    model = build_model(args, benchmark)
    plugins = build_plugins(args, model, device, n_classes)
    if args.strategy in MEMORY_FREE and not args.no_audit:
        from plugins.ExemplarFreeAudit import ExemplarFreeAudit

        plugins.append(ExemplarFreeAudit())
    tracker = EvalTracker(
        benchmark, device, eval_every=args.eval_every, legacy_stats=args.legacy_stats
    )
    plugins.append(tracker)
    strategy = build_strategy(args, model, plugins, device, n_classes)

    params_initial = count_parameters(model)
    print(f"[run] parâmetros antes do treinamento: {params_initial:,}")
    print(
        f"[run] ordem das classes ({benchmark.order_info}): {[int(c) for c in benchmark.classes_order]}"
    )
    t_start = time.time()
    for experience in benchmark.train_stream:
        step = experience.current_experience
        params_before = count_parameters(strategy.model)
        t0 = time.time()
        strategy.train(
            experience,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        seconds = time.time() - t0
        if any(
            type(p).__name__ in ("LwFPlugin", "LwMPlugin") for p in strategy.plugins
        ):
            setattr(
                strategy, "model_old", copy.deepcopy(strategy.model)
            )  # igual ao main.py
        params_after = count_parameters(strategy.model)
        row = tracker.end_of_step(
            strategy.model, step, args.epochs, params_before, params_after, seconds
        )
        print(
            f"[run] seed {seed} etapa {step}: acc(classes vistas)={row['acc_seen'] * 100:.2f}% "
            f"parâmetros {params_before:,} -> {params_after:,} ({seconds:.0f}s)"
        )

    summary = tracker.run_summary()
    summary["params_initial"] = params_initial
    expansion_log = next(
        (
            p.history
            for p in strategy.plugins
            if type(p).__name__ == "BudgetedExpansionPlugin"
        ),
        None,
    )
    lr_decay_log = next(
        (p.history for p in strategy.plugins if type(p).__name__ == "LRDecayPlugin"),
        None,
    )
    return {
        "config": {
            **vars(args),
            "seed": seed,
            "tag": method_tag(args),
            "n_classes": n_classes,
        },
        "params_initial": params_initial,
        "classes_order": [int(c) for c in benchmark.classes_order],
        "class_order_info": benchmark.order_info,
        "steps": tracker.steps,
        "curve": tracker.curve,
        "epoch_log": tracker.epoch_log,
        "expansion_log": expansion_log,
        "lr_decay_log": lr_decay_log,
        "der_loss_log": getattr(strategy, "loss_log", None),
        "proto_log": next(
            (
                p.history
                for p in strategy.plugins
                if type(p).__name__ == "PrototypePlugin"
            ),
            None,
        ),
        "mem_guard_log": next(
            (
                p.history
                for p in strategy.plugins
                if type(p).__name__ == "MemoryGuardPlugin"
            ),
            None,
        ),
        "summary": summary,
        "total_seconds": time.time() - t_start,
    }


def main(argv=None):
    args = get_args(argv)
    args = resolve_args(args, arch_given=args.arch is not None)
    if args.prune and args.strategy not in DER_FAMILY:
        raise SystemExit("--prune só vale para --strategy der/orth_der")
    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    tag = method_tag(args)
    out_dir = args.out_dir or os.path.join(
        "results", args.dataset, tag + ("_smoke" if args.smoke else "")
    )
    os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        path = os.path.join(out_dir, f"seed{seed}.json")
        if os.path.exists(path) and not args.no_resume:
            print(
                f"[run] seed {seed} já existe em {path}; pulando (use --no_resume para refazer)"
            )
            continue
        print(
            f"\n===== {args.dataset} | {tag} | seed {seed} | dispositivo {device} ====="
        )
        result = run_seed(args, seed, device)
        tmp = (
            path + f".tmp{os.getpid()}"
        )  # escrita atômica: outro processo nunca lê JSON pela metade
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, path)
        print(f"[run] salvo {path}")

    if args.no_aggregate:
        return
    agg = write_aggregate(out_dir)
    if agg:
        print(f"\n##### {tag}: {agg['n_runs']} execuções (seeds {agg['seeds']}) #####")
        print(f"acurácia final (classes vistas)  {fmt(agg['final_acc_seen'])} %")
        print(f"acurácia média incremental       {fmt(agg['avg_incremental_acc'])} %")
        if not np.isnan(agg["forgetting"][0]):
            print(f"esquecimento                     {fmt(agg['forgetting'])} p.p.")
            print(f"BWT                              {fmt(agg['bwt'])} p.p.")
        print(
            f"parâmetros antes -> depois       {agg['params_initial'][0]:,.0f} -> {agg['params_final'][0]:,.0f}"
        )
        print(
            f"arquivos: {out_dir}/seed*.json, summary.json, per_class_final.csv, acc_per_step.csv"
        )


if __name__ == "__main__":
    main()
