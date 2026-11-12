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
    build_model,
    build_plugins,
    build_strategy,
    make_benchmark,
    method_tag,
)
from experiment.summary import fmt, write_aggregate
from experiment.tracking import EvalTracker, count_parameters


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        ],
        default="cnn",
        help="cnn=baseline, we=Width Expansion, cnn_att=só atenção, we_att=WE+Atenção; "
        "ecnn=CNN de largura ajustável (--width_mult), we2=expansão com orçamento "
        "(conv+FC), we2_fc=expansão com orçamento só nas FC; sufixo _att = com atenção "
        "(ecnn_att, we2_att, we2_fc_att)",
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
    p.add_argument("--n_exp", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=128)
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
    p.add_argument("--ewc_lambda", type=float, default=1e9)
    p.add_argument("--si_lambda", type=float, default=1e9)
    p.add_argument("--lwf_beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--mem_per_class", type=int, default=100)
    p.add_argument("--agem_ref_batch", type=int, default=256)
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
    # DER / Orth-DER
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
    p.add_argument("--orth_reduce", choices=["sum", "mean"], default="sum")
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
    t_start = time.time()
    for experience in benchmark.train_stream:
        step = experience.current_experience
        params_before = count_parameters(strategy.model)
        t0 = time.time()
        strategy.train(experience)
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
    return {
        "config": {
            **vars(args),
            "seed": seed,
            "tag": method_tag(args),
            "n_classes": n_classes,
        },
        "params_initial": params_initial,
        "steps": tracker.steps,
        "curve": tracker.curve,
        "epoch_log": tracker.epoch_log,
        "expansion_log": expansion_log,
        "summary": summary,
        "total_seconds": time.time() - t_start,
    }


def main(argv=None):
    args = get_args(argv)
    if args.strategy in DER_FAMILY and args.arch != "cnn":
        print(
            "[aviso] --arch é ignorado para der/orth_der (usam o DERNet com a CNN do artigo)"
        )
    if args.prune and args.strategy not in DER_FAMILY:
        raise SystemExit("--prune só vale para --strategy der/orth_der")
    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    tag = method_tag(args)
    out_dir = args.out_dir or os.path.join(
        "results", tag + ("_smoke" if args.smoke else "")
    )
    os.makedirs(out_dir, exist_ok=True)

    for seed in args.seeds:
        path = os.path.join(out_dir, f"seed{seed}.json")
        if os.path.exists(path) and not args.no_resume:
            print(
                f"[run] seed {seed} já existe em {path}; pulando (use --no_resume para refazer)"
            )
            continue
        print(f"\n===== {tag} | seed {seed} | dispositivo {device} =====")
        result = run_seed(args, seed, device)
        with open(path, "w") as f:
            json.dump(result, f)
        print(f"[run] salvo {path}")

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
