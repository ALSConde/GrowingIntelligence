import math
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW

from avalanche.benchmarks import SplitCIFAR100, SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import (
    EvaluationPlugin,
    EWCPlugin,
    ReplayPlugin,
    SynapticIntelligencePlugin,
)
from avalanche.training.storage_policy import ClassBalancedBuffer

MEMORY_FREE = {"naive", "ewc", "si", "lwf", "lwm", "joint"}
DER_FAMILY = {"der", "orth_der"}
WE_ARCHS = {"we", "we_att"}
RES_ARCHS = {
    "rcnn",
    "rwe",
    "rwe_fc",
    "rcnn_att",
    "rwe_att",
    "rwe_fc_att",
}
ECNN_ARCHS = {"ecnn", "we2", "we2_fc", "ecnn_att", "we2_att", "we2_fc_att"} | RES_ARCHS
MLP_ARCHS = {
    "mlp",
    "mlp_att",
    "mwe",
    "mwe_att",
}
WE2_ARCHS = {
    "we2",
    "we2_fc",
    "we2_att",
    "we2_fc_att",
    "rwe",
    "rwe_fc",
    "rwe_att",
    "rwe_fc_att",
    "mwe",
    "mwe_att",
}
FC_ONLY_ARCHS = {"we2_fc", "we2_fc_att", "rwe_fc", "rwe_fc_att"}
SMOKE_CLASSES_PER_EXP = {"cifar100": 4, "mnist": 2}
SMOKE_SHAPE = {"cifar100": (3, 32, 32), "mnist": (1, 28, 28)}


def recipe_suffix(args) -> str:
    """'_v2' quando vale a receita nova (cosseno, GAP, embedding, ...), mais --tag_suffix livre."""
    return ("_v2" if getattr(args, "recipe", "legacy") == "v2" else "") + getattr(
        args, "tag_suffix", ""
    )


def lr_decay_suffix(args) -> str:
    """Sufixo do diretório de resultados para não sobrescrever os runs sem decaimento."""
    d = getattr(args, "lr_decay", 1.0)
    if d >= 1.0:
        return ""
    return ("_lrd" if args.lr_decay_on == "expansion" else "_lrs") + f"{d:g}"


def method_tag(args) -> str:
    if args.strategy in DER_FAMILY:
        orth_sfx = ""
        if args.strategy == "orth_der":
            # sem sufixo = comportamento antigo (soma); não sobrescreve resultados anteriores
            orth_sfx = (
                ("" if args.orth_reduce == "sum" else "_" + args.orth_reduce)
                + ("_aux" if args.keep_aux else "")
                + ("_notrm" if args.no_trm else "")
            )
        return (
            args.strategy
            + ("_" + args.backbone if args.backbone in ("res", "mlp") else "")
            + ("_p" if args.prune else "")
            + orth_sfx
            + recipe_suffix(args)
        )
    if args.strategy == "joint":
        return f"{args.arch}_joint{recipe_suffix(args)}"
    return f"{args.arch}_{args.strategy}{lr_decay_suffix(args)}{recipe_suffix(args)}"


def n_total_classes(args) -> int:
    if args.smoke:
        return args.n_exp * SMOKE_CLASSES_PER_EXP[args.dataset]
    return 10 if args.dataset == "mnist" else 100


def shift_class_order(seed, n, stride=None, offset=None):
    rng = np.random.RandomState((1_000_003 * int(seed) + n) % (2**32 - 1))
    coprime = [k for k in range(2, n) if math.gcd(k, n) == 1]
    spread = [k for k in coprime if min(k, n - k) >= 3]
    coprime = spread or coprime
    drawn_stride = int(coprime[rng.randint(len(coprime))]) if coprime else 1
    drawn_offset = int(rng.randint(n))
    stride = drawn_stride if stride is None else int(stride)
    offset = drawn_offset if offset is None else int(offset) % n
    if math.gcd(stride, n) != 1:
        raise ValueError(
            f"--class_stride {stride} is not coprime with {n} so it cannot generate a full permutation of classes."
        )
    return [(offset + i * stride) % n for i in range(n)], stride, offset


def derive_class_order(args, seed):
    n = n_total_classes(args)
    if args.class_order == "fixed":
        return list(range(n)), {"mode": "fixed"}
    if args.class_order == "random":
        return None, {"mode": "random"}
    order, stride, offset = shift_class_order(
        seed, n, args.class_stride, args.class_offset
    )
    return order, {"mode": "shift", "stride": stride, "offset": offset}


def make_benchmark(args, seed):
    n_exp = 1 if args.strategy == "joint" else args.n_exp
    order, info = derive_class_order(args, seed)
    if args.smoke:
        from experiment.synthetic import make_synthetic_benchmark

        bm = make_synthetic_benchmark(
            n_experiences=n_exp,
            n_classes=n_total_classes(args),
            shape=SMOKE_SHAPE[args.dataset],
            seed=seed,
            fixed_order=order,
        )
    elif args.dataset == "mnist":
        bm = SplitMNIST(
            n_experiences=n_exp,
            return_task_id=False,
            fixed_class_order=order,
            seed=seed,
            dataset_root="./data/mnist",
        )
    else:
        bm = SplitCIFAR100(
            n_experiences=n_exp,
            return_task_id=False,
            fixed_class_order=order,
            seed=seed,
            dataset_root="./data/cifar100",
        )
    bm.order_info = info
    return bm


def build_model(args, benchmark):
    if args.strategy in DER_FAMILY:
        from models.Model_DER import DERNet

        return DERNet(
            initial_out_features=benchmark.n_classes // benchmark.n_experiences,
            prune=args.prune,
            backbone=args.backbone,
            inner_ratio=args.inner_ratio,
            width_mult=args.width_mult,
            pool=args.pool,
        )
    if args.arch in MLP_ARCHS:
        from models.Model_EMLP import ExpandableMLP

        return ExpandableMLP(
            width_mult=args.width_mult,
            attention=args.arch.endswith("_att"),
            d_att=args.d_att,
            mem_size=args.att_mem,
        )
    if args.arch in RES_ARCHS:
        from models.Model_RCNN import ResExpandableCNN

        model = ResExpandableCNN(
            width_mult=args.width_mult,
            inner_ratio=args.inner_ratio,
            attention=args.arch.endswith("_att"),
            d_att=args.d_att,
            mem_size=args.att_mem,
            pool=args.pool,
            emb_dim=args.emb_dim,
            mem_mode=args.mem_mode,
            mem_gain=args.att_mem_gain,
        )
        model.report_head = args.classifier
        return model
    if args.arch in ECNN_ARCHS:
        from models.Model_ECNN import ExpandableCNN, ExpandableCNNAttention

        if args.arch.endswith("_att"):
            return ExpandableCNNAttention(
                width_mult=args.width_mult, d_att=args.d_att, mem_size=args.att_mem
            )
        return ExpandableCNN(width_mult=args.width_mult)
    from models.Model_MLP import Model_CNN_Cifar, Model_CNN_Cifar_attention
    from models.Model_WE import Model_WE_CIL_CIFAR, Model_WE_CIL_Cifar_attention

    cls = {
        "cnn": Model_CNN_Cifar,
        "we": Model_WE_CIL_CIFAR,
        "cnn_att": Model_CNN_Cifar_attention,
        "we_att": Model_WE_CIL_Cifar_attention,
    }[args.arch]
    return cls()


def build_plugins(args, model, device, n_classes):
    from plugins.AgemPlugin import AGEMPlugin
    from plugins.DEWCPlugin import DEWCPlugin
    from plugins.DSIPlugin import DSynapticIntelligencePlugin
    from plugins.LwFPlugin import LwFPlugin
    from plugins.LwMPlugin import LwMPlugin
    from plugins.WDExpansionPlugin import WDExpansionPlugin

    plugins = []
    dynamic = args.strategy not in DER_FAMILY and (
        args.arch in WE_ARCHS or args.arch in WE2_ARCHS
    )
    if args.strategy not in DER_FAMILY and args.arch in WE2_ARCHS:
        from plugins.BudgetedExpansionPlugin import BudgetedExpansionPlugin

        plugins.append(
            BudgetedExpansionPlugin(
                layers="fc" if args.arch in FC_ONLY_ARCHS else "all",
                need_signal=args.need_signal,
                budget_per_step=args.budget_per_step,
                loss_threshold=args.loss_threshold,
                max_growth_frac=args.max_growth_frac,
                max_width_factor=args.max_width_factor,
                stats_images=args.stats_images,
                device=device,
                verbose=not args.quiet,
            )
        )
    elif dynamic:
        kw = dict(
            growth_threshold=args.growth_threshold,
            growth_factor=args.growth_factor,
            w_loss=args.w_loss,
            w_local=args.w_local,
            loss_threshold=args.loss_threshold,
            device=device,
            limited_expansion=args.limited_expansion,
            capacity_limit=args.capacity_limit,
        )
        if args.arch == "we_att":
            kw.update(Q_layer=model.wd_2, K_layer=model.wd_1, V_layer=model.wd_1)
        plugins.append(WDExpansionPlugin(**kw))

    if args.lr_decay < 1.0:
        from plugins.LRDecayPlugin import LRDecayPlugin

        decay_plugin = LRDecayPlugin(
            decay=args.lr_decay,
            generation_on=args.lr_decay_on,
            exclude=args.lr_decay_exclude,
            min_scale=args.lr_decay_min,
            verbose=not args.quiet,
        )
        if args.lr_decay_on == "expansion":
            # plugins[0] é o BudgetedExpansionPlugin (ver acima); ele avisa a cada expansão
            decay_plugin.attach(plugins[0])
        plugins.append(decay_plugin)

    if args.strategy not in DER_FAMILY and args.arch in RES_ARCHS:
        if args.emb_dim > 0:
            from plugins.PrototypePlugin import PrototypePlugin

            # antes da expansão: o início da etapa precisa do embedding do modelo antigo (SDC)
            plugins.insert(
                0,
                PrototypePlugin(
                    sdc=not args.no_proto_sdc,
                    sigma=args.proto_sigma,
                    verbose=not args.quiet,
                ),
            )
        if args.mem_guard != "none":
            from plugins.MemoryGuardPlugin import MemoryGuardPlugin

            plugins.append(
                MemoryGuardPlugin(
                    mode=args.mem_guard,
                    energy=args.guard_energy,
                    beta=args.guard_beta,
                    max_rank_frac=args.guard_max_rank,
                    verbose=not args.quiet,
                )
            )
    if args.lr_schedule == "cosine":
        from plugins.CosineLRPlugin import CosineLRPlugin

        plugins.append(CosineLRPlugin(warmup_epochs=args.warmup_epochs))

    s = args.strategy
    mem = args.mem_per_class * n_classes
    if s == "ewc":
        plugins.append(
            DEWCPlugin(dewc_lambda=args.ewc_lambda)
            if dynamic
            else EWCPlugin(ewc_lambda=args.ewc_lambda)
        )
    elif s == "si":
        plugins.append(
            DSynapticIntelligencePlugin(si_lambda=args.si_lambda)
            if dynamic
            else SynapticIntelligencePlugin(si_lambda=args.si_lambda)
        )
    elif s == "lwf":
        plugins.append(LwFPlugin(beta=args.lwf_beta, temperature=args.temperature))
    elif s == "lwm":
        plugins.append(LwMPlugin(beta=args.lwf_beta, temperature=args.temperature))
    elif s == "er":
        plugins.append(
            ReplayPlugin(
                mem_size=mem,
                storage_policy=ClassBalancedBuffer(
                    max_size=mem, adaptive_size=False, total_num_classes=n_classes
                ),
            )
        )
    elif s == "agem":
        plugins.append(
            AGEMPlugin(
                memory_per_class=args.mem_per_class,
                max_ref_batch_size=args.agem_ref_batch,
            )
        )
    return plugins


def build_strategy(args, model, plugins, device, n_classes):
    if args.weight_decay > 0:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        loss_metrics(epoch=True),
        loggers=[] if args.quiet else [InteractiveLogger()],
    )
    common = dict(
        model=model,
        optimizer=optimizer,
        criterion=CrossEntropyLoss(label_smoothing=args.label_smoothing),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,
        eval_mb_size=args.eval_batch_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator,
    )
    if args.strategy in DER_FAMILY:
        from strategies.DERStrategy import DERStrategy

        orth = args.strategy == "orth_der"
        return DERStrategy(
            mem_per_class=args.mem_per_class,
            total_classes=n_classes,
            aux_weight=0.0 if (orth and not args.keep_aux) else args.aux_weight,
            lambda_s=args.lambda_s if orth else 0.0,
            lambda_i=args.lambda_i if orth else 0.0,
            tau=args.tau,
            orth_reduce=args.orth_reduce,
            use_trm=not args.no_trm,
            prune_lambda=args.prune_lambda,
            smax=args.smax,
            ft_epochs=args.ft_epochs,
            ft_lr=args.ft_lr,
            ft_temp=args.ft_temp,
            **common,
        )
    return Naive(**common)
