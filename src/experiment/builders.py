"""Construção de benchmark, modelo, plugins e estratégia a partir dos argumentos da CLI,
espelhando as configurações do main.py / do artigo (Split CIFAR-100, 10 etapas, Adam).
"""

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks import SplitCIFAR100
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


def method_tag(args) -> str:
    if args.strategy in DER_FAMILY:
        return args.strategy + ("_p" if args.prune else "")
    if args.strategy == "joint":
        return f"{args.arch}_joint"
    return f"{args.arch}_{args.strategy}"


def make_benchmark(args, seed):
    n_exp = 1 if args.strategy == "joint" else args.n_exp
    if args.smoke:
        from experiment.synthetic import make_synthetic_benchmark

        return make_synthetic_benchmark(
            n_experiences=n_exp, classes_per_exp=4, seed=seed
        )
    return SplitCIFAR100(
        n_experiences=n_exp,
        return_task_id=False,
        fixed_class_order=list(range(100)),
        seed=seed,
    )


def build_model(args, benchmark):
    if args.strategy in DER_FAMILY:
        from models.Model_DER import DERNet

        return DERNet(
            initial_out_features=benchmark.n_classes // benchmark.n_experiences,
            prune=args.prune,
        )
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
    dynamic = args.strategy not in DER_FAMILY and args.arch in WE_ARCHS
    if dynamic:
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
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True),
        loss_metrics(epoch=True),
        loggers=[] if args.quiet else [InteractiveLogger()],
    )
    common = dict(
        model=model,
        optimizer=optimizer,
        criterion=CrossEntropyLoss(),
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
