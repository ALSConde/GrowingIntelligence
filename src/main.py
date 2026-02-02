import copy
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from avalanche.benchmarks import SplitMNIST, SplitCIFAR100
from avalanche.training import Naive, AGEM
from avalanche.training.plugins import (
    EvaluationPlugin,
    EWCPlugin,
    SynapticIntelligencePlugin,
    MASPlugin,
    ReplayPlugin,
)
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
)
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.logging import CSVLogger
from models.Model_MLP import Model_MLP, Model_MLP_Cifar, model_MLP_attention
from models.Model_DEN import (
    Model_DEN_CIL,
    Model_DEN_CIL_CIFAR,
    Model_DEN_CIL_Cifar_attention,
    Model_DEN_CIL_attention,
)
from plugins.DEWCPlugin import DEWCPlugin
from plugins.DSIPlugin import DSynapticIntelligencePlugin
from plugins.DENExpansionPlugin import DENExpansionPlugin
from plugins.LwFPlugin import LwFPlugin
from plugins.AgemPlugin import AGEMPlugin
from plugins.LwMPlugin import LwMPlugin


def run_experiment(seed: int = 0):
    # Create the benchmark
    benchmark = SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        seed=seed,
        fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    # benchmark = SplitCIFAR100(
    #     n_experiences=10,
    #     return_task_id=False,
    #     fixed_class_order=list(range(100)),
    #     seed=seed,
    # )

    # Create the model
    # model = Model_DEN_CIL()
    # model = Model_MLP()
    # model = Model_MLP_Cifar()
    # model = Model_DEN_CIL_CIFAR()
    # model = Model_DEN_CIL_attention()
    model = model_MLP_attention()
    # model = Model_DEN_CIL_Cifar_attention()
    # ewc = DEWCPlugin(dewc_lambda=1e9)
    # ewc = EWCPlugin(ewc_lambda=1e9)
    # si = DSynapticIntelligencePlugin(si_lambda=1e9)
    # si = SynapticIntelligencePlugin(si_lambda=1e9)
    # agem = AGEMPlugin(memory_per_class=100, max_ref_batch_size=128)
    # replay = ReplayPlugin(mem_size=10000,storage_policy=ClassBalancedBuffer(max_size=10000, adaptive_size=False, total_num_classes=100))
    # expansion_plugin = DENExpansionPlugin(
    #     growth_factor=0.25,
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     Q_layer=model.den_2,
    #     K_layer=model.den_1,
    #     V_layer=model.den_1,
    # )
    lwf = LwFPlugin(beta=1.0, temperature=2.0)
    # lwm = LwMPlugin(beta=1.0, temperature=2.0)

    # Create the optimizer and loss function
    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
    )
    criterion = CrossEntropyLoss()

    # Create the evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            epoch=True, experience=True, stream=True, trained_experience=True
        ),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    # Create the training strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=5,
        eval_mb_size=64,
        evaluator=eval_plugin,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # plugins=[expansion_plugin, ewc],
        # plugins=[expansion_plugin, si],
        # plugins=[expansion_plugin, lwf],
        # plugins=[expansion_plugin, lwm],
        # plugins=[expansion_plugin, replay],
        # plugins=[expansion_plugin, agem],
        # plugins=[ewc],
        # plugins=[si],
        plugins=[lwf],
        # plugins=[lwm],
        # plugins=[replay],
        # plugins=[agem],
    )

    # Training loop
    results: dict = {}
    for experience in benchmark.train_stream:
        # print(f"Start training on experience {experience.current_experience}")
        # print("Classes in this experience:", experience.classes_in_this_experience)
        strategy.train(experience)
        # print(f"Training completed for experience {experience.current_experience}")

        if any(
            isinstance(plugin, LwFPlugin) or isinstance(plugin, LwMPlugin)
            for plugin in strategy.plugins
        ):
            setattr(strategy, "model_old", copy.deepcopy(strategy.model))

        # print("Starting evaluation...")
        results = strategy.eval(benchmark.test_stream)
        # print("Evaluation completed.")

    return results


def main():
    final_results = {}

    for run_id in range(1):
        print(f"Run {run_id + 1}/10")
        r = run_experiment(seed=run_id + 1)

        # --- Acurácia final do stream ---
        final_acc = r.get("Top1_Acc_Stream/eval_phase/test_stream/Task000")

        # --- Acurácia por experiência ---
        exp_acc = {
            k.split("/")[-1]: v
            for k, v in r.items()
            if k.startswith("Top1_Acc_Exp/eval_phase/test_stream/")
        }

        # --- Forgetting por experiência ---
        exp_forgetting = {
            k.split("/")[-1]: v
            for k, v in r.items()
            if k.startswith("ExperienceForgetting/eval_phase/test_stream/")
        }

        # --- BWT por experiência ---
        exp_bwt = {
            k.split("/")[-1]: v
            for k, v in r.items()
            if k.startswith("ExperienceBWT/eval_phase/test_stream/")
        }

        def sort_exp(d):
            return dict(sorted(d.items(), key=lambda x: int(x[0].replace("Exp", ""))))

        exp_acc = sort_exp(exp_acc)
        exp_forgetting = sort_exp(exp_forgetting)
        exp_bwt = sort_exp(exp_bwt)

        final_results[run_id] = {
            "final_accuracy": final_acc,
            "per_experience_accuracy": exp_acc,
            "forgetting": exp_forgetting,
            "bwt": exp_bwt,
        }

    for run_id, data in final_results.items():
        print(f"\nRun {run_id + 1}")
        print(f"  Final Accuracy: {data['final_accuracy'] * 100:.2f}%")

        print("  Per-experience accuracy:")
        for exp, acc in data["per_experience_accuracy"].items():
            print(f"    {exp}: {acc * 100:.2f}%")

        print("  Forgetting:")
        for exp, val in data["forgetting"].items():
            print(f"    {exp}: {val:.4f}")

        print("  BWT:")
        for exp, val in data["bwt"].items():
            print(f"    {exp}: {val:.4f}")


if __name__ == "__main__":
    main()
