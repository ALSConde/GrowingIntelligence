import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from avalanche.benchmarks import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import (
    EvaluationPlugin,
    EWCPlugin,
    SynapticIntelligencePlugin,
    MASPlugin,
)
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    bwt_metrics,
    forward_transfer_metrics,
)
from avalanche.logging.interactive_logging import InteractiveLogger
from Model_MLP import Model_MLP
from DEWCPlugin import DEWCPlugin
from DSIPlugin import DSynapticIntelligencePlugin
from Model_DEN import Model_DEN_CIL
from DENExpansionPlugin import DENExpansionPlugin


def main():
    # Create the benchmark
    benchmark = SplitMNIST(
        n_experiences=5,
        return_task_id=False,
        fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # Create the model
    model = Model_DEN_CIL()
    # model = Model_MLP()
    # ewc = DEWCPlugin(dewc_lambda=1e9)
    # ewc = EWCPlugin(ewc_lambda=1e9)
    si = DSynapticIntelligencePlugin(si_lambda=1e9)
    expansion_plugin = DENExpansionPlugin(
        growth_threshold=0.95,
        growth_factor=0.5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Create the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
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
        train_epochs=4,
        eval_mb_size=64,
        evaluator=eval_plugin,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plugins=[expansion_plugin, si],
        # plugins=[expansion_plugin, ewc],
        # plugins=[expansion_plugin],
        # plugins=[ewc],
    )

    # Training loop
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        print("Classes in this experience:", experience.classes_in_this_experience)
        strategy.train(experience)
        print(f"Training completed for experience {experience.current_experience}")

        print("Starting evaluation...")
        strategy.eval(benchmark.test_stream)
        print("Evaluation completed.")


if __name__ == "__main__":
    main()
