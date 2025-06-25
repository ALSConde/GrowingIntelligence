from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchinfo
from avalanche.models import SimpleMLP
from avalanche.benchmarks import SplitMNIST
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)

from DENExpansionPlugin import DENExpansionPlugin
from DEWCPlugin import DEWCPlugin
from Model_DEN import Model_DEN, Model_DEN_TIL
from Model_MLP import Model_MLP_TIL, Model_MLP


def main():
    result: dict = {}
    for i in range(10):
        # TIL
        # benchmark = SplitMNIST(
        #     5, return_task_id=True, class_ids_from_zero_in_each_exp=True
        # )

        # DIL
        # benchmark = SplitMNIST(
        #     5, return_task_id=False, class_ids_from_zero_in_each_exp=True
        # )

        # CIL
        benchmark = SplitMNIST(
            5, return_task_id=False, class_ids_from_zero_in_each_exp=False
        )

        interactive_logger = InteractiveLogger()

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            cpu_usage_metrics(stream=True),
            # confusion_matrix_metrics(
            #     num_classes=2, save_image=False, stream=True
            # ), # For TIL and DIL
            confusion_matrix_metrics(
                num_classes=benchmark.n_classes, save_image=False, stream=True
            ), # For CIL
            disk_usage_metrics(
                minibatch=True, epoch=True, experience=True, stream=True
            ),
            loggers=[interactive_logger],
        )

        # model = Model_DEN_TIL()
        # model = Model_MLP_TIL()
        # model = Model_MLP()
        model = Model_DEN()

        dewc = DEWCPlugin(dewc_lambda=1000)
        # ewc = EWCPlugin(ewc_lambda=1000)
        
        # den_expansion = DENExpansionPlugin(expansion_neurons_fn=lambda: 80, n_exp=benchmark.n_experiences)
        # den_expansion = DENExpansionPlugin(expansion_neurons_fn=lambda: 80, n_exp=benchmark.n_experiences, learning_type="DIL")
        den_expansion = DENExpansionPlugin(expansion_neurons_fn=lambda: 80, n_exp=benchmark.n_experiences, learning_type="CIL")

        cl_strategy = Naive(
            model=model,
            optimizer=Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999]),
            criterion=CrossEntropyLoss(),
            train_mb_size=128,
            train_epochs=4,
            eval_mb_size=64,
            # plugins=[ewc],
            # plugins=[den_expansion],
            plugins=[den_expansion, dewc],
            evaluator=eval_plugin,
        )

        print(f"Starting experiment {i + 1}...")
        results = []
        for experience in benchmark.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            # train returns a dictionary which contains all the metric values
            res = cl_strategy.train(experience)
            print("Training completed")

            print("Computing accuracy on the whole test set")
            # test also returns a dictionary which contains all the metric values
            results.append(cl_strategy.eval(benchmark.test_stream))

        result_iter: dict = results[-1]
        acc = []
        for k, v in result_iter.items():
            if k.startswith("Top1_Acc_Exp/eval_phase/test_stream/Task"):
                acc.append(v)

        acc = sum(acc) / len(acc)
        result[i + 1] = acc * 100

        print(f"Final mean accuracy for all tasks in test set: {(acc*100):.2f}%")

    print(model)
    i = 1
    for k, v in result.items():
        print(f"Para a iteração {i} a acuracia media foi {v:.2f}%")
        i += 1


if __name__ == "__main__":
    main()
