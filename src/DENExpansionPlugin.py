from typing import Callable
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer


class DENExpansionPlugin(SupervisedPlugin):
    def __init__(
        self,
        expansion_neurons_fn=lambda: 20,
        expansion_fn=None,
        lr=0.001,
        threshold=0.95,
        neuron_fraction=0.5,
        use_accuracy=True,
        use_saturation=True,
        learning_type: str = "TIL",
        n_exp: int = 0,
    ):

        super().__init__()

        assert learning_type in (
            "TIL",
            "DIL",
            "CIL",
        ), "Wrong learning type. Must be TIL, DIL or CIL"
        assert not (
            n_exp <= 0 and learning_type == "TIL"
        ), "You must set `n_exp > 0` when using learning_type='TIL'"
        self.learning_type = learning_type

        self.threshold = threshold
        self.neuron_fraction = neuron_fraction
        self.lr = lr
        self.expansion_neurons_fn = expansion_neurons_fn
        self.expansion_fn = expansion_fn
        self.use_expansion_fn = False if self.expansion_fn is None else True
        self.use_accuracy = use_accuracy
        self.use_saturation = use_saturation
        self.n_exp = n_exp

    def after_training_epoch(self, strategy: "SupervisedTemplate", *args, **kwargs):
        curr_epoch = strategy.train_epochs
        d = self.compute_output_decision(strategy)
        print(f"\nMargem de decisão: {d:.2f}")
        e = self.compute_global_entropy(strategy)
        print(f"\nEntropia: {e:.2f}")
        a = self.compute_output_entropy_per_class(strategy)
        print(f"\nEntropia media por classe: {a}")
        l = self.compute_saturated_neurons(strategy)
        print(f"\nMedia de Saturação da Camada: {l}")
        t = self.compute_gradients_norm(strategy)
        print(f"\nNorma dos gradientes da Camada: {t}")

    # Mean Decision margin
    def compute_output_decision(self, strategy: "SupervisedTemplate"):
        outputs = strategy.mb_output
        probs = torch.softmax(outputs, dim=1)

        top2 = torch.topk(probs, k=2, dim=1)

        margins = top2.values[:, 0] - top2.values[:, 1]

        return margins.mean()

    # Mean Global Entropy
    def compute_global_entropy(self, strategy: "SupervisedTemplate"):
        outputs = strategy.mb_output
        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)

        num_classes = probs.size(1)
        max_entropy = torch.log(
            torch.tensor(num_classes, dtype=probs.dtype, device=probs.device)
        )

        norm_entropy = entropy / max_entropy

        return norm_entropy.mean()

    # Mean Entropy per class
    def compute_output_entropy_per_class(self, strategy: "SupervisedTemplate"):
        outputs = strategy.mb_output
        labels = strategy.mb_y

        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)

        num_classes = probs.size(1)
        max_entropy = torch.log(
            torch.tensor(num_classes, dtype=probs.dtype, device=probs.device)
        )

        norm_entropy = entropy / max_entropy
        entropy_per_class = {}
        unique_classes = torch.unique(labels)

        for c in unique_classes:
            idx = labels == c
            if idx.sum() > 0:
                class_entropy = norm_entropy[idx].mean()
                entropy_per_class[int(c.item())] = class_entropy.item()
            else:
                entropy_per_class[int(c.item())] = float("nan")

        return entropy_per_class

    # Mean Neurons saturateds by layer
    def compute_saturated_neurons(self, strategy: "SupervisedTemplate", threshold=1e-3):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                mean = module.get_activation_mean()

                saturated = (mean < threshold).sum().item()

                total = mean.numel()

                layers[name] = saturated / total

        return layers

    def compute_gradients_norm(self, strategy: "SupervisedTemplate"):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                grads = module.compute_average_gradient_norm()
                layers[name] = grads

        return layers
