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

        self.last_grads = None

    def before_training_epoch(self, strategy: "SupervisedTemplate"):
        grads = self.get_flat_gradients(strategy.model)
        if grads is None:
            return
        self.last_grads = grads.detach()

    def before_update(self, strategy: "SupervisedTemplate", *args, **kwargs):
        ec_list = []
        if self.last_grads == [] or self.last_grads is None:
            self.last_grads = self.get_flat_gradients(strategy.model)
        m = self.compute_output_decision(strategy)
        print(f"\nMargem de decisão: {m:.2f}")
        e = self.compute_global_entropy(strategy)
        print(f"\nEntropia: {e:.2f}")
        ec = self.compute_output_entropy_per_class(strategy)
        print(f"\nEntropia media por classe: {ec}")
        s = self.compute_saturated_neurons(strategy)
        print(f"\nMedia de Saturação da Camada: {s}")
        g = self.compute_gradients_norm(strategy)
        print(f"\nNorma dos gradientes da Camada: {g}")
        cg = self.compute_cos_grad(strategy)
        print(f"\nCos dos gradientes da Camada: {cg}")
        dg = self.compute_variance(strategy)
        print(f"\nVariancia dos gradientes da Camada: {dg}")


        for k, v in ec.items():
            ec_list.append(v)

        if e >= 0.2 or (1 - m) >= 0.1 or max(ec_list) >= 0.3:
            for l, v in s.items():
                if v >= 0.85:
                    print(f"Expansion is needed in layer {l}")
            for l, v in g.items():
                if v >= 0.75:
                    print(f"Expansion is needed in layer {l}")
            for l,v in dg.items():
                if v >= 0.75:
                    print(f"Expansion is needed in layer {l}")

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                module.reset_state()

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

        num_classes = strategy.experience.benchmark.n_classes
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

    # Grads Norm
    def compute_gradients_norm(self, strategy: "SupervisedTemplate"):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                grads = module.compute_average_gradient_norm()
                layers[name] = grads

        return layers

    def compute_cos_grad(self, strategy: "SupervisedTemplate"):
        model = strategy.model

        grads = self.get_flat_gradients(model).detach()
        last_grads = self.last_grads

        grads, last_grads = self.match_gradients(grads, last_grads)

        dot_prod = torch.dot(last_grads, grads)

        norm_last = torch.norm(last_grads, p=2)
        norm_grads = torch.norm(grads, p=2)

        cg = dot_prod / (norm_last * norm_grads + 1e-8)

        return cg

    def compute_variance(self, strategy: "SupervisedTemplate"):
        layers = {}

        for name, module in strategy.model.named_modules():
            norms = []
            if isinstance(module, DENLayer):
                for param in module.parameters():
                    grads = param.grad.data.norm(2).item()
                    norms.append(grads)

                if len(norms) == 0:
                    continue

                norms_tensor = torch.tensor(norms)
                dg = torch.var(norms_tensor, unbiased=False).item()

                layers[name] = dg

        return layers

    def get_flat_gradients(self, model: nn.Module):
        grads = []

        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))

        if grads == []:
            return

        return torch.cat(grads)

    def match_gradients(self, g1: torch.Tensor, g2: torch.Tensor):
        min_len = min(g1.numel(), g2.numel())
        return g1[:min_len], g2[:min_len]
    
    def compute_expansion(self, s, g, dg, cg, nl):
        pass

    def __auto_expand_downstream(self, model: nn.Module, layers: dict):
        pass