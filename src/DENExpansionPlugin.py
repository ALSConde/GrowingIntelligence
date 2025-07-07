from typing import Callable
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer


class DENExpansionPlugin(SupervisedPlugin):
    def __init__(
        self,
        threshold=0.95,
        learning_type: str = "TIL",
        n_exp: int = 0,
        alpha_factor: float = 1,
        beta_factor: float = 1,
        gamma_factor: float = 1,
        scale_factor: float = 1,
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
        self.n_exp = n_exp

        self.alpha_factor = alpha_factor
        self.beta_factor = beta_factor
        self.gamma_factor = gamma_factor
        self.scale_factor = scale_factor

        self.ema_variance = {}
        self.ema_var2 = {}
        self.ema_alpha = 0.3

        self.last_grads = None
        self.expanded = False

    def before_training_epoch(self, strategy: "SupervisedTemplate"):
        grads = self.get_flat_gradients(strategy.model)
        if grads is None:
            return
        self.last_grads = grads.detach()

    def before_update(self, strategy: "SupervisedTemplate", *args, **kwargs):
        self.expanded = False
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
                    print(f"Neurons needed: {self.compute_expansion(v, g[l], dg[l], )}")
                    self.expanded = True
            for l, v in g.items():
                if v >= 0.75:
                    print(f"Expansion is needed in layer {l}")
                    print(f"Neurons needed: {self.compute_expansion(v, g[l], dg[l], )}")
                    self.expanded = True
            for l, v in dg.items():
                if v >= 0.85:
                    print(f"Expansion is needed in layer {l}")
                    print(f"Neurons needed: {self.compute_expansion(v, g[l], dg[l], )}")
                    self.expanded = True

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                module.reset_state()

            if self.expanded:
                module.zero_grad()

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
                current_var = torch.var(norms_tensor, unbiased=False).item()

                layers[name] = current_var

                if name not in self.ema_variance:
                    self.ema_variance[name] = current_var
                    self.ema_var2[name] = current_var
                else:
                    prev_ema = self.ema_variance[name]
                    self.ema_variance[name] = (
                        self.ema_alpha * current_var + (1 - self.ema_alpha) * prev_ema
                    )

                    self.ema_var2[name] = (
                        self.ema_alpha * (current_var ** 2) + (1 - self.ema_alpha) * self.ema_var2[name]
                    )

                delta = (self.ema_var2[name] - self.ema_variance[name] ** 2)**(1/2)
                norm_var = (current_var - self.ema_variance[name]) / delta
                norm_var = max(0.0, min(1.0, norm_var))

                layers[name] = norm_var

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

    def compute_expansion(self, s, g, dg):
        a = self.alpha_factor * (s / 0.85)
        b = self.beta_factor * (g / 0.75)
        c = self.gamma_factor * max((dg - 0.75) / 0.75, 0)
        sum = int(a + b + c) * self.scale_factor
        return sum

    def __auto_expand_downstream(self, model: nn.Module, layer: str):
        modules = list(model.named_modules())
        layer_dict = dict(modules)

        if layer not in layer_dict:
            return

        expanded_layer = layer_dict[layer]
        expanded_output_dim = expanded_layer.out_features

        found = False
        for i, (name, module) in enumerate(modules):
            if name == layer:
                found = True
                continue

            if found and isinstance(module, nn.Linear):
                old_linear = module
                new_linear = nn.Linear(expanded_output_dim, old_linear.out_features)

                with torch.no_grad():
                    in_features_to_copy = min(
                        old_linear.in_features, expanded_output_dim
                    )
                    new_linear.weight[:, :in_features_to_copy] = old_linear.weight[
                        :, :in_features_to_copy
                    ]
                    new_linear.bias = old_linear.bias

                parent = model
                path = name.split(".")
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], new_linear)
                return
