from typing import Callable
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer
import numpy as np


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

        self.expansion_cooldown = {}
        self.cooldown_batches = 20
        self.cooldown_between_expansions = 5

        self.layer_cap = {}

    def set_cooldown_init(self, strategy: "SupervisedTemplate"):
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                self.expansion_cooldown[name] = self.cooldown_batches
                self.layer_cap[name] = module.out_features

    def set_cooldown(self, strategy: "SupervisedTemplate", layer: str, cooldown: int):
        self.expansion_cooldown[layer] += cooldown

    def before_training_epoch(self, strategy: "SupervisedTemplate"):
        if strategy.clock.train_exp_counter == 0:
            self.set_cooldown_init(strategy)

        for layer in list(self.expansion_cooldown.keys()):
            if self.expansion_cooldown[layer] > 0:
                self.expansion_cooldown[layer] = 0

        grads = self.get_flat_gradients(strategy.model)
        if grads is None:
            return
        self.last_grads = grads.detach()

    def before_update(self, strategy: "SupervisedTemplate"):
        if strategy.clock.train_exp_epochs == strategy.train_epochs - 1:
            return

        self.expanded = False
        ec_list = []

        if self.last_grads == [] or self.last_grads is None:
            self.last_grads = self.get_flat_gradients(strategy.model)

        m = self.compute_output_decision(strategy)
        print(f"\nMargem de decisão: {m:.2f}")
        e = self.compute_global_entropy(strategy)
        print(f"\nEntropia: {e:.2f}")
        ec = self.compute_output_entropy_per_class(strategy)
        print(f"\nEntropia média por classe: {ec}")
        s = self.compute_saturated_neurons(strategy)
        print(f"\nMédia de Saturação da Camada: {s}")
        act = self.compute_activation_usage(strategy)
        print(f"\nAtivação da Camada: {act}")
        g = self.compute_gradients_norm(strategy)
        print(f"\nNorma dos gradientes da Camada: {g}")
        cg = self.compute_cos_grad(strategy)
        print(f"\nCosseno entre gradientes: {cg}")
        dg = self.compute_variance(strategy)
        print(f"\nVariância normalizada dos gradientes da Camada: {dg}")

        for v in ec.values():
            ec_list.append(v)

        if e > 0.2 or max(ec_list) > 0.3 or (1 - m) > 0.1:
            layer_votes = {}
            high_variance_layers = []

            for name, module in strategy.model.named_modules():
                if not isinstance(module, DENLayer):
                    continue

                if self.expansion_cooldown.get(name, 0) > 0:
                    self.expansion_cooldown[name] -= 1
                    continue

                s_val = s.get(name, 0)
                g_val = g.get(name, 0)
                dg_val = dg.get(name, 0)

                votes = 0
                if s_val >= self.threshold:
                    votes += 1
                if g_val >= self.threshold:
                    votes += 1
                if dg_val >= self.threshold:
                    votes += 1
                    high_variance_layers.append((name, module, s_val, g_val, dg_val))

                if votes > 0:
                    layer_votes[name] = {
                        "votes": votes,
                        "module": module,
                        "s": s_val,
                        "g": g_val,
                        "dg": dg_val,
                    }

            if layer_votes:
                top_layer = max(layer_votes.items(), key=lambda x: x[1]["votes"])
                for layer_name, info in layer_votes.items():
                    if layer_name != top_layer[0]:
                        self.set_cooldown(
                            strategy, layer_name, self.cooldown_between_expansions
                        )

                name, info = top_layer
                module = info["module"]
                s_val, g_val, dg_val = info["s"], info["g"], info["dg"]

                n_neurons = int(
                    self.compute_expansion(s_val, g_val, dg_val, self.layer_cap[name])
                )

                if n_neurons > 0 and isinstance(module, DENLayer):
                    print(
                        f"\n[EXPANSÃO POR VOTAÇÃO] Camada {name} será expandida com {n_neurons} neurônios."
                    )
                    module.expand(n_neurons)
                    params = module.new_parameters
                    strategy.optimizer.add_param_group(
                        {
                            "params": params,
                            "lr": strategy.optimizer.param_groups[0]["lr"],
                        }
                    )
                    self.__auto_expand_downstream(strategy.model, name)
                    self.expansion_cooldown[name] = self.cooldown_batches
                    self.expanded = True

            # if len(high_variance_layers) > 1:
            #     print(
            #         "\n[EXPANSÃO POR VARIÂNCIA] Múltiplas camadas com variância alta detectadas."
            #     )
            #     for name, module, s_val, g_val, dg_val in high_variance_layers:
            #         if self.expansion_cooldown.get(name, 0) > 0:
            #             continue

            #         n_neurons = int(
            #             self.compute_expansion(
            #                 s_val, g_val, dg_val, module.out_features
            #             )
            #         )

            #         n_neurons = int(
            #             self.compute_expansion(
            #                 s_val, g_val, dg_val, module.out_features
            #             )
            #         )
            #         n_neurons = max(
            #             self.min_expansion, min(n_neurons, self.max_expansion)
            #         )

            #         if n_neurons + module.out_features > self.max_neuron_layer:
            #             n_neurons = self.max_neuron_layer - module.out_features

            #         if n_neurons > 0 and isinstance(module, DENLayer):
            #             print(
            #                 f"[EXPANSÃO VARIÂNCIA] Camada {name} será expandida com {n_neurons} neurônios."
            #             )
            #             module.expand(n_neurons)
            #             params = module.new_parameters
            #             strategy.optimizer.add_param_group(
            #                 {
            #                     "params": params,
            #                     "lr": strategy.optimizer.param_groups[0]["lr"],
            #                 }
            #             )
            #             self.__auto_expand_downstream(strategy.model, name)
            #             self.expansion_cooldown[name] = self.cooldown_batches
            #             self.expanded = True

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                module.reset_state()

            if self.expanded:
                module.zero_grad()

    def after_training_epoch(self, strategy: "SupervisedTemplate"):
        if strategy.train_epochs - 2 == strategy.clock.train_exp_epochs:
            self.prune_model(strategy)

    def after_training_exp(self, strategy: "SupervisedTemplate"):
        self.prune_model(strategy)
        print(f"model: {strategy.model}")

    def prune_model(self, strategy: "SupervisedTemplate"):
        prune_threshold = 1e-3

        print("\n[PRUNING] Iniciando poda das camadas expansíveis...")

        any_pruned = False

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                print(f"\n[PRUNING] Analisando camada '{name}'...")
                usage_ratio = module.get_activation_usage_ratio()
                mean_activation = module.get_activation_mean()

                if usage_ratio is None or mean_activation is None:
                    continue

                usage_ratio = usage_ratio.detach().cpu()
                mean_activation = mean_activation.detach().cpu()

                inactive_indices = torch.nonzero(
                    (usage_ratio < prune_threshold)
                    & (mean_activation < prune_threshold)
                ).flatten()

                if len(inactive_indices) > 0:
                    print(
                        f"[PRUNING] Camada '{name}' → removendo {len(inactive_indices)} neurônios."
                    )
                    module.prune(inactive_indices.tolist())
                    self.__auto_expand_downstream(strategy.model, name)
                    any_pruned = True

        if any_pruned:
            self.recreate_optimizer_with_safe_state(strategy)
            print("[PRUNING] Otimizador atualizado com os novos parâmetros.")

        else:
            print("[PRUNING] Nenhum neurônio foi podado.")

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

    def compute_activation_usage(self, strategy: "SupervisedTemplate", threshold=1e-3):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                mean = module.get_activation_usage_ratio()

                usage = (mean < threshold).sum().item()

                total = mean.numel()

                layers[name] = (usage / total, usage)

        return layers

    # Grads Norm
    def compute_gradients_norm(self, strategy: "SupervisedTemplate"):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                grads = module.compute_average_gradient_norm()
                layers[name] = grads

        return layers

    # Cosine Gradient
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
            if isinstance(module, DENLayer):
                if module.weights.grad is None:
                    continue

                grads_per_neuron = module.weights.grad.data.norm(2, dim=1)

                temporal_weights = module.compute_temporal_weights()

                current_var = self.weighted_variance(grads_per_neuron, temporal_weights)

                if name not in self.ema_variance:
                    self.ema_variance[name] = current_var
                    self.ema_var2[name] = current_var
                else:
                    prev_ema = self.ema_variance[name]
                    self.ema_variance[name] = (
                        self.ema_alpha * current_var + (1 - self.ema_alpha) * prev_ema
                    )

                    self.ema_var2[name] = (
                        self.ema_alpha * (current_var**2)
                        + (1 - self.ema_alpha) * self.ema_var2[name]
                    )

                delta = (self.ema_var2[name] - self.ema_variance[name] ** 2) ** 0.5
                norm_var = abs((current_var - self.ema_variance[name]) / delta)

                layers[name] = norm_var

        return layers

    def weighted_variance(self, values: torch.Tensor, weights: torch.Tensor):
        weights = weights.to(values.device)
        weights = weights / weights.sum()

        mean = torch.sum(weights * values)
        variance = torch.sum(weights * (values - mean) ** 2)

        return variance.item()

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

    def compute_expansion(self, s, g, dg, nl):
        a = self.alpha_factor * (s / self.threshold)
        b = self.beta_factor * (g / self.threshold)
        c = self.gamma_factor * max((dg - self.threshold) / self.threshold, 0)
        sum = (a + b + c) * self.scale_factor * nl
        return sum

    def compute_lr_rate(
        self, strategy: "SupervisedTemplate", model: nn.Module, lr: float
    ):
        for name, module in model.named_modules():
            if isinstance(module, DENLayer):
                old_params = module.old_parameters
                new_params = module.new_parameters

                delta = len(new_params) - len(old_params)

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

            if not found:
                continue

            if hasattr(module, "in_features"):
                old_in_features = module.in_features
                if old_in_features == expanded_output_dim:
                    return

                if isinstance(module, nn.Linear):
                    new_module = nn.Linear(expanded_output_dim, module.out_features)
                elif isinstance(module, DENLayer):
                    new_module = DENLayer(expanded_output_dim, module.out_features)
                else:
                    continue

                with torch.no_grad():
                    in_features_to_copy = min(old_in_features, expanded_output_dim)

                    if isinstance(module, nn.Linear):
                        new_module.weight[:, :in_features_to_copy] = module.weight[
                            :, :in_features_to_copy
                        ]
                        new_module.bias = module.bias
                    elif isinstance(module, DENLayer):
                        new_module.weights[:, :in_features_to_copy] = module.weights[
                            :, :in_features_to_copy
                        ]
                        new_module.bias = module.bias
                        new_module._last_activation_sum = module._last_activation_sum
                        new_module._last_activation_usage = (
                            module._last_activation_usage
                        )
                        new_module._activation_sum = module._activation_sum
                        new_module._activation_usage = module._activation_usage
                        new_module._counter = module._counter
                        new_module._last_counter = module._last_counter

                parent = model
                path = name.split(".")
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], new_module)
                return

    def recreate_optimizer_with_safe_state(self, strategy):
        old_optimizer = strategy.optimizer
        optimizer_class = type(old_optimizer)
        lr = old_optimizer.param_groups[0]["lr"]

        new_optimizer = optimizer_class(strategy.model.parameters(), lr=lr)

        try:
            self.safe_restore_optimizer_state(old_optimizer, new_optimizer)
            print(
                "[PRUNING] Estado parcial do otimizador restaurado (parâmetros compatíveis)."
            )
        except Exception as e:
            print(f"[PRUNING] Falha ao restaurar estado parcial do otimizador: {e}")

        strategy.optimizer = new_optimizer

    def safe_restore_optimizer_state(self, old_optimizer, new_optimizer):
        old_state_dict = old_optimizer.state_dict()
        old_state = old_state_dict["state"]

        new_params = list(new_optimizer.param_groups[0]["params"])
        old_params = list(old_optimizer.param_groups[0]["params"])

        new_state = {}
        param_mapping = {}

        for old_p, new_p in zip(old_params, new_params):
            if old_p.shape == new_p.shape:
                param_mapping[old_p] = new_p

        for old_p, state in old_state.items():
            if old_p in param_mapping:
                new_p = param_mapping[old_p]
                new_state[new_p] = state

        filtered_state_dict = {
            "state": new_state,
            "param_groups": new_optimizer.state_dict()["param_groups"],
        }

        new_optimizer.load_state_dict(filtered_state_dict)
        return new_optimizer
