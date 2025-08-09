from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer
from DEWCPlugin import DEWCPlugin
from typing import Union, List


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
        self.cooldown_between_expansions = 15

        self.layer_cap = {}

        self.base_lambda = 1e20
        self.cosine_grad_reg = None
        self.consolidator = DEWCPlugin(self.base_lambda)

        self.si_importances = {}
        self.si_accumulator = {}
        self.si_prev_params = {}
        self.si_delta = {}

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

    def before_backward(self, strategy: "SupervisedTemplate"):
        if self.cosine_grad_reg is None:
            self.cosine_grad_reg = self.compute_cos_grad(strategy)
        scaling = 1.0
        if self.cosine_grad_reg < 0:
            scaling += abs(self.cosine_grad_reg)
        self.consolidator.dewc_lambda = self.base_lambda * scaling
        self.consolidator.before_backward(strategy)
        print(f"[DEWC] Penalty: {strategy.loss.item():.4f}")

    def after_backward(self, strategy: "SupervisedTemplate"):
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                for pname, param in module.named_parameters():
                    if param.grad is None or not param.requires_grad:
                        continue

                    current_data = param.data
                    current_shape = current_data.shape
                    param_name = name + "." + pname
                    if param_name.endswith(".weights") or param_name.endswith(".bias"):
                        if param_name not in self.si_prev_params:
                            self.si_prev_params[param_name] = current_data.clone()
                            self.si_accumulator[param_name] = torch.zeros_like(
                                current_data
                            )
                            self.si_delta[param_name] = torch.zeros_like(current_data)
                            continue

                    prev_data = self.si_prev_params[param_name]
                    if prev_data.shape != current_shape:
                        print(f"[SI BUFFER SYNC] Expansão detectada em '{param_name}'")
                        print(f"  Antigo shape: {prev_data.shape}")
                        print(f"  Novo shape:   {current_shape}")

                        acc_old = self.si_accumulator[param_name]
                        delta_old = self.si_delta[param_name]

                        acc_new = torch.zeros_like(current_data)
                        delta_new = torch.zeros_like(current_data)

                        min_shape = tuple(
                            min(a, b) for a, b in zip(prev_data.shape, current_shape)
                        )

                        slices = tuple(slice(0, m) for m in min_shape)
                        acc_new[slices] = acc_old[slices]
                        delta_new[slices] = delta_old[slices]

                        self.si_accumulator[param_name] = acc_new
                        self.si_delta[param_name] = delta_new
                        self.si_prev_params[param_name] = current_data.clone()
                        continue

                    delta = current_data - self.si_prev_params[param_name]
                    self.si_accumulator[param_name] += delta * param.grad
                    self.si_delta[param_name] = delta
                    self.si_prev_params[param_name] = current_data.clone()

    def before_update(self, strategy: "SupervisedTemplate"):
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

        if strategy.clock.train_exp_epochs == strategy.train_epochs - 1:
            return

        self.expanded = False

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

                n_needed = int(
                    self.compute_expansion(s_val, g_val, dg_val, self.layer_cap[name])
                )
                n_reused = self.try_reuse_neurons(strategy, module, name, n_needed)
                n_remaing = n_needed - n_reused

                if n_remaing > 0 and isinstance(module, DENLayer):
                    print(
                        f"\n[EXPANSÃO POR VOTAÇÃO] Camada {name} será expandida com {n_remaing} neurônios."
                    )
                    module.expand(n_remaing)
                    module.added_times[-n_remaing:] = strategy.clock.train_exp_epochs
                    optimizer_class = type(strategy.optimizer)
                    main_group = strategy.optimizer.param_groups[0]
                    optimizer_kwargs = {
                        "lr": main_group["lr"],
                        # "momentum": main_group.get("momentum", 0),
                        "weight_decay": main_group.get("weight_decay", 0),
                    }
                    strategy.optimizer = optimizer_class(
                        strategy.model.parameters(), **optimizer_kwargs
                    )
                    self.__auto_expand_downstream(strategy.model, name)
                    self.sync_si_buffers_after_expansion(strategy.model)
                    self.expansion_cooldown[name] = self.cooldown_batches
                    self.expanded = True

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                module.reset_state()

            if self.expanded:
                module.zero_grad()

    def after_training_exp(self, strategy: "SupervisedTemplate"):
        print(
            f"contribuição dos novos neuronios: {self.compute_new_neuron_contribution(strategy)}"
        )
        self.consolidator.after_training_exp(strategy)
        self.si_importances = self.compute_si_importance_by_neuron(strategy.model)
        self.prune_model(strategy)
        self.si_importances = self.compute_si_importance_by_neuron(strategy.model)
        print(f"model: {strategy.model}")

    def prune_model(self, strategy: "SupervisedTemplate"):
        prune_threshold = 1e-3
        si_threshold = 1e-2
        consolidator_threshold = 1e-2

        print("\n[PRUNING] Iniciando poda das camadas expansíveis...")

        any_pruned = False
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                print(
                    f"\n[PRUNING] Analisando camada '{name}' com {len(module.weights)} parametros..."
                )

                usage_ratio = module.get_activation_usage_ratio()
                # mean_activation = module.get_activation_mean()

                # if usage_ratio is None or mean_activation is None:
                if usage_ratio is None:
                    continue

                usage_ratio = usage_ratio.detach().cpu()
                # mean_activation = mean_activation.detach().cpu()

                if (
                    name not in self.si_importances
                    and (name + ".weights" or name + ".bias")
                    not in self.consolidator.importances
                ):
                    print(f"[PRUNING] Nenhuma importância encontrada para '{name}'")
                    continue

                importance = self.si_importances[name].detach().cpu()
                n_neurons = importance.shape[0]

                low_consolidator_mask = torch.ones(n_neurons, dtype=torch.bool)

                for task_id, task_importances in self.consolidator.importances.items():
                    for suffix in ["weights", "bias"]:
                        key = f"{name}.{suffix}"
                        if key in task_importances:
                            imp_tensor = task_importances[key].data.detach().cpu()

                            if imp_tensor.ndim == 2:  # weights
                                task_imp = imp_tensor.abs().sum(dim=1)
                            elif imp_tensor.ndim == 1:  # bias
                                task_imp = imp_tensor.abs()
                            else:
                                continue

                            task_imp = task_imp.to(importance.device)
                            expanded_task_imp = torch.zeros_like(importance)
                            min_len = min(expanded_task_imp.shape[0], task_imp.shape[0])
                            expanded_task_imp[:min_len] = task_imp[:min_len]
                            task_imp_norm = expanded_task_imp / (
                                expanded_task_imp.sum() + 1e-8
                            )
                            low_consolidator_mask &= (
                                task_imp_norm < consolidator_threshold
                            )

                inactive_indices = torch.nonzero(
                    (importance < si_threshold)
                    & (usage_ratio < prune_threshold)
                    & low_consolidator_mask
                ).flatten()

                if len(inactive_indices) > 0:
                    print(
                        f"[PRUNING] Camada '{name}' → removendo {len(inactive_indices)} neurônios."
                    )
                    module.prune(inactive_indices.tolist())
                    self.consolidator.prune_neurons(name, inactive_indices)
                    self.__auto_prune_downstream(
                        strategy.model, name, inactive_indices.tolist()
                    )
                    self.remove_pruned_si_entries(name, inactive_indices)
                    any_pruned = True

        if any_pruned:
            self.recreate_optimizer_with_safe_state(strategy)
            print("[PRUNING] Otimizador atualizado com os novos parâmetros.")
        else:
            print("[PRUNING] Nenhum neurônio foi podado.")

    # Mean Decision margin
    @torch.no_grad()
    def compute_output_decision(self, strategy: "SupervisedTemplate"):
        outputs = strategy.mb_output
        probs = torch.softmax(outputs, dim=1)

        top2 = torch.topk(probs, k=2, dim=1)

        margins = top2.values[:, 0] - top2.values[:, 1]

        return margins.mean()

    # Mean Global Entropy
    @torch.no_grad()
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
    @torch.no_grad()
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
    @torch.no_grad()
    def compute_saturated_neurons(self, strategy: "SupervisedTemplate", threshold=1e-3):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                mean = module.get_activation_mean()

                saturated = (mean < threshold).sum().item()

                total = mean.numel()

                layers[name] = saturated / total

        return layers

    @torch.no_grad()
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
    @torch.no_grad()
    def compute_gradients_norm(self, strategy: "SupervisedTemplate"):
        layers = {}
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                grads = module.compute_average_gradient_norm()
                layers[name] = grads

        return layers

    # Cosine Gradient
    @torch.no_grad()
    def compute_cos_grad(self, strategy: "SupervisedTemplate"):
        model = strategy.model

        grads = self.get_flat_gradients(model)
        last_grads = self.last_grads

        if grads is None or last_grads is None:
            return torch.tensor(0.0)

        grads = grads.detach()
        last_grads = last_grads.detach()

        if torch.isnan(grads).any() or torch.isnan(last_grads).any():
            return torch.tensor(0.0)

        grads, last_grads = self.match_gradients(grads, last_grads)

        norm_last = torch.norm(last_grads, p=2)
        norm_grads = torch.norm(grads, p=2)

        if norm_last < 1e-8 or norm_grads < 1e-8:
            return torch.tensor(0.0)

        dot_prod = torch.dot(last_grads, grads)
        cg = dot_prod / (norm_last * norm_grads)

        return cg

    @torch.no_grad()
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
                if abs(delta) < 1e-8:
                    norm_var = torch.tensor(0.0)
                else:
                    norm_var = abs((current_var - self.ema_variance[name]) / delta)

                layers[name] = norm_var

        return layers

    @torch.no_grad()
    def weighted_variance(self, values: torch.Tensor, weights: torch.Tensor):
        weights = weights.to(values.device)
        weights = weights / weights.sum()

        mean = torch.sum(weights * values)
        variance = torch.sum(weights * (values - mean) ** 2)

        return variance.item()

    @torch.no_grad()
    def get_flat_gradients(self, model: nn.Module):
        grads = []

        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))

        if grads == []:
            return

        return torch.cat(grads)

    @torch.no_grad()
    def match_gradients(self, g1: torch.Tensor, g2: torch.Tensor):
        min_len = min(g1.numel(), g2.numel())
        return g1[:min_len], g2[:min_len]

    def compute_expansion(self, s, g, dg, nl):
        a = self.alpha_factor * (s / self.threshold)
        b = self.beta_factor * (g / self.threshold)
        c = self.gamma_factor * max((dg - self.threshold) / self.threshold, 0)
        sum = (a + b + c) * self.scale_factor * nl
        return sum

    @torch.no_grad()
    def compute_si_importance_by_neuron(self, model):
        si_by_layer = {}
        xi = 1e-3

        for name, module in model.named_modules():
            if isinstance(module, DENLayer):
                weights = module.weights
                bias = module.bias
                n = weights.shape[0]

                imp_total = torch.zeros(n, device=weights.device)

                for suffix in ["weights", "bias"]:
                    pname = f"{name}.{suffix}"
                    if pname in self.si_accumulator and pname in self.si_delta:
                        acc = self.si_accumulator[pname]
                        delta = self.si_delta[pname]
                        imp = acc / (delta**2 + xi)

                        # reshape and sum over input dims (keep per-neuron info)
                        if suffix == "weights":
                            imp = imp.view(n, -1).abs().sum(dim=1)
                        elif suffix == "bias":
                            imp = imp.view(n).abs()  # bias is [n] shape

                        imp_total += imp

                imp_total = imp_total / (imp_total.sum() + 1e-8)

                si_by_layer[name] = imp_total

        return si_by_layer

    def remove_pruned_si_entries(
        self, layer_name: str, pruned_indices: Union[List[int], torch.Tensor]
    ):
        if isinstance(pruned_indices, list):
            pruned_indices = torch.tensor(pruned_indices, dtype=torch.long)

        for param_suffix in ["weights", "bias"]:
            key = f"{layer_name}.{param_suffix}"

            for buffer_dict in [
                self.si_prev_params,
                self.si_accumulator,
                self.si_delta,
            ]:
                if key in buffer_dict:
                    tensor = buffer_dict[key]

                    if tensor.dim() == 1:
                        mask = torch.ones(tensor.size(0), dtype=torch.bool)
                        mask[pruned_indices] = False
                        buffer_dict[key] = tensor[mask]
                    elif tensor.dim() >= 2:
                        mask = torch.ones(tensor.size(0), dtype=torch.bool)
                        mask[pruned_indices] = False
                        buffer_dict[key] = tensor[mask]

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

                        new_module.bias.data.copy_(module.bias.data)

                        if module.weight.grad is not None:
                            new_module.weight.grad = torch.zeros_like(new_module.weight)
                            new_module.weight.grad[:, :in_features_to_copy] = (
                                module.weight.grad[:, :in_features_to_copy]
                            )

                        if module.bias.grad is not None:
                            new_module.bias.grad = module.bias.grad.clone()

                    elif isinstance(module, DENLayer):
                        new_module.weights[:, :in_features_to_copy] = module.weights[
                            :, :in_features_to_copy
                        ]
                        new_module.bias.data.copy_(module.bias.data)

                        if module.weights.grad is not None:
                            new_module.weights.grad = torch.zeros_like(
                                new_module.weights
                            )
                            new_module.weights.grad[:, :in_features_to_copy] = (
                                module.weights.grad[:, :in_features_to_copy]
                            )

                        if module.bias.grad is not None:
                            new_module.bias.grad = module.bias.grad.clone()

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

    def __auto_prune_downstream(
        self, model: nn.Module, layer: str, pruned_indices: List[int]
    ):
        modules = list(model.named_modules())
        layer_dict = dict(modules)

        if layer not in layer_dict:
            return

        pruned_layer = layer_dict[layer]
        pruned_output_dim = pruned_layer.out_features
        pruned_indices = torch.tensor(pruned_indices, dtype=torch.long)

        found = False
        for name, module in modules:
            if name == layer:
                found = True
                continue

            if not found:
                continue

            if hasattr(module, "in_features"):
                old_in_features = module.in_features
                if old_in_features <= pruned_output_dim:
                    continue

                if isinstance(module, nn.Linear):
                    new_module = nn.Linear(
                        old_in_features - len(pruned_indices), module.out_features
                    )
                elif isinstance(module, DENLayer):
                    new_module = DENLayer(
                        old_in_features - len(pruned_indices), module.out_features
                    )
                else:
                    continue

                mask = torch.ones(old_in_features, dtype=torch.bool)
                mask[pruned_indices] = False

                with torch.no_grad():
                    if isinstance(module, nn.Linear):
                        new_module.weight.data.copy_(module.weight[:, mask])
                        new_module.bias.data.copy_(module.bias.data)

                        if module.weight.grad is not None:
                            new_module.weight.grad = module.weight.grad[:, mask].clone()
                        if module.bias.grad is not None:
                            new_module.bias.grad = module.bias.grad.clone()

                    elif isinstance(module, DENLayer):
                        new_module.weights.data.copy_(module.weights[:, mask])
                        new_module.bias.data.copy_(module.bias.data)

                        if module.weights.grad is not None:
                            new_module.weights.grad = module.weights.grad[
                                :, mask
                            ].clone()
                        if module.bias.grad is not None:
                            new_module.bias.grad = module.bias.grad.clone()

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

    def sync_si_buffers_after_expansion(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.sync_buffer_entry(name, param.data)

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

    def sync_buffer_entry(self, name, current_data):
        if name not in self.si_prev_params:
            self.si_prev_params[name] = current_data.clone()
            self.si_accumulator[name] = torch.zeros_like(current_data)
            self.si_delta[name] = torch.zeros_like(current_data)
        else:
            prev_data = self.si_prev_params[name]
            if prev_data.shape != current_data.shape:
                acc_old = self.si_accumulator[name]
                delta_old = self.si_delta[name]

                acc_new = torch.zeros_like(current_data)
                delta_new = torch.zeros_like(current_data)

                min_shape = tuple(
                    min(a, b) for a, b in zip(prev_data.shape, current_data.shape)
                )
                slices = tuple(slice(0, m) for m in min_shape)

                acc_new[slices] = acc_old[slices]
                delta_new[slices] = delta_old[slices]

                self.si_accumulator[name] = acc_new
                self.si_delta[name] = delta_new
                self.si_prev_params[name] = current_data.clone()

    @torch.no_grad()
    def compute_new_neuron_contribution(self, strategy: "SupervisedTemplate"):
        contributions = {}

        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                new_mask = module.added_times == module.current_time
                if new_mask.sum() == 0:
                    continue

                last_act = module._last_activation_sum  # shape: [neurônios]
                last_cnt = module._last_counter

                if last_act is None or last_cnt == 0:
                    continue

                new_act_sum = last_act[new_mask]
                contribution = new_act_sum / (last_cnt + 1e-8)

                y = strategy.mb_y
                classes = torch.unique(y)
                contrib_by_class = {}
                for c in classes:
                    mask = y == c
                    if mask.sum() == 0:
                        continue
                    contrib_by_class[int(c.item())] = contribution.mean().item()

                contributions[name] = contrib_by_class

        return contributions

    def try_reuse_neurons(
        self,
        strategy: "SupervisedTemplate",
        module: DENLayer,
        name: str,
        min_required: int,
    ):
        usage = module.get_activation_usage_ratio()  # shape: [N]
        importance = self.si_importances.get(name, None) 

        if usage is None or importance is None:
            return 0

        if importance.data.shape[0] < usage.shape[0]:
            pad_size = usage.shape[0] - importance.data.shape[0]
            padded_importance = torch.cat(
                [importance.data, torch.zeros(pad_size, device=importance.data.device)]
            )
        else:
            padded_importance = importance.data

        try:
            reuse_candidates = torch.nonzero(
                (usage < 1e-3) & (padded_importance < 1e-2)
            ).flatten()
        except RuntimeError as e:
            print(f"[ERRO REUSO] Erro ao gerar candidatos de reuso: {e}")
            return 0

        n_reuse = min(min_required, len(reuse_candidates))
        if n_reuse > 0:
            reuse_indices = reuse_candidates[:n_reuse]
            print(f"[REUSO] {n_reuse} neurônios reutilizados na camada {name}")
            return n_reuse
        return 0
