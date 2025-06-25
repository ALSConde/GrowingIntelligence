from typing import Callable
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer


class DENExpansionPlugin(SupervisedPlugin):
    """
    DENExpansionPlugin is a plugin for supervised continual learning that implements Dynamic Expansion Networks (DEN) strategies.
    It dynamically expands neural network layers during training based on saturation and/or accuracy criteria, supporting different
    incremental learning scenarios: Task Incremental Learning (TIL), Domain Incremental Learning (DIL), and Class Incremental Learning (CIL).
    Args:
        expansion_neurons_fn (callable): Function returning the number of neurons to add during expansion (default: lambda: 20).
        expansion_fn (callable, optional): Custom function for expansion logic (default: None).
        lr (float): Learning rate for newly added parameters (default: 0.001).
        threshold (float): Threshold for triggering expansion based on accuracy or saturation (default: 0.95).
        neuron_fraction (float): Fraction of neurons to consider for expansion (default: 0.5).
        use_accuracy (bool): Whether to trigger expansion based on accuracy (default: True).
        use_saturation (bool): Whether to trigger expansion based on neuron saturation (default: True).
        learning_type (str): Type of incremental learning. Must be one of "TIL" (Task Incremental Learning), "DIL" (Domain Incremental Learning), or "CIL" (Class Incremental Learning).
        n_exp (int): Number of experiences/tasks (required for TIL).
    Methods:
        after_training_epoch(strategy, **kwargs):
            Called after each training epoch. Checks for expansion conditions and expands layers if necessary.
        after_training_exp(strategy, *args, **kwargs):
            Called after each training experience. In TIL mode, forces expansion and freezes old weights.
        after_training(strategy, *args, **kwargs):
            Called after all training experiences. In TIL mode, removes parameters added after the last experience.
        _get_task_accuracy(strategy):
            Computes the average task accuracy from the evaluator's last metrics.
        _get_new_parameters(model):
            Retrieves new parameters added to DENLayers for optimizer update.
        _auto_expand_downstream(model, expanded_layer_name):
            Automatically updates downstream layers (e.g., nn.Linear) to match the new output dimension of an expanded layer.
        _remove_old_parameters_from_optimizer(strategy, model):
            Removes old parameters from the optimizer after expansion.
        _remove_last_expansion(model, optimizer):
            Removes parameters and expansions added during the last experience.
    """

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
        """
        :param learning_type (str): The current incremental learning type.
        :param threshold (float): Expansion threshold.
        :param neuron_fraction (float): Fraction of neurons for expansion.
        :param lr (float): Learning rate for new parameters.
        :param expansion_neurons_fn (callable): Function to determine number of neurons to add.
        :param expansion_fn (callable or None): Custom expansion function.
        :param use_expansion_fn (bool): Whether a custom expansion function is used.
        :param use_accuracy (bool): Whether to use accuracy-based expansion.
        :param use_saturation (bool): Whether to use saturation-based expansion.
        :param n_exp (int): Number of experiences/tasks.
        :param last_exp (int): Index of the last experience (set during training).
        """

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

    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        model = strategy.model
        expanded_layers = []
        expanded_by_accuracy: bool = False

        if self.use_saturation:
            for name, module in model.named_modules():
                if isinstance(module, DENLayer):
                    expanded = module.conditional_expand(
                        n_new_neurons=self.expansion_neurons_fn(),
                        threshold=self.threshold,
                        neuron_fraction=self.neuron_fraction,
                    )
                    if expanded:
                        expanded_layers.append(name)

        if self.use_accuracy:
            acc = self._get_task_accuracy(strategy)
            print(f"[DENPlugin] Acurácia atual: {acc:.4f}")
            for name, module in model.named_modules():
                if isinstance(module, DENLayer):
                    module.conditional_expand(
                        n_new_neurons=self.expansion_neurons_fn(),
                        threshold=self.threshold,
                        neuron_fraction=self.neuron_fraction,
                    )
            if acc < self.threshold:
                expanded_by_accuracy = True

        if expanded_by_accuracy:
            for name, module in model.named_modules():
                if isinstance(module, DENLayer) and name not in expanded_layers:
                    print(f"\n[DENPlugin] Expandindo '{name}' devido à baixa acurácia.")
                    module.expand_den_layer(self.expansion_neurons_fn())
                    expanded_layers.append(name)

        if not expanded_layers:
            print("\n[DENPlugin] Nenhuma expansão necessária.")
            return

        print(f"\n[DENPlugin] Camadas expandidas: {expanded_layers}")

        for name in expanded_layers:
            if hasattr(model, "expand_downstream"):
                model.expand_downstream()
            else:
                self._auto_expand_downstream(model, name)

        new_params = self._get_new_parameters(model)
        strategy.optimizer.add_param_group({"params": new_params, "lr": self.lr})

        print(
            f"\n[DENPlugin] {len(new_params[0])+len(new_params[1])} novos parâmetros adicionados ao otimizador."
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        model = strategy.model
        expanded_layers = []

        self.last_exp = strategy.experience.current_experience

        if self.learning_type == "TIL":
            print(
                "\n[DENPlugin] Modo TIL: expandindo obrigatoriamente e congelando pesos antigos."
            )
            for name, module in model.named_modules():
                if isinstance(module, DENLayer):
                    module.expand_den_layer(self.expansion_neurons_fn())
                    expanded_layers.append(name)

            if not expanded_layers:
                print("\n[DENPlugin] Nenhuma expansão necessária.")
                return

            print(f"\n[DENPlugin] Camadas expandidas: {expanded_layers}")

            for name in expanded_layers:
                if hasattr(model, "expand_downstream"):
                    model.expand_downstream()
                else:
                    self._auto_expand_downstream(model, name)

            new_params = self._get_new_parameters(model)
            strategy.optimizer.add_param_group({"params": new_params, "lr": self.lr})

            self._remove_old_parameters_from_optimizer(strategy, model)
            print(f"[DENPlugin] Pesos antigos removidos do otimizador.")

            print(
                f"\n[DENPlugin] {len(new_params[0])+len(new_params[1])} novos parâmetros adicionados ao otimizador."
            )

    def after_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        if self.learning_type == "TIL" and ((self.n_exp - 1) == self.last_exp):
            print(
                "[DENPlugin] Removendo parâmetros adicionados após a última experiência."
            )
            self._remove_last_expansion(strategy.model, strategy.optimizer)

            for name, module in strategy.model.named_modules():
                if isinstance(module, DENLayer):
                    self._auto_expand_downstream(strategy.model, name)

    def _get_task_accuracy(self, strategy: "SupervisedTemplate"):
        all_metrics = strategy.evaluator.get_last_metrics()
        acc_values = []
        for k, v in all_metrics.items():
            if k.startswith("Top1_Acc_Epoch/train_phase/train_stream/Task"):
                acc_values.append(v)

        if len(acc_values) == 0:
            return 1.0
        return sum(acc_values) / len(acc_values)

    def _get_new_parameters(self, model):
        new_params = []
        for layer in model.modules():
            if isinstance(layer, DENLayer):
                new_params.extend(layer.get_new_parameters())
        return new_params

    def _auto_expand_downstream(self, model, expanded_layer_name):
        modules = list(model.named_modules())
        layer_dict = dict(modules)

        if expanded_layer_name not in layer_dict:
            print(
                f"\n[DENPlugin] Camada {expanded_layer_name} não encontrada no modelo."
            )
            return

        expanded_layer = layer_dict[expanded_layer_name]
        expanded_output_dim = expanded_layer.out_features

        found = False
        for i, (name, module) in enumerate(modules):
            if name == expanded_layer_name:
                found = True
                continue

            if found and isinstance(module, nn.Linear):
                print(
                    f"\n[DENPlugin] Atualizando '{name}' para nova entrada {expanded_output_dim}."
                )
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

    def _remove_old_parameters_from_optimizer(
        self, strategy: "SupervisedTemplate", model: nn.Module
    ):
        old_params = []
        for module in model.modules():
            if isinstance(module, DENLayer):
                old_params.extend(module.get_old_parameters())

        old_param_ids = set(id(p) for p in old_params)
        for group in strategy.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in old_param_ids]

        # old_params = []
        # for module in model.modules():
        #     if isinstance(module, DENLayer):
        #         old_params.extend(module.get_old_parameters())

        # if not old_params:
        #     return

        # old_param_ids = set(id(p) for p in old_params)

        # for group in strategy.optimizer.param_groups:
        #     group["params"] = [p for p in group["params"] if id(p) not in old_param_ids]

        # base_group = strategy.optimizer.param_groups[0]
        # new_group = {k: v for k, v in base_group.items() if k != "params"}
        # new_group["lr"] = base_group["lr"] / 10.0
        # new_group["params"] = old_params

        # strategy.optimizer.add_param_group(new_group)

    def _remove_last_expansion(self, model: nn.Module, optimizer):
        for module in model.modules():
            if isinstance(module, DENLayer):
                module.remove_last_expansion()

        new_params = []
        for module in model.modules():
            if isinstance(module, DENLayer):
                new_params.extend(module.get_new_parameters())

        param_ids = set(id(p) for p in new_params)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in param_ids]
