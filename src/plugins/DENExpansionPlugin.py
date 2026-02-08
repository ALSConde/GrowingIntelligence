import math
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from models.layers.DENLayer import DENLayer
from typing import Optional, Union, List
from torch.utils.data import DataLoader
from models.layers.LinearAttention import LinearAttention
from avalanche.benchmarks.utils import concat_datasets


class DENExpansionPlugin(SupervisedPlugin):
    def __init__(
        self,
        growth_threshold: float = 0.5,
        growth_factor: float = 0.2,
        w_loss: float = 0.7,
        w_local: float = 0.3,
        loss_threshold: float = 0.5,
        Q_layer: Optional[nn.Module] = None,
        K_layer: Optional[nn.Module] = None,
        V_layer: Optional[nn.Module] = None,
        device: Union[str, torch.device] = "cpu",
        limited_expansion: bool = False,
        capacity_limit: int = 400,
        growth_together: Optional[List[tuple[str, str]]] = None,
    ):
        super().__init__()
        self.growth_threshold = growth_threshold
        self.growth_factor = growth_factor
        self.device = device

        self.seen_classes: List[int] = []
        self.w_loss = w_loss
        self.w_local = w_local
        self.loss_threshold = loss_threshold

        if Q_layer is not None and K_layer is not None and V_layer is not None:
            self.Q_layer = Q_layer
            self.K_layer = K_layer
            self.V_layer = V_layer

        self.limited_expansion = limited_expansion
        if self.limited_expansion:
            self.capacity_limit = capacity_limit

        self.growth_together = growth_together or []

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):

        self.seen_classes.extend(strategy.experience.classes_in_this_experience)

        if strategy.clock.train_exp_counter == 0:
            return

        model = strategy.model

        data = concat_datasets([strategy.experience.dataset])
        dataloader = DataLoader(
            data,
            batch_size=strategy.train_mb_size,
            shuffle=False,
        )

        model.eval()
        with torch.no_grad():
            criterion = strategy._criterion
            total_loss = 0.0
            total_samples = 0
            for batch in dataloader:
                x, y, *_ = batch
                inputs = x.to(self.device)
                targets = y.to(self.device) if isinstance(y, torch.Tensor) else y
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

            max_expected_loss = (
                math.log(len(self.seen_classes)) if self.seen_classes else 0.0
            )
            global_loss_factor = min(avg_loss / max_expected_loss, 1.0)

            if avg_loss > self.loss_threshold:
                for name, module in model.named_modules():
                    if isinstance(module, DENLayer):
                        usage_ratio = module.stats.usage_ratio()

                        combined_factor = (self.w_loss * global_loss_factor) + (
                            self.w_local * usage_ratio
                        )

                        base_growth = module.out_features * self.growth_factor
                        n_new_neurons = int(round(base_growth * combined_factor))

                        min_guaranteed = (
                            int(module.out_features * 0.05)
                            if global_loss_factor > 0.5
                            else 0
                        )

                        max_allowed = int(module.out_features * 0.75)

                        n_new_neurons = max(
                            min_guaranteed, min(n_new_neurons, max_allowed)
                        )

                        if self.limited_expansion:
                            potential_size = module.out_features + n_new_neurons
                            if potential_size > self.capacity_limit:
                                n_new_neurons = max(
                                    0, self.capacity_limit - module.out_features
                                )

                        if n_new_neurons > 0:
                            print(
                                f"Expanding layer {name}: Adding {n_new_neurons} neurons (Avg Loss: {avg_loss:.4f}, Usage Ratio: {usage_ratio:.4f})"
                            )
                            for layer_a, layer_b in self.growth_together:
                                if name == layer_a:
                                    self.expand_den_and_tied(
                                        model=model,
                                        layer_name=name,
                                        n_new_neurons=n_new_neurons,
                                    )
                                    self.expand_model(
                                        model=model,
                                        layer_name=name,
                                        n_new_features=n_new_neurons,
                                    )
                                    self.expand_model(
                                        model=model,
                                        layer_name=layer_b,
                                        n_new_features=n_new_neurons,
                                    )
                            if name not in [
                                a for a, b in self.growth_together
                            ] and name not in [b for a, b in self.growth_together]:
                                module.expand(n_new_neurons)
                                self.expand_model(
                                    model=model,
                                    layer_name=name,
                                    n_new_features=n_new_neurons,
                                )
                        self.recreate_optimizer(strategy)
        model.train()

    def after_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        for name, module in strategy.model.named_modules():
            if isinstance(module, DENLayer):
                # Reset stats after each experience
                module.stats.reset()

    # utility methods to expand input features of layers to match DENLayer expansions
    def expand_model(self, model: nn.Module, layer_name: str, n_new_features: int):
        next_layer_found = False
        for name, module in model.named_modules():
            if name == layer_name:
                next_layer_found = True
                continue
            if next_layer_found:
                if isinstance(module, LinearAttention):
                    module.proj_Q = self.expand_attention_projection(
                        module.proj_Q,
                        (self.Q_layer.out_features - module.proj_Q.in_features),
                    )
                    module.proj_K = self.expand_attention_projection(
                        module.proj_K,
                        (self.K_layer.out_features - module.proj_K.in_features),
                    )
                    module.proj_V = self.expand_attention_projection(
                        module.proj_V,
                        (self.V_layer.out_features - module.proj_V.in_features),
                    )
                    break
                if isinstance(module, nn.Linear):
                    module.in_features += n_new_features
                    new_weight = torch.empty(
                        (module.out_features, n_new_features),
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
                    module.weight = nn.Parameter(
                        torch.cat([module.weight, new_weight], dim=1)
                    )
                    break
                elif isinstance(module, DENLayer):
                    module.expand_input(n_new_features)
                    break
                else:
                    continue

    def recreate_optimizer(self, strategy: "SupervisedTemplate"):
        # Recreate optimizer to include new parameters after expansion
        optimizer_class = type(strategy.optimizer)
        optimizer_params = strategy.optimizer.defaults
        strategy.optimizer = optimizer_class(
            strategy.model.parameters(), **optimizer_params
        )

    def expand_attention_projection(
        self,
        proj: nn.Linear,
        n_new_features: int,
    ):
        """
        Expande apenas a dimensão de entrada de uma projeção linear,
        preservando pesos existentes.
        """
        old_weight = proj.weight.data
        old_bias = proj.bias.data if proj.bias is not None else None

        out_features, in_features = old_weight.shape
        new_in = in_features + n_new_features

        # Nova camada
        new_proj = nn.Linear(new_in, out_features, bias=proj.bias is not None)
        new_proj = new_proj.to(old_weight.device)

        # Copia pesos antigos
        new_proj.weight.data[:, :in_features] = old_weight

        # Inicializa apenas os novos neurônios
        nn.init.kaiming_uniform_(new_proj.weight.data[:, in_features:], a=math.sqrt(5))

        if old_bias is not None:
            new_proj.bias.data = old_bias

        return new_proj

    def expand_den_and_tied(self, model, layer_name, n_new_neurons):
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, DENLayer):
                module.expand(n_new_neurons)

                # verifica se há camadas acopladas
                for a, b in self.growth_together:
                    if a == layer_name:
                        tied = dict(model.named_modules())[b]
                        if isinstance(tied, DENLayer):
                            tied.expand(n_new_neurons)

                return
