import math
from avalanche.models import IncrementalClassifier
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import torch.nn as nn
from DENLayer import DENLayer
from DEWCPlugin import DEWCPlugin
from typing import Union, List, Dict, Any
from torch.utils.data import DataLoader
from avalanche.models.dynamic_modules import DynamicModule, MultiTaskModule
from torch.nn.parallel import DistributedDataParallel
from avalanche.benchmarks.utils import concat_datasets


class DENExpansionPlugin(SupervisedPlugin):
    def __init__(
        self,
        growth_threshold: float = 0.5,
        growth_factor: float = 0.2,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.growth_threshold = growth_threshold
        self.growth_factor = growth_factor
        self.device = device

    def before_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            return

        model = strategy.model

        model.eval()
        dataset = concat_datasets([strategy.experience.dataset])
        data = DataLoader(
            dataset,
            batch_size=strategy.train_mb_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            for batch in data:
                x, y, *_ = batch
                inputs = x.to(self.device)
                _ = model(inputs)

            for name, module in model.named_modules():
                if isinstance(module, DENLayer):
                    act_mean = module.stats.get_activation_mean(self.growth_threshold)
                    usage_ratio = module.stats.usage_ratio(self.growth_threshold)

                    if usage_ratio < self.growth_threshold and act_mean > 0:
                        # Improved adaptive growth decision:
                        # - base growth proportional to current size and growth_threshold
                        # - scale by bounded activity and unused capacity factors
                        # - enforce sensible min/max caps to avoid explosive growth
                        activity_factor = min(
                            act_mean, 1.0
                        )  # cap unexpected large activations
                        unused_factor = max(0.0, 1.0 - usage_ratio)
                        base = module.out_features * self.growth_factor

                        # tunable weighting of each factor
                        activity_weight = 0.75
                        unused_weight = 0.5

                        raw_growth = base * (
                            1.0
                            + activity_factor * activity_weight
                            + unused_factor * unused_weight
                        )
                        n_new_neurons = int(round(raw_growth))

                        # enforce at least one new neuron (since condition met) and cap growth to 50% of current size
                        min_new = 1
                        max_new = max(1, int(module.out_features * self.growth_threshold))
                        n_new_neurons = max(min_new, min(n_new_neurons, max_new))
                        module.expand(n_new_neurons)
                        self.expand_model(
                            model=model, layer_name=name, n_new_features=n_new_neurons
                        )

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
