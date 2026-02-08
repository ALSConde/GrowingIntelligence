from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import numpy as np
from torch import Tensor
import torch
from torch.nn.modules.batchnorm import _NormBase
from .DEWCPlugin import DEwcDataType, ParamDict
from utils.DParamData import DParamData
from avalanche.training.utils import get_layers_and_params

SynDataType = Dict[str, Dict[str, Union[DParamData, Tensor]]]

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class DSynapticIntelligencePlugin(SupervisedPlugin):
    def __init__(
        self,
        si_lambda: Union[float, Sequence[float]],
        eps: float = 0.0000001,
        excluded_parameters: Optional[Sequence[str]] = None,
        device: Any = "as_strategy",
    ):
        super().__init__()
        if excluded_parameters is None:
            excluded_parameters = []
        self.si_lambda = (
            si_lambda if isinstance(si_lambda, (list, tuple)) else [si_lambda]
        )
        self.eps: float = eps
        self.excluded_parameters: Set[str] = set(excluded_parameters)
        self.dewc_data: DEwcDataType = (dict(), dict())

        self.syn_data: SynDataType = {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }

        self._device = device

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        super().before_training_exp(strategy, **kwargs)
        DSynapticIntelligencePlugin.create_syn_data(
            strategy.model,
            self.dewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

        DSynapticIntelligencePlugin.init_batch(
            strategy.model,
            self.dewc_data,
            self.syn_data,
            self.excluded_parameters,
        )

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        super().before_backward(strategy, **kwargs)

        exp_id = strategy.clock.train_exp_counter
        try:
            si_lamb = self.si_lambda[exp_id]
        except IndexError:  # less than one lambda per experience, take last
            si_lamb = self.si_lambda[-1]

        syn_loss = DSynapticIntelligencePlugin.compute_ewc_loss(
            strategy.model,
            self.dewc_data,
            self.excluded_parameters,
            lambd=si_lamb,
            device=self.device(strategy),
        )

        if syn_loss is not None:
            strategy.loss += syn_loss.to(strategy.device)

    def before_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        super().before_training_iteration(strategy, **kwargs)
        DSynapticIntelligencePlugin.pre_update(
            strategy.model,
            self.syn_data,
            self.excluded_parameters,
        )

    def after_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        super().after_training_iteration(strategy, **kwargs)
        DSynapticIntelligencePlugin.post_update(
            strategy.model, self.syn_data, self.excluded_parameters
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        super().after_training_exp(strategy, **kwargs)
        DSynapticIntelligencePlugin.update_dewc_data(
            strategy.model,
            self.dewc_data,
            self.syn_data,
            0.001,
            self.excluded_parameters,
            1,
            eps=self.eps,
        )

    def device(self, strategy: "SupervisedTemplate"):
        if self._device == "as_strategy":
            return strategy.device
        return self._device

    @staticmethod
    @torch.no_grad()
    def create_syn_data(
        model: torch.nn.Module,
        dewc_data: DEwcDataType,
        syn_data: SynDataType,
        excluded_parameters: Set[str],
    ):
        params = DSynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        for param_name, param in params:
            if param_name not in dewc_data[0]:
                dewc_data[0][param_name] = DParamData(param_name, param.flatten().shape)
                dewc_data[1][param_name] = DParamData(
                    f"imp_{param_name}", param.flatten().shape
                )
                syn_data["old_theta"][param_name] = DParamData(
                    f"old_theta_{param_name}", param.flatten().shape
                )
                syn_data["new_theta"][param_name] = DParamData(
                    f"new_theta_{param_name}", param.flatten().shape
                )
                syn_data["grad"][param_name] = DParamData(
                    f"grad{param_name}", param.flatten().shape
                )
                syn_data["trajectory"][param_name] = DParamData(
                    f"trajectory_{param_name}", param.flatten().shape
                )
                syn_data["cum_trajectory"][param_name] = DParamData(
                    f"cum_trajectory_{param_name}", param.flatten().shape
                )
            elif dewc_data[0][param_name].data.shape != param.flatten().shape:
                # parameter expansion
                dewc_data[0][param_name].expand(param.flatten().shape)
                dewc_data[1][param_name].expand(param.flatten().shape)
                syn_data["old_theta"][param_name].expand(param.flatten().shape)
                syn_data["new_theta"][param_name].expand(param.flatten().shape)
                syn_data["grad"][param_name].expand(param.flatten().shape)
                syn_data["trajectory"][param_name].expand(param.flatten().shape)
                syn_data["cum_trajectory"][param_name].expand(param.flatten().shape)

    @staticmethod
    @torch.no_grad()
    def extract_weights(
        model: torch.nn.Module,
        target: ParamDict,
        excluded_parameters: Set[str],
    ):
        params = DSynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        for name, param in params:
            if target[name].data.shape == param.flatten().shape:
                target[name].data = param.detach().cpu().flatten()
            else:
                target[name].expand_(param.flatten().shape)
                target[name].data = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target: ParamDict, excluded_parameters: Set[str]):
        params = DSynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        for name, param in params:
            if target[name].data.shape == param.grad.flatten().shape:
                target[name].data = param.grad.detach().cpu().flatten()
            else:
                target[name].expand_(param.grad.flatten().shape)
                target[name].data = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(
        model: torch.nn.Module,
        dewc_data: DEwcDataType,
        syn_data: SynDataType,
        excluded_parameters: Set[str],
    ):
        DSynapticIntelligencePlugin.extract_weights(
            model, dewc_data[0], excluded_parameters
        )

        for param_name, param_trajectory in syn_data["trajectory"].items():
            param_trajectory.data.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(
        model: torch.nn.Module,
        syn_data: SynDataType,
        excluded_parameters: Set[str],
    ):
        DSynapticIntelligencePlugin.extract_weights(
            model, syn_data["old_theta"], excluded_parameters
        )

    @staticmethod
    @torch.no_grad()
    def post_update(
        model: torch.nn.Module, syn_data: SynDataType, excluded_parameters: Set[str]
    ):
        DSynapticIntelligencePlugin.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        DSynapticIntelligencePlugin.extract_grad(
            model, syn_data["grad"], excluded_parameters
        )

        for param_name in syn_data["trajectory"]:
            if (
                syn_data["trajectory"][param_name].data.shape
                == syn_data["grad"][param_name].data.shape
            ):
                syn_data["trajectory"][param_name].data += syn_data["grad"][
                    param_name
                ].data * (
                    syn_data["new_theta"][param_name].data
                    - syn_data["old_theta"][param_name].data
                )
            else:
                syn_data["trajectory"][param_name].expand_(
                    syn_data["grad"][param_name].data.shape
                )
                syn_data["trajectory"][param_name].data += syn_data["grad"][
                    param_name
                ].data * (
                    syn_data["new_theta"][param_name].data
                    - syn_data["old_theta"][param_name].data
                )

    @staticmethod
    def compute_ewc_loss(
        model: torch.nn.Module,
        dewc_data: DEwcDataType,
        excluded_parameters: Set[str],
        device,
        lambd=0.0,
    ):
        params = DSynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters
        )

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            ewc_data0 = (
                dewc_data[0][name].expand(weights.shape).data.to(device)
            )  # Flat, detached
            ewc_data1 = (
                dewc_data[1][name].expand(weights.shape).data.to(device)
            )  # Flat, detached
            syn_loss: Tensor = torch.dot(ewc_data1, (weights - ewc_data0) ** 2) * (
                lambd / 2
            )

        if loss is None:
            loss = syn_loss
        else:
            loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def update_dewc_data(
        net,
        dewc_data: DEwcDataType,
        syn_data: SynDataType,
        clip_to: float,
        excluded_parameters: Set[str],
        c=0.0015,
        eps: float = 0.0000001,
    ):
        DSynapticIntelligencePlugin.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )

        for param_name in syn_data["cum_trajectory"]:
            if syn_data["cum_trajectory"][param_name].data.shape == syn_data["trajectory"][
                param_name
            ].data.shape:
                syn_data["cum_trajectory"][param_name].data += (
                    c
                    * syn_data["trajectory"][param_name].data
                    / (
                        np.square(
                            syn_data["new_theta"][param_name].data
                            - dewc_data[0][param_name].data
                        )
                        + eps
                    )
                )
            else:
                syn_data["cum_trajectory"][param_name].expand_(
                    syn_data["trajectory"][param_name].data.shape
                )
                syn_data["cum_trajectory"][param_name].data += (
                    c
                    * syn_data["trajectory"][param_name].data
                    / (
                        np.square(
                            syn_data["new_theta"][param_name].data
                            - dewc_data[0][param_name].data
                        )
                        + eps
                    )
                )

        for param_name in syn_data["cum_trajectory"]:
            if dewc_data[1][param_name].data.shape == syn_data["cum_trajectory"][param_name].data.shape:
                dewc_data[1][param_name].data = torch.empty_like(
                    syn_data["cum_trajectory"][param_name].data
                ).copy_(-syn_data["cum_trajectory"][param_name].data)
            else:
                dewc_data[1][param_name].expand_(
                    syn_data["cum_trajectory"][param_name].data.shape
                )
                dewc_data[1][param_name].data = torch.empty_like(
                    syn_data["cum_trajectory"][param_name].data
                ).copy_(-syn_data["cum_trajectory"][param_name].data)

    @staticmethod
    def explode_excluded_parameters(
        excluded: Set[str],
    ) -> Set[str]:
        result = set()
        for x in excluded:
            result.add(x)
            if not x.endswith("*"):
                result.add(x + ".*")
        return result

    @staticmethod
    def not_excluded_parameters(
        model: torch.nn.Module, excluded_parameters: Set[str]
    ) -> List[Tuple[str, Tensor]]:
        result: List[Tuple[str, Tensor]] = []
        excluded_parameters = DSynapticIntelligencePlugin.explode_excluded_parameters(
            excluded_parameters
        )
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp.layer, _NormBase):
                excluded_parameters.add(lp.parameter_name)

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(
        model: torch.nn.Module, excluded_parameters: Set[str]
    ) -> List[Tuple[str, Tensor]]:
        allow_list = DSynapticIntelligencePlugin.not_excluded_parameters(
            model, excluded_parameters
        )

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result
