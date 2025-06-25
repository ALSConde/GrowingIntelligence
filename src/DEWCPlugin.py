from collections import defaultdict
from typing import Dict, Tuple, Union
import warnings
import itertools
import torch
from torch.utils.data import DataLoader
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from DParamData import DParamData


class DEWCPlugin(SupervisedPlugin):
    def __init__(
        self,
        dewc_lambda,
        mode="separate",
        decay_factor=None,
        keep_importance_data=False,
    ):
        super().__init__()

        assert (decay_factor is None) or (
            mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
            mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
            mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.dewc_lambda = dewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params: Dict[int, Dict[str, DParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, DParamData]] = defaultdict(dict)

    def before_backward(self, strategy: "SupervisedTemplate", *args, **kwargs):
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for k, cur_param in strategy.model.named_parameters():
                    if k not in self.saved_params[experience]:
                        continue
                    saved_param = self.saved_params[experience][k]
                    imp = self.importances[experience][k]
                    new_shape = cur_param.shape
                    penalty += (
                        imp.expand(new_shape)
                        * (cur_param - saved_param.expand(new_shape)).pow(2)
                    ).sum()
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for k, cur_param in strategy.model.named_parameters():
                if k not in self.saved_params[prev_exp]:
                    continue
                saved_param = self.saved_params[prev_exp][k]
                imp = self.importances[prev_exp][k]
                new_shape = cur_param.shape
                penalty += (
                    imp.expand(new_shape)
                    * (cur_param - saved_param.expand(new_shape)).pow(2)
                ).sum()
        else:
            raise ValueError("Wrong DEWC mode.")

        strategy.loss += self.dewc_lambda * penalty

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
            num_workers=kwargs.get("num_workers", 0),
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = self.copy_params_dict(strategy.model)

        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size, num_workers=0
    ) -> Dict[str, DParamData]:
        model.eval()

        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        importances = self.zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for i, batch in enumerate(dataloader):
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                self.importances[t][k1] = DParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]
        else:
            raise ValueError("Wrong DEWC mode.")

    def copy_params_dict(
        self, model: torch.nn.Module, copy_grad=False
    ) -> Dict[str, DParamData]:
        out: Dict[str, DParamData] = {}
        for k, p in model.named_parameters():
            if copy_grad and p.grad is None:
                continue
            init = p.grad.data.clone() if copy_grad else p.data.clone()
            out[k] = DParamData(k, p.shape, device=p.device, init_tensor=init)
        return out

    def zerolike_params_dict(self, model: torch.nn.Module) -> Dict[str, DParamData]:
        return dict(
            [
                (k, DParamData(k, p.shape, device=p.device))
                for k, p in model.named_parameters()
            ]
        )


ParamDict = Dict[str, Union[DParamData]]
DEwcDataType = Tuple[ParamDict, ParamDict]
