from typing import Iterator, Optional
import torch
from torch import Tensor
from avalanche.benchmarks.utils.data_loader import (
    GroupBalancedInfiniteDataLoader,
)
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class AGEMPlugin(SupervisedPlugin):
    """
    Average Gradient Episodic Memory (A-GEM).

    - Buffer global estratificado por classe
    - Memory budget definido apenas por `memory_per_class`
    - Amostragem uniforme da memória
    - Não depende de task-id (Class-IL)
    """

    def __init__(
        self,
        memory_per_class: int,
        max_ref_batch_size: int = 256,
    ):
        """
        :param memory_per_class: número máximo de exemplos armazenados por classe
        :param max_ref_batch_size: limite superior do batch usado
                                   para o gradiente de referência
        """
        super().__init__()

        self.memory_per_class = int(memory_per_class)
        self.max_ref_batch_size = int(max_ref_batch_size)

        # buffer estratificado: lista[(x, y)]
        self.class_buffers: list = []

        # dataset e loader da memória
        self.buffer_dataloader: Optional[GroupBalancedInfiniteDataLoader] = None
        self.buffer_dliter: Iterator = iter([])

        # gradiente de referência concatenado
        self.reference_gradients: Tensor = torch.empty(0)

    # -------------------------------------------------------------
    # Gradiente de referência (memória)
    # -------------------------------------------------------------
    def before_training_iteration(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Compute reference gradient on a memory mini-batch.
        """
        if len(self.class_buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            reference_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(reference_gradients_list)
            strategy.optimizer.zero_grad()

    # -------------------------------------------------------------
    # Projeção do gradiente (A-GEM)
    # -------------------------------------------------------------
    @torch.no_grad()
    def after_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if len(self.class_buffers) > 0:
            current_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients + 1e-7
                )
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param

    # -------------------------------------------------------------
    # Atualização da memória
    # -------------------------------------------------------------
    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.experience is not None:
            self.update_memory(strategy.experience.dataset)

    @torch.no_grad()
    def update_memory(self, dataset):
        """
        Atualiza o buffer estratificado por classe usando reservoir sampling.
        """
        idx = []
        for k, v in dataset.targets.val_to_idx.items():
            idx.append(v[: self.memory_per_class])
        idx = torch.tensor(idx).flatten()
        self.class_buffers.append(dataset.subset(idx))

        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.class_buffers,
            batch_size=self.max_ref_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            persistent_workers=False,
        )

        self.buffer_dliter = iter(self.buffer_dataloader)

    def sample_from_memory(self):
        return next(self.buffer_dliter)
