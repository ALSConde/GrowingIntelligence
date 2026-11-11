"""Benchmark sintético (imagens 3x32x32) para testar o pipeline sem baixar o CIFAR-100."""

import torch
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark


def make_synthetic_benchmark(
    n_experiences: int = 3,
    classes_per_exp: int = 4,
    n_train: int = 60,
    n_test: int = 20,
    seed: int = 0,
):
    n_classes = n_experiences * classes_per_exp
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(n_classes, 3, 32, 32, generator=g)

    def make(n):
        y = torch.arange(n_classes).repeat_interleave(n)
        x = protos[y] + 0.5 * torch.randn(len(y), 3, 32, 32, generator=g)
        return TensorDataset(x, y)

    return nc_benchmark(
        make(n_train),
        make(n_test),
        n_experiences=n_experiences,
        task_labels=False,
        fixed_class_order=list(range(n_classes)),
        seed=seed,
    )
