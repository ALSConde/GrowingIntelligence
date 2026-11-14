import torch
from torch.utils.data import TensorDataset

from avalanche.benchmarks import nc_benchmark


def make_synthetic_benchmark(
    n_experiences: int = 3,
    n_classes: int = 12,
    shape=(3, 32, 32),
    n_train: int = 60,
    n_test: int = 20,
    seed: int = 0,
    fixed_order=None,
):
    """fixed_order=None -> ordem aleatória (a partir de `seed`); lista -> essa ordem."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(n_classes, *shape, generator=g)

    def make(n):
        y = torch.arange(n_classes).repeat_interleave(n)
        x = protos[y] + 0.5 * torch.randn(len(y), *shape, generator=g)
        return TensorDataset(x, y)

    return nc_benchmark(
        make(n_train),
        make(n_test),
        n_experiences=n_experiences,
        task_labels=False,
        fixed_class_order=fixed_order,
        seed=seed,
    )
