import math
from avalanche.core import SupervisedPlugin


class CosineLRPlugin(SupervisedPlugin):
    def __init__(self, min_factor: float = 0.0, warmup_epochs: float = 0.0):
        super().__init__()
        self.min_factor = float(min_factor)
        self.warmup_epochs = float(warmup_epochs)
        self._it = 0
        self._total = 1
        self._warm = 0

    def before_training_exp(self, strategy, *args, **kwargs):
        self._it = 0
        self._total = None

    def before_training_epoch(self, strategy, *args, **kwargs):
        if self._total is None:
            per_epoch = max(1, len(strategy.dataloader))
            self._total = max(1, int(strategy.train_epochs) * per_epoch)
            self._warm = int(self.warmup_epochs * per_epoch)

    def _factor(self):
        if self._warm and self._it < self._warm:
            return (self._it + 1) / self._warm
        span = max(1, self._total - self._warm)
        t = min(1.0, (self._it - self._warm) / span)
        return self.min_factor + (1 - self.min_factor) * 0.5 * (
            1 + math.cos(math.pi * t)
        )

    def before_training_iteration(self, strategy, *args, **kwargs):
        base = strategy.optimizer.defaults["lr"]
        f = self._factor()
        for g in strategy.optimizer.param_groups:
            g["lr"] = base * f
        self._it += 1

    def after_training_exp(self, strategy, *args, **kwargs):
        base = strategy.optimizer.defaults["lr"]
        for g in strategy.optimizer.param_groups:
            g["lr"] = base
