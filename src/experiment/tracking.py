"""Rastreamento de parâmetros, acurácia ao longo do treino e acurácia por classe."""

import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward


@contextlib.contextmanager
def suppress_wd_stats(enabled: bool = True):
    """WDLayer.forward registra ativações em TODA passada (inclusive em avaliação), e essas
    estatísticas alimentam o `usage_ratio` usado pelo WDExpansionPlugin para decidir a
    expansão. Passadas de avaliação não devem alterar essas estatísticas; este contexto
    desliga o registro enquanto avaliamos."""
    if not enabled:
        yield
        return
    try:
        from models.layers.WDStats import WDStats
    except Exception:  # modelo sem WDLayer
        yield
        return
    original = WDStats.record_activations
    WDStats.record_activations = lambda self, activations: None
    try:
        yield
    finally:
        WDStats.record_activations = original


def count_parameters(model) -> int:
    """Parâmetros do modelo. O DERNet informa os parâmetros de inferência após a poda."""
    if hasattr(model, "count_params"):
        return int(model.count_params())
    return int(sum(p.numel() for p in model.parameters()))


def describe_model(model) -> dict:
    d = {"params": count_parameters(model)}
    widths = {
        n: m.out_features
        for n, m in model.named_modules()
        if type(m).__name__ == "WDLayer"
    }
    if widths:
        d["wd_widths"] = widths
    if isinstance(getattr(model, "widths", None), list):
        d["widths"] = list(model.widths)
    if hasattr(model, "prune_report"):
        d["der"] = model.prune_report()
    return d


class EvalTracker(SupervisedPlugin):
    """Avalia o modelo no conjunto de teste COMPLETO (todas as classes) sem passar pelo
    `strategy.eval`, registrando:

    - por época (a cada `eval_every` épocas e ao fim de cada etapa): acurácia nas classes
      já vistas (`acc_seen`), em todas as classes (`acc_all`), por experiência e por classe;
    - por época de treino (todas): perda total e acurácia no minibatch de treino.

    Não chama `model.adaptation`, então não altera a largura do classificador nem
    dispara expansões; as passadas de avaliação não alteram as estatísticas dos WDLayer
    (a menos que legacy_stats=True no fim da etapa, que reproduz o protocolo antigo do
    main.py em que strategy.eval(test_stream) alimentava essas estatísticas).
    """

    def __init__(
        self,
        benchmark,
        device,
        eval_every: int = 5,
        batch_size: int = 256,
        legacy_stats: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.legacy_stats = legacy_stats
        self.n_classes = int(benchmark.n_classes)
        self.exp_classes = [
            sorted(e.classes_in_this_experience) for e in benchmark.train_stream
        ]

        data = concat_datasets([e.dataset for e in benchmark.test_stream]).eval()
        xs, ys = [], []
        for x, y, *_ in DataLoader(data, batch_size=512):
            xs.append(x)
            ys.append(y)
        self.X = torch.cat(xs)
        self.Y = torch.cat(ys).long()

        self.seen = set()
        self.curve = []  # avaliações dentro do treino
        self.epoch_log = []  # perda/acurácia de treino por época
        self.steps = []  # avaliação ao fim de cada etapa
        self._ep = 0
        self._step = 0
        self._reset_epoch_acc()

    # ------------------------------------------------------------------
    def _reset_epoch_acc(self):
        self._loss_sum, self._correct, self._n = 0.0, 0, 0

    @torch.no_grad()
    def _predict(self, model, suppress: bool):
        flags = {m: m.training for m in model.modules()}
        model.eval()
        preds = []
        try:
            with suppress_wd_stats(suppress):
                for i in range(0, len(self.X), self.batch_size):
                    x = self.X[i : i + self.batch_size].to(self.device)
                    t = torch.zeros(len(x), dtype=torch.long, device=self.device)
                    out = avalanche_forward(model, x, t)
                    preds.append(out.argmax(1).cpu())
        finally:
            for m, f in flags.items():
                m.training = f
        return torch.cat(preds)

    def _metrics(self, preds):
        correct = preds == self.Y
        tot = torch.bincount(self.Y, minlength=self.n_classes).float()
        cor = torch.bincount(self.Y[correct], minlength=self.n_classes).float()
        per_class = (cor / tot.clamp(min=1)).tolist()
        seen = sorted(self.seen)
        acc_seen = 0.0
        if seen:
            mask = torch.isin(self.Y, torch.tensor(seen))
            acc_seen = correct[mask].float().mean().item()
        per_exp = [
            (cor[c].sum() / tot[c].sum().clamp(min=1)).item() for c in self.exp_classes
        ]
        return {
            "acc_seen": acc_seen,
            "acc_all": correct.float().mean().item(),
            "per_exp": per_exp,
            "per_class": per_class,
            "seen_classes": seen,
        }

    # ------------------------------------------------------------------ hooks
    def before_training_exp(self, strategy, *args, **kwargs):
        self.seen |= set(strategy.experience.classes_in_this_experience)
        self._step = strategy.experience.current_experience
        self._ep = 0
        self._reset_epoch_acc()

    def after_training_iteration(self, strategy, *args, **kwargs):
        self._loss_sum += float(strategy.loss.detach()) * len(strategy.mb_y)
        self._correct += int((strategy.mb_output.argmax(1) == strategy.mb_y).sum())
        self._n += len(strategy.mb_y)

    def after_training_epoch(self, strategy, *args, **kwargs):
        self._ep += 1
        self.epoch_log.append(
            {
                "step": self._step,
                "epoch": self._ep,
                "train_loss": self._loss_sum / max(self._n, 1),
                "train_acc": self._correct / max(self._n, 1),
            }
        )
        self._reset_epoch_acc()
        last = self._ep >= strategy.train_epochs
        if self.eval_every > 0 and self._ep % self.eval_every == 0 and not last:
            m = self._metrics(self._predict(strategy.model, suppress=True))
            self.curve.append({"step": self._step, "epoch": self._ep, **m})

    # ------------------------------------------------------------------ fim de etapa
    def end_of_step(self, model, step, epochs, params_before, params_after, seconds):
        m = self._metrics(self._predict(model, suppress=not self.legacy_stats))
        row = {
            "step": step,
            "params_before": params_before,
            "params_after": params_after,
            "seconds": seconds,
            "model": describe_model(model),
            **m,
        }
        self.steps.append(row)
        self.curve.append({"step": step, "epoch": epochs, **m})
        return row

    # ------------------------------------------------------------------ métricas finais
    def run_summary(self):
        T = len(self.steps)
        A = [s["acc_seen"] for s in self.steps]
        a = [
            s["per_exp"] for s in self.steps
        ]  # a[t][j]: acurácia na experiência j após a etapa t
        out = {
            "final_acc_seen": A[-1],
            "avg_incremental_acc": float(np.mean(A)),
            "acc_seen_per_step": A,
            "per_class_final": self.steps[-1]["per_class"],
            "params_initial": None,
            "params_final": self.steps[-1]["params_after"],
        }
        if T > 1:
            fg = [
                max(a[t][j] for t in range(j, T - 1)) - a[T - 1][j]
                for j in range(T - 1)
            ]
            bwt = [a[T - 1][j] - a[j][j] for j in range(T - 1)]
            out["forgetting"] = float(np.mean(fg))
            out["bwt"] = float(np.mean(bwt))
        else:
            out["forgetting"] = None
            out["bwt"] = None
        return out
