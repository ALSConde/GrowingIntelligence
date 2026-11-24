import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward


@contextlib.contextmanager
def suppress_wd_stats(enabled: bool = True):
    if not enabled:
        yield
        return
    try:
        from models.layers.WDStats import WDStats
    except Exception:
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
        self.curve = []
        self.epoch_log = []
        self.steps = []
        self._ep = 0
        self._step = 0
        self._reset_epoch_acc()

    # ------------------------------------------------------------------
    def _reset_epoch_acc(self):
        self._loss_sum, self._correct, self._n = 0.0, 0, 0

    @torch.no_grad()
    def _predict(self, model, suppress: bool, head=None):
        flags = {m: m.training for m in model.modules()}
        model.eval()
        preds = []
        if head is not None and hasattr(model, "proto_mode"):
            model.proto_mode = head == "ncm"
        try:
            with suppress_wd_stats(suppress):
                for i in range(0, len(self.X), self.batch_size):
                    x = self.X[i : i + self.batch_size].to(self.device)
                    t = torch.zeros(len(x), dtype=torch.long, device=self.device)
                    out = avalanche_forward(model, x, t)
                    preds.append(out.argmax(1).cpu())
        finally:
            if hasattr(model, "proto_mode"):
                model.proto_mode = False
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
        sup = not self.legacy_stats
        has_proto = getattr(model, "emb_dim", 0) > 0 and hasattr(model, "proto_valid")
        extra = {}
        if has_proto:
            m_lin = self._metrics(self._predict(model, sup, head="linear"))
            m_ncm = self._metrics(self._predict(model, sup, head="ncm"))
            m = m_ncm if getattr(model, "report_head", "linear") == "ncm" else m_lin
            for h, mm in (("linear", m_lin), ("ncm", m_ncm)):
                extra[f"acc_seen_{h}"] = mm["acc_seen"]
                extra[f"per_exp_{h}"] = mm["per_exp"]
                extra[f"per_class_{h}"] = mm["per_class"]
        else:
            m = self._metrics(self._predict(model, suppress=sup))
        row = {
            "step": step,
            "params_before": params_before,
            "params_after": params_after,
            "seconds": seconds,
            "model": describe_model(model),
            **m,
            **extra,
        }
        self.steps.append(row)
        self.curve.append({"step": step, "epoch": epochs, **m})
        return row

    # ------------------------------------------------------------------ métricas finais
    def run_summary(self):
        T = len(self.steps)
        A = [s["acc_seen"] for s in self.steps]
        a = [s["per_exp"] for s in self.steps]
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

        # --- cabeças alternativas (linear x protótipos), quando o modelo tem embedding -------
        for h in ("linear", "ncm"):
            if f"acc_seen_{h}" not in self.steps[0]:
                continue
            Ah = [s[f"acc_seen_{h}"] for s in self.steps]
            ah = [s[f"per_exp_{h}"] for s in self.steps]
            out[f"final_acc_seen_{h}"] = Ah[-1]
            out[f"avg_incremental_acc_{h}"] = float(np.mean(Ah))
            out[f"per_class_final_{h}"] = self.steps[-1][f"per_class_{h}"]
            out[f"forgetting_{h}"] = (
                float(
                    np.mean(
                        [
                            max(ah[t][j] for t in range(j, T - 1)) - ah[T - 1][j]
                            for j in range(T - 1)
                        ]
                    )
                )
                if T > 1
                else None
            )

        # --- per class ---------------------------------------------------------------
        pc = [s["per_class"] for s in self.steps]
        n = self.n_classes
        learned_step = [None] * n
        for j, classes in enumerate(self.exp_classes):
            for c in classes:
                learned_step[c] = j
        acc_learned, forg, drop = [None] * n, [None] * n, [None] * n
        for c, j in enumerate(learned_step):
            if j is None or j >= T:
                continue
            acc_learned[c] = pc[j][c]
            if j <= T - 2:
                forg[c] = max(pc[t][c] for t in range(j, T - 1)) - pc[T - 1][c]
                drop[c] = pc[j][c] - pc[T - 1][c]
        out["class_learned_step"] = learned_step
        out["per_class_learned"] = acc_learned
        out["per_class_forgetting"] = forg
        out["per_class_drop"] = drop
        return out
