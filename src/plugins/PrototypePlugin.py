import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from avalanche.core import SupervisedPlugin


class PrototypePlugin(SupervisedPlugin):
    def __init__(
        self,
        sdc: bool = True,
        sigma: float = 0.3,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        super().__init__()
        self.sdc = sdc
        self.sigma = float(sigma)
        self.batch_size = batch_size
        self.verbose = verbose
        self.history = []
        self._eb = None
        self._y = None

    @staticmethod
    def _supported(model):
        return getattr(model, "emb_dim", 0) > 0 and hasattr(model, "proto")

    @torch.no_grad()
    def _embed(self, strategy):
        model = strategy.model
        was_training = model.training
        model.eval()
        model.proto_mode = False
        ds = strategy.experience.dataset.eval()
        embs, ys = [], []
        for batch in DataLoader(ds, batch_size=self.batch_size, shuffle=False):
            x, y = batch[0].to(strategy.device), batch[1]
            model(x)
            embs.append(F.normalize(model.last_emb.detach(), dim=1).cpu())
            ys.append(y.cpu())
        model.train(was_training)
        return torch.cat(embs), torch.cat(ys)

    def before_training_exp(self, strategy, *args, **kwargs):
        model = strategy.model
        self._eb = None
        if not (self.sdc and self._supported(model)):
            return
        model._sync_proto()
        if bool(model.proto_valid.any()):
            self._eb, self._y = self._embed(strategy)

    @torch.no_grad()
    def after_training_exp(self, strategy, *args, **kwargs):
        model = strategy.model
        if not self._supported(model):
            return
        model._sync_proto()
        ea, y = self._embed(strategy)
        dev = model.proto.device
        new_classes = torch.unique(y)
        old_valid = model.proto_valid.clone()
        old_valid[new_classes.to(dev)] = (
            False 
        )

        shift = 0.0
        if self._eb is not None and bool(old_valid.any()):
            idx = torch.nonzero(old_valid).squeeze(1)
            P = F.normalize(model.proto[idx].cpu(), dim=1)  # (K, E)
            d2 = torch.cdist(P, self._eb).pow(2)  # (K, N)
            W = torch.exp(-d2 / (2 * self.sigma**2))
            den = W.sum(1, keepdim=True)
            delta = (W @ (ea - self._eb)) / den.clamp_min(1e-8)
            delta = torch.where(den > 1e-6, delta, torch.zeros_like(delta))
            newP = F.normalize(P + delta, dim=1)
            model.proto[idx] = newP.to(dev)
            shift = float(delta.norm(dim=1).mean())

        for c in new_classes.tolist():
            m = ea[y == c].mean(0)
            model.proto[c] = F.normalize(m, dim=0).to(dev)
            model.proto_valid[c] = True
        self.history.append(
            {
                "step": int(strategy.experience.current_experience),
                "new_classes": int(len(new_classes)),
                "old_classes": int(old_valid.sum()),
                "mean_sdc_shift": shift,
            }
        )
        if self.verbose:
            print(
                f"[proto] etapa {self.history[-1]['step']}: {len(new_classes)} protótipos novos, "
                f"{int(old_valid.sum())} antigos (deslocamento SDC médio {shift:.4f})"
            )
