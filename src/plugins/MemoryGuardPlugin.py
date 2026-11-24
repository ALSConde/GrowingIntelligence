import torch
from torch.utils.data import DataLoader
from avalanche.core import SupervisedPlugin
from models.layers.LinearAttention import LinearAttention


def gpm_update(U, X, energy, max_rank):
    """Acrescenta a U (D x r, ortonormal) as direções de X (R x D) que faltam para reter `energy`."""
    X = X.t().float()  # (D, R)
    total = (X**2).sum()
    if float(total) <= 0:
        return U
    if U is None or U.shape[1] == 0:
        Uu, S, _ = torch.linalg.svd(X, full_matrices=False)
        cum = torch.cumsum(S**2, 0) / total
        k = min(int((cum < energy).sum().item()) + 1, max_rank, Uu.shape[1])
        return Uu[:, :k].contiguous()
    proj = U.t() @ X
    kept = (proj**2).sum()
    if float(kept / total) >= energy or U.shape[1] >= max_rank:
        return U
    Uu, S, _ = torch.linalg.svd(X - U @ proj, full_matrices=False)
    cum = (kept + torch.cumsum(S**2, 0)) / total
    k = min(int((cum < energy).sum().item()) + 1, max_rank - U.shape[1], Uu.shape[1])
    Q, _ = torch.linalg.qr(torch.cat([U, Uu[:, :k]], dim=1))
    return Q.contiguous()


class MemoryGuardPlugin(SupervisedPlugin):
    def __init__(
        self,
        mode: str = "both",
        energy: float = 0.95,
        beta: float = 4.0,
        max_rank_frac: float = 0.9,
        max_samples: int = 1024,
        max_rows: int = 4096,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        super().__init__()
        assert mode in ("importance", "projection", "both")
        self.use_imp = mode in ("importance", "both")
        self.use_proj = mode in ("projection", "both")
        self.energy = energy
        self.beta = beta
        self.max_rank_frac = max_rank_frac
        self.max_samples = max_samples
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.verbose = verbose
        self.U_q, self.U_a, self.imp = {}, {}, {}
        self.history = []
        self._snap = {}

    @staticmethod
    def _slot_modules(model):
        return [
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, LinearAttention) and m.mem_mode == "slots"
        ]

    def _scale(self, name, M, dev):
        imp = self.imp.get(name)
        if not self.use_imp or imp is None or float(imp.max()) <= 0:
            return torch.ones(M, device=dev)
        return torch.exp(-self.beta * imp.to(dev) / imp.max().to(dev))

    @torch.no_grad()
    def after_training_exp(self, strategy, *args, **kwargs):
        model = strategy.model
        mods = self._slot_modules(model)
        if not mods:
            return
        for _, m in mods:
            m._collect = {"q": [], "w": [], "mass": []}
        was_training = model.training
        model.eval()
        seen = 0
        for batch in DataLoader(
            strategy.experience.dataset.eval(), batch_size=self.batch_size, shuffle=True
        ):
            model(batch[0].to(strategy.device))
            seen += len(batch[0])
            if seen >= self.max_samples:
                break
        model.train(was_training)
        entry = {"step": int(strategy.experience.current_experience), "modules": {}}
        for name, m in mods:
            col, m._collect = m._collect, None
            q, w, mass = (torch.cat(col[k]) for k in ("q", "w", "mass"))
            if q.size(0) > self.max_rows:
                sel = torch.randperm(q.size(0), device=q.device)[: self.max_rows]
                q, w, mass = q[sel], w[sel], mass[sel]
            M, d = m.k_mem.shape
            self.imp[name] = self.imp.get(
                name, torch.zeros(M, device=q.device)
            ) + mass.mean(0)
            if self.use_proj:
                self.U_q[name] = gpm_update(
                    self.U_q.get(name), q, self.energy, int(self.max_rank_frac * d)
                )
                self.U_a[name] = gpm_update(
                    self.U_a.get(name), w, self.energy, int(self.max_rank_frac * M)
                )
            entry["modules"][name] = {
                "rank_q": int(self.U_q[name].shape[1]) if name in self.U_q else 0,
                "rank_a": int(self.U_a[name].shape[1]) if name in self.U_a else 0,
                "mass_total": float(mass.sum(1).mean()),
                "min_slot_scale": float(self._scale(name, M, q.device).min()),
            }
        self.history.append(entry)
        if self.verbose:
            print(f"[mem-guard] etapa {entry['step']}: {entry['modules']}")

    # ---- aplica a proteção ao passo do otimizador
    def before_update(self, strategy, *args, **kwargs):
        self._snap = {}
        if not (self.imp or self.U_q):
            return
        for name, m in self._slot_modules(strategy.model):
            if name in self.imp:
                self._snap[name] = (
                    m,
                    m.k_mem.detach().clone(),
                    m.v_mem.detach().clone(),
                )

    @torch.no_grad()
    def after_update(self, strategy, *args, **kwargs):
        for name, (m, kp, vp) in self._snap.items():
            dk, dv = m.k_mem - kp, m.v_mem - vp
            if self.use_imp:
                s = self._scale(name, dk.size(0), dk.device).unsqueeze(1)
                dk, dv = dk * s, dv * s
            if self.use_proj:
                Uq, Ua = self.U_q.get(name), self.U_a.get(name)
                if Uq is not None:
                    dk = dk - (dk @ Uq) @ Uq.t()
                if Ua is not None:
                    dv = dv - Ua @ (Ua.t() @ dv)
            m.k_mem.copy_(kp + dk)
            m.v_mem.copy_(vp + dv)
        self._snap = {}
