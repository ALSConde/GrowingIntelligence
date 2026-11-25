import torch
from torch.utils.data import DataLoader
from avalanche.core import SupervisedPlugin
from models.layers.LinearAttention import LinearAttention


def _eig_desc(Z):
    """Autovalores (decrescentes) e autovetores de Z Z^T, Z: (D, R) em float64 na CPU.

    Usa a matriz de covariância D x D (D = 256 ou 64) com eigh, em vez de SVD da matriz alta:
    é bem mais robusto (o SVD em GPU pode não convergir com matrizes mal condicionadas ou com
    valores singulares repetidos) e custa pouco."""
    C = Z @ Z.t()
    C = 0.5 * (C + C.t())
    try:
        ev, V = torch.linalg.eigh(C)
    except Exception:  # raro: tenta de novo com um pequeno jitter na diagonal
        jitter = 1e-10 * float(torch.trace(C).clamp_min(1e-30)) / C.shape[0]
        ev, V = torch.linalg.eigh(C + jitter * torch.eye(C.shape[0], dtype=C.dtype))
    ev = ev.flip(0).clamp_min(0.0)
    return ev, V.flip(1)


def gpm_update(U, X, energy, max_rank):
    """Acrescenta a U (D x r, ortonormal) as direções de X (R x D) que faltam para reter `energy`.

    Devolve U inalterado (sem levantar exceção) se X tiver valores não finitos ou energia nula.
    """
    dev = X.device
    if not bool(torch.isfinite(X).all()):
        print(
            "[mem-guard] AVISO: valores não finitos nas consultas/pesos; base não atualizada"
        )
        return U
    Z = X.t().double().cpu()  # (D, R)
    total = float((Z**2).sum())
    if not (total > 0 and total < float("inf")):
        return U
    Z = Z / total**0.5  # energia total = 1: evita estouro numérico em escalas extremas
    Uc = None if U is None or U.shape[1] == 0 else U.double().cpu()
    kept = 0.0
    if Uc is not None:
        proj = Uc.t() @ Z
        kept = float((proj**2).sum())
        if kept >= energy or Uc.shape[1] >= max_rank:
            return U
        Z = Z - Uc @ proj
    ev, V = _eig_desc(Z)
    tol = 1e-12 * max(float(ev[0]), 1e-300)
    n_sig = int((ev > tol).sum())  # posto numérico: não acrescenta direções de ruído
    if n_sig == 0:
        return U
    cum = kept + torch.cumsum(ev, 0)
    k = int((cum < energy).sum().item()) + 1
    budget = max_rank - (0 if Uc is None else Uc.shape[1])
    k = min(k, n_sig, budget)
    if k <= 0:
        return U
    new = V[:, :k]
    if Uc is None:
        out = new
    else:
        out, _ = torch.linalg.qr(
            torch.cat([Uc, new], dim=1)
        )  # reortonormaliza em float64
    return out.float().to(dev).contiguous()


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
            if not (bool(torch.isfinite(q).all()) and bool(torch.isfinite(mass).all())):
                print(
                    f"[mem-guard] AVISO: '{name}' produziu valores não finitos (o treino "
                    "provavelmente divergiu); proteção não atualizada nesta etapa"
                )
                entry["modules"][name] = {"skipped": "nao_finito"}
                continue
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
