import torch
from avalanche.core import SupervisedPlugin


class LRDecayPlugin(SupervisedPlugin):
    def __init__(
        self,
        decay: float = 0.98,
        generation_on: str = "expansion",
        exclude=(),
        min_scale: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()
        assert 0.0 < decay <= 1.0, "decay deve estar em (0, 1]"
        assert generation_on in ("expansion", "step")
        self.decay = float(decay)
        self.generation_on = generation_on
        self.exclude = tuple(exclude)
        self.min_scale = float(min_scale)
        self.verbose = verbose
        self.generation = 0
        self.history = []
        self._scale = (
            {}
        )  # nome do parâmetro -> tensor de escalas (mesmo shape do parâmetro)
        self._params = []
        self._prev = []
        self._scales = []

    # ------------------------------------------------------------------ ligação
    def attach(self, expansion_plugin):
        """Registra este plugin para ser avisado a cada expansão efetiva."""
        if self.generation_on != "expansion":
            raise ValueError("attach só faz sentido com generation_on='expansion'")
        expansion_plugin.listeners.append(self.on_expansion)

    # ------------------------------------------------------------------ escalas
    def _tracked(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and not name.startswith(self.exclude):
                yield name, p

    @torch.no_grad()
    def _sync(self, model):
        """Garante um tensor de escalas por parâmetro, com o shape atual do parâmetro."""
        live = set()
        for name, p in self._tracked(model):
            live.add(name)
            s = self._scale.get(name)
            if s is not None and s.shape == p.shape and s.device == p.device:
                continue
            new = torch.ones_like(p, dtype=torch.float32)
            if s is not None:
                s = s.to(p.device)
                if s.dim() == p.dim() and all(a <= b for a, b in zip(s.shape, p.shape)):
                    new[tuple(slice(0, d) for d in s.shape)] = s
                else:  # shape inesperado (não é crescimento por prefixo): recomeça
                    print(
                        f"[lr-decay] aviso: '{name}' mudou de {tuple(s.shape)} para "
                        f"{tuple(p.shape)} sem ser prefixo; escala reiniciada em 1.0"
                    )
            self._scale[name] = new
        for name in [n for n in self._scale if n not in live]:
            del self._scale[name]

    @torch.no_grad()
    def _advance(self):
        """Nova geração: tudo o que já existe passa a treinar mais devagar."""
        self.generation += 1
        for s in self._scale.values():
            s.mul_(self.decay)
            if self.min_scale > 0.0:
                s.clamp_(min=self.min_scale)
        self._log()

    @torch.no_grad()
    def _log(self):
        n = sum(s.numel() for s in self._scale.values())
        if n == 0:
            return
        total = sum(float(s.sum()) for s in self._scale.values())
        lo = min(float(s.min()) for s in self._scale.values())
        entry = {
            "generation": self.generation,
            "elements": n,
            "mean_scale": total / n,
            "min_scale": lo,
        }
        self.history.append(entry)
        if self.verbose:
            print(
                f"[lr-decay] geração {self.generation}: escala média {entry['mean_scale']:.4f}, "
                f"mínima {entry['min_scale']:.4f} ({n:,} elementos rastreados)"
            )

    def on_expansion(self):
        if self.generation_on == "expansion":
            self._advance()

    # ------------------------------------------------------------------ callbacks
    def before_training(self, strategy, *args, **kwargs):
        # Antes de qualquer expansão da etapa: registra o modelo como está.
        self._sync(strategy.model)

    def before_training_exp(self, strategy, *args, **kwargs):
        if self.generation_on == "step" and strategy.experience.current_experience > 0:
            self._advance()

    def before_update(self, strategy, *args, **kwargs):
        self._sync(strategy.model)
        names, params = [], []
        for name, p in self._tracked(strategy.model):
            names.append(name)
            params.append(p)
        self._params = params
        self._scales = [self._scale[n] for n in names]
        self._prev = [p.detach().clone() for p in params]

    @torch.no_grad()
    def after_update(self, strategy, *args, **kwargs):
        for p, prev, s in zip(self._params, self._prev, self._scales):
            p.sub_(prev).mul_(s).add_(prev)  # p = prev + s * (p - prev)
        self._params, self._prev, self._scales = [], [], []
