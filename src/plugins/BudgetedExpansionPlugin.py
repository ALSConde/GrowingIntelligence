import math

import torch
from torch.utils.data import DataLoader

from avalanche.core import SupervisedPlugin


class BudgetedExpansionPlugin(SupervisedPlugin):
    """Expansão de largura com ORÇAMENTO de parâmetros, para o `ExpandableCNN`.

    Substitui a regra do WDExpansionPlugin, em que o número de neurônios novos é
    proporcional à largura atual (crescimento geométrico) e o fator de perda satura em 1.
    Aqui, no início de cada etapa t >= 1:

      1. Porta: só expande se a perda média nos dados da etapa (modelo anterior) passa de
         `loss_threshold` (igual ao plugin antigo).
      2. Orçamento: gasta no máximo `budget_per_step * P0` parâmetros, com P0 = parâmetros
         do extrator na etapa 0. O crescimento é linear e previsível, não geométrico.
      3. Alocação: o orçamento é dividido entre as camadas em proporção a um sinal de
         necessidade medido nos dados da etapa (`need_signal`):
            "rank"    razão de rank efetivo das ativações (razão de participação / largura):
                      camada cujas ativações já ocupam toda a largura está saturada;
            "usage"   fração de unidades que disparam em mais de `firing_threshold` das amostras;
            "uniform" mesma fatia para todas.
         O número de unidades de cada camada é fatia / custo da unidade, onde o custo inclui
         os pesos que a unidade recebe, a BatchNorm e as colunas que entram na camada seguinte
         (canais iniciais são baratos, canais da última conv e neurônios da fc1 são caros).
      4. Tetos: no máximo `max_growth_frac` da largura atual por etapa e
         `max_width_factor` vezes a largura inicial. O orçamento que sobra (por causa dos
         tetos) é redistribuído entre as camadas ainda com folga.

    `layers`: "all" (5 conv + 2 FC) ou "fc" (só as FC, equivalente ao WE original).
    O custo previsto ignora termos cruzados entre camadas expandidas na mesma etapa; o
    aumento real de parâmetros é registrado em `history`.
    """

    def __init__(
        self,
        layers: str = "all",
        need_signal: str = "rank",
        budget_per_step: float = 0.05,
        loss_threshold: float = 0.5,
        max_growth_frac: float = 0.25,
        max_width_factor: float = 2.0,
        stats_images: int = 2048,
        firing_threshold: float = 0.05,
        device="cpu",
        verbose: bool = True,
    ):
        super().__init__()
        assert layers in ("all", "fc") and need_signal in ("rank", "usage", "uniform")
        self.layer_ids = list(range(7)) if layers == "all" else [5, 6]
        self.need_signal = need_signal
        self.budget_per_step = budget_per_step
        self.loss_threshold = loss_threshold
        self.max_growth_frac = max_growth_frac
        self.max_width_factor = max_width_factor
        self.stats_images = stats_images
        self.firing_threshold = firing_threshold
        self.device = torch.device(device)
        self.verbose = verbose
        self.history = []
        self._p0 = None
        self._w0 = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect(self, strategy, model):
        data = strategy.experience.dataset.eval()  # sem aumento de dados
        loader = DataLoader(data, batch_size=strategy.train_mb_size, shuffle=False)
        criterion = strategy._criterion
        loss_sum, n_total, n_stats = 0.0, 0, 0
        acc = [None] * 7
        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, acts = model.forward_acts(x)
            loss_sum += float(criterion(logits, y)) * len(x)
            n_total += len(x)
            if n_stats >= self.stats_images:
                continue
            n_stats += len(x)
            for l, a in enumerate(acts):
                if (
                    a.dim() == 4
                ):  # cada posição espacial é uma amostra do vetor de canais
                    a = a.permute(0, 2, 3, 1).reshape(-1, a.shape[1])
                    if a.shape[0] > 4096:
                        a = a[torch.randperm(a.shape[0], device=a.device)[:4096]]
                a = a.double()
                if acc[l] is None:
                    d = a.shape[1]
                    acc[l] = [
                        torch.zeros(d, dtype=torch.double, device=a.device),
                        torch.zeros(d, d, dtype=torch.double, device=a.device),
                        torch.zeros(d, dtype=torch.double, device=a.device),
                        0,
                    ]
                acc[l][0] += a.sum(0)
                acc[l][1] += a.t() @ a
                acc[l][2] += (a > 0).sum(0)
                acc[l][3] += a.shape[0]
        signals = []
        for s, xtx, fire, cnt in acc:
            mean = s / cnt
            cov = xtx / cnt - torch.outer(mean, mean)
            fro2 = float((cov**2).sum())
            rank = (
                (float(torch.trace(cov)) ** 2 / fro2 / len(mean))
                if fro2 > 1e-12
                else 0.0
            )
            usage = float(((fire / cnt) > self.firing_threshold).double().mean())
            signals.append({"rank": min(rank, 1.0), "usage": usage})
        return loss_sum / max(n_total, 1), signals

    def _allocate(self, model, need):
        budget = self.budget_per_step * self._p0
        add = {l: 0 for l in self.layer_ids}
        cap = {}
        for l in self.layer_ids:
            by_frac = math.ceil(self.max_growth_frac * model.widths[l])
            by_total = int(self.max_width_factor * self._w0[l]) - model.widths[l]
            cap[l] = max(0, min(by_frac, by_total))
        remaining = budget
        for _ in range(8):
            active = [l for l in self.layer_ids if add[l] < cap[l] and need[l] > 0]
            if not active or remaining <= 0:
                break
            total = sum(need[l] for l in active)
            spent = 0
            for l in active:
                cost = model.unit_cost(l)
                k = min(int((remaining * need[l] / total) // cost), cap[l] - add[l])
                if k > 0:
                    add[l] += k
                    spent += k * cost
            if spent == 0:
                break
            remaining -= spent
        return add, budget, budget - remaining

    # ------------------------------------------------------------------
    def before_training_exp(self, strategy, *args, **kwargs):
        model = strategy.model
        step = strategy.experience.current_experience
        if self._p0 is None:
            self._p0 = model.extractor_params()
            self._w0 = list(model.widths)
            self.history.append(
                {
                    "step": step,
                    "expanded": False,
                    "widths": list(model.widths),
                    "extractor_params": self._p0,
                }
            )
        if step == 0:
            return

        flags = {m: m.training for m in model.modules()}
        model.eval()
        avg_loss, signals = self._collect(strategy, model)
        p_before = model.extractor_params()
        entry = {
            "step": step,
            "avg_loss": avg_loss,
            "signals": signals,
            "expanded": False,
        }
        if avg_loss > self.loss_threshold:
            need = {
                l: (
                    1.0
                    if self.need_signal == "uniform"
                    else signals[l][self.need_signal]
                )
                for l in self.layer_ids
            }
            add, budget, planned = self._allocate(model, need)
            for l, n in add.items():
                model.expand(l, n)
            if any(add.values()):
                opt = strategy.optimizer
                strategy.optimizer = type(opt)(model.parameters(), **opt.defaults)
            entry.update(
                {
                    "expanded": any(add.values()),
                    "units_added": add,
                    "budget": budget,
                    "params_planned": int(planned),
                }
            )
            if self.verbose:
                shown = {model.NAMES[l]: n for l, n in add.items() if n}
                print(
                    f"[expansão] etapa {step}: perda {avg_loss:.3f}, unidades novas {shown}, "
                    f"orçamento {budget:,.0f} (previsto {planned:,.0f})"
                )
        entry["widths"] = list(model.widths)
        entry["extractor_params"] = model.extractor_params()
        entry["params_added_actual"] = entry["extractor_params"] - p_before
        self.history.append(entry)
        for m, f in flags.items():
            m.training = f
