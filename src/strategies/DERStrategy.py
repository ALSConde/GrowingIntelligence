from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


class DERStrategy(SupervisedTemplate):
    """DER (Yan et al., 2021), com poda opcional, e Orth-DER (Dong et al., 2026) sobre o
    `DERNet`.

    Ambos usam memória de exemplares: o treino é em D_t ∪ M_t e o ajuste fino do
    classificador usa amostras balanceadas de memória.

    DER:       L = CE + aux_weight * CE_aux                       (lambda_s = lambda_i = 0)
    Orth-DER:  L = CE + lambda_s * L_s + lambda_i * (L_in + L_io)  (aux_weight = 0)
    Poda (se o DERNet foi criado com prune=True), somada nos dois casos:
               + prune_lambda * L_S   (Eq. 10 do DER, esparsidade das máscaras)

    Poda: as máscaras do extrator novo são aprendidas junto com a representação, com o
    fator s anulado de 1/smax a smax dentro de cada época (Eq. 8); ao fim da etapa são
    binarizadas e congeladas (`finalize_masks`) ANTES do ajuste fino do classificador.

    Orth-DER, equações do paper (médias de produto interno por classe no lote = produto
    interno das médias de classe):
      L_s  = Σ_j Σ_{k,m} w_{k,m} (μ^t_k · μ^j_m)^2    novas sob F_t vs antigas sob o F_j da sua tarefa
      L_in = Σ_{k≠l} (μ^t_k · μ^t_l)^2                 entre classes novas, sob F_t
      L_io = Σ_k Σ_m (μ^t_k · ν^t_m)^2                 novas vs antigas (da memória), ambas sob F_t
      w = softmax por linha (temperatura tau) da similaridade de cosseno entre protótipos
          (TRM): pares mais parecidos recebem MAIS peso.
    Suposições não especificadas no paper (configuráveis): características normalizadas
    em L2 (orth_normalize), soma sobre pares (orth_reduce="sum"), softmax da TRM sobre
    todas as classes antigas, TRM calculada numa passada prévia, exemplares aleatórios
    em vez de herding.

    Ajuste fino do classificador (DER Sec. 3.4): o classificador é REINICIALIZADO e treinado
    com extratores congelados em amostras balanceadas, SGD lr 0.1 com queda ×0.1 na
    metade, wd 5e-4, CE com temperatura ft_temp (δ=5 no CIFAR-100). Momentum 0.9 é suposição.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_per_class: int = 100,
        total_classes: int = 100,
        aux_weight: float = 1.0,
        lambda_s: float = 0.0,
        lambda_i: float = 0.0,
        tau: float = 0.07,
        use_trm: bool = True,
        orth_normalize: bool = True,
        orth_reduce: str = "sum",
        prune_lambda: float = 1.0,
        smax: float = 400.0,
        ft_epochs: int = 30,
        ft_lr: float = 0.1,
        ft_wd: float = 5e-4,
        ft_temp: float = 5.0,
        train_mb_size: int = 256,
        train_epochs: int = 30,
        eval_mb_size: Optional[int] = 128,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs,
        )
        assert orth_reduce in ("sum", "mean")
        self.prune = bool(getattr(model, "prune", False))
        self.prune_lambda = prune_lambda
        self.smax = smax
        self.aux_weight = aux_weight
        self.lambda_s = lambda_s
        self.lambda_i = lambda_i
        self.tau = tau
        self.use_trm = use_trm
        self.orth_normalize = orth_normalize
        self.orth_reduce = orth_reduce
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.ft_wd = ft_wd
        self.ft_temp = ft_temp

        # mesma política de memória do ER: fixa, mem_per_class por classe
        self.storage_policy = ClassBalancedBuffer(
            max_size=mem_per_class * total_classes,
            adaptive_size=False,
            total_num_classes=total_classes,
        )
        self._n_updates = 0
        self._epoch_it = 0
        self._aux_logits = None
        self._feats_all = None
        self._trm = None
        self.param_history: List[dict] = []

    # ------------------------------------------------------------------
    # Dados: treino em D_t ∪ memória
    # ------------------------------------------------------------------
    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        if self._n_updates > 0:
            self.adapted_dataset = concat_datasets(
                [self.adapted_dataset, self.storage_policy.buffer.train()]
            ).train()

    def _before_training_exp(self, **kwargs):
        super()._before_training_exp(**kwargs)  # adapta o modelo (adiciona extrator)
        self._trm = None
        if self.model.known_classes > 0 and self.lambda_s > 0 and self.use_trm:
            self._trm = self._compute_trm()

    def _before_training_epoch(self, **kwargs):
        self._epoch_it = 0  # contador de iterações dentro da época (para o Eq. 8)
        super()._before_training_epoch(**kwargs)

    def _before_training_iteration(self, **kwargs):
        if self.prune:
            # Eq. 8: s cresce linearmente de 1/smax a smax dentro de cada época
            n_batches = max(len(self.dataloader), 2)
            s = 1.0 / self.smax + (self.smax - 1.0 / self.smax) * self._epoch_it / (
                n_batches - 1
            )
            self.model.set_mask_scale(min(s, self.smax))
        self._epoch_it += 1
        super()._before_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        if self.prune:
            self.model.finalize_masks()  # binariza e congela ANTES do ajuste fino
        self.storage_policy.update(self, **kwargs)
        self._n_updates += 1
        if self.model.known_classes > 0:
            self._balanced_finetune()
        rep = self.model.prune_report()
        rep["step"] = self.experience.current_experience
        self.param_history.append(rep)
        print(
            f"[DER] etapa {rep['step']}: extratores={len(self.model.experts)}, "
            f"parâmetros efetivos={rep['params_effective']:,} "
            f"(alocados {rep['params_allocated']:,}), "
            f"canais mantidos no extrator novo={rep['kept_channels'][-1]}"
        )
        super()._after_training_exp(**kwargs)

    # ------------------------------------------------------------------
    # Forward e perda
    # ------------------------------------------------------------------
    def forward(self):
        out = self.model.forward_all(self.mb_x)
        self._aux_logits = out["aux"]
        self._feats_all = out["feats"]
        return out["logits"]

    def criterion(self):
        ce = self._criterion(self.mb_output, self.mb_y)
        if not self.is_training:
            return ce

        loss = ce
        if self.prune and self.prune_lambda > 0:
            loss = loss + self.prune_lambda * self.model.sparsity_loss()

        known = self.model.known_classes
        if known == 0:
            return loss

        y = self.mb_y
        if self.aux_weight > 0 and self._aux_logits is not None:
            aux_y = torch.where(y >= known, y - known + 1, torch.zeros_like(y))
            loss = loss + self.aux_weight * F.cross_entropy(self._aux_logits, aux_y)
        if self.lambda_s > 0 or self.lambda_i > 0:
            l_s, l_in, l_io = self._orth_losses(y, known)
            loss = loss + self.lambda_s * l_s + self.lambda_i * (l_in + l_io)
        return loss

    # ------------------------------------------------------------------
    # Orth-DER
    # ------------------------------------------------------------------
    @staticmethod
    def _class_means(feats, y):
        """Média das características por classe dentro do lote -> (ids, médias)."""
        cls, inv = torch.unique(y, return_inverse=True)
        onehot = F.one_hot(inv, num_classes=len(cls)).to(feats.dtype)
        return cls, (onehot.t() @ feats) / onehot.sum(0).unsqueeze(1)

    def _red(self, x):
        if x.numel() == 0:
            return x.new_zeros(())
        return x.sum() if self.orth_reduce == "sum" else x.mean()

    def _orth_losses(self, y, known):
        feats = [
            F.normalize(f, dim=1) if self.orth_normalize else f for f in self._feats_all
        ]
        f_t = feats[-1]
        zero = f_t.new_zeros(())
        is_new = y >= known
        if not is_new.any() or is_new.all():
            return zero, zero, zero
        old = ~is_new

        cls_new, mu_new = self._class_means(f_t[is_new], y[is_new])
        _, mu_old_t = self._class_means(f_t[old], y[old])

        # L_in: entre classes novas (k != l), sob F_t
        g = mu_new @ mu_new.t()
        off_diag = ~torch.eye(len(cls_new), dtype=torch.bool, device=g.device)
        l_in = self._red(g[off_diag] ** 2)

        # L_io: novas vs antigas (amostras de memória do lote), ambas sob F_t
        l_io = self._red((mu_new @ mu_old_t.t()) ** 2)

        # L_s: novas sob F_t vs antigas sob o extrator F_j da tarefa de cada uma
        task_of = self.model.class_task[y]
        terms = []
        for j in range(len(feats) - 1):
            sel = old & (task_of == j)
            if not sel.any():
                continue
            cls_j, mu_j = self._class_means(feats[j][sel], y[sel])
            sq = (mu_new @ mu_j.t()) ** 2
            if self._trm is not None:
                sq = sq * self._trm[cls_new[:, None], cls_j[None, :]]
            terms.append(sq.reshape(-1))
        l_s = self._red(torch.cat(terms)) if terms else zero
        return l_s, l_in, l_io

    @torch.no_grad()
    def _compute_trm(self):
        """TRM (Eqs. 1-2): para cada tarefa antiga j, cosseno entre o protótipo de cada
        classe nova e o de cada classe antiga de j, ambos sob F_j (antigas: amostras da
        memória); softmax por linha com temperatura tau sobre todas as classes antigas.
        """
        model = self.model
        flags = {m: m.training for m in model.modules()}
        model.eval()
        n_old_exp = len(model.experts) - 1
        n = model.fc.out_features
        d = type(model.experts[-1]).feat_dim
        known = model.known_classes
        dev = self.device

        new_sum = torch.zeros(n_old_exp, n, d, device=dev)
        old_sum = torch.zeros(n_old_exp, n, d, device=dev)
        new_cnt = torch.zeros(n, device=dev)
        old_cnt = torch.zeros(n, device=dev)

        for x, y, *_ in DataLoader(self.experience.dataset.eval(), batch_size=256):
            x, y = x.to(dev), y.to(dev)
            new_cnt.index_add_(0, y, torch.ones_like(y, dtype=new_cnt.dtype))
            for j in range(n_old_exp):
                new_sum[j].index_add_(0, y, model.experts[j](x))

        for x, y, *_ in DataLoader(self.storage_policy.buffer.eval(), batch_size=256):
            x, y = x.to(dev), y.to(dev)
            old_cnt.index_add_(0, y, torch.ones_like(y, dtype=old_cnt.dtype))
            task_of = model.class_task[y]
            for j in range(n_old_exp):
                m = task_of == j
                if m.any():
                    old_sum[j].index_add_(0, y[m], model.experts[j](x[m]))

        new_cls = torch.arange(known, n, device=dev)
        new_cls = new_cls[new_cnt[new_cls] > 0]
        sim = torch.full((n, n), float("-inf"), device=dev)
        for j in range(n_old_exp):
            cols = (model.class_task == j).nonzero().squeeze(1)
            cols = cols[old_cnt[cols] > 0]
            if len(cols) == 0:
                continue
            pn = F.normalize(new_sum[j][new_cls] / new_cnt[new_cls].unsqueeze(1), dim=1)
            po = F.normalize(old_sum[j][cols] / old_cnt[cols].unsqueeze(1), dim=1)
            sim[new_cls[:, None], cols[None, :]] = pn @ po.t()

        trm = torch.zeros(n, n, device=dev)
        trm[new_cls] = torch.softmax(sim[new_cls] / self.tau, dim=1)
        for m, f in flags.items():
            m.training = f
        return trm

    # ------------------------------------------------------------------
    # Ajuste fino do classificador na memória balanceada (DER / Orth-DER)
    # ------------------------------------------------------------------
    def _balanced_finetune(self):
        if self.ft_epochs <= 0:
            return
        model = self.model
        flags = {m: m.training for m in model.modules()}
        model.eval()
        loader = DataLoader(
            self.storage_policy.buffer.eval(),
            batch_size=self.train_mb_size,
            shuffle=True,
        )
        # DER (Sec. 3.4): o classificador é reinicializado com pesos aleatórios e treinado
        # só com o subconjunto balanceado (a memória, já balanceada por classe)
        model.fc.reset_parameters()
        opt = torch.optim.SGD(
            model.fc.parameters(), lr=self.ft_lr, momentum=0.9, weight_decay=self.ft_wd
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[max(1, self.ft_epochs // 2)], gamma=0.1
        )
        for _ in range(self.ft_epochs):
            for x, y, *_ in loader:
                x, y = x.to(self.device), y.to(self.device)
                feats = model.features(x)
                loss = F.cross_entropy(model.head(feats) / self.ft_temp, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        for m, f in flags.items():
            m.training = f
