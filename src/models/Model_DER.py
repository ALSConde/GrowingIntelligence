import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models


class _SoftMask(torch.autograd.Function):
    """Forward: m = sigmoid(s * e). Backward: derivada de sigmoid(e), isto é, o gradiente
    compensado da Eq. 9 do DER (remove a influência do fator de escala s)."""

    @staticmethod
    def forward(ctx, e, s):
        ctx.save_for_backward(e)
        return torch.sigmoid(s * e)

    @staticmethod
    def backward(ctx, grad_out):
        (e,) = ctx.saved_tensors
        sig = torch.sigmoid(e)
        return grad_out * sig * (1.0 - sig), None


class ChannelMask(nn.Module):
    """Máscara de canais aprendível (DER, Sec. 3.3, adaptada do HAT).

    treino:      m = sigmoid(s * e), com s crescendo de 1/smax a smax em cada época
    avaliação:   m binária (e > 0) enquanto a máscara não foi congelada
    congelada:   m = keep (buffer binário fixo)

    O parâmetro treinável é u, com e = scale * u. Com Adam o passo de u é ~lr por
    iteração independentemente da escala do gradiente, então `scale` (padrão 10) é o que
    permite ao embedding e atravessar zero em algumas centenas de passos (o DER original
    usa SGD, onde o gradiente maior produzia esse efeito).
    """

    def __init__(self, n: int, init: float = 1.0, scale: float = 10.0):
        super().__init__()
        self.init = init
        self.scale = scale
        self.u = nn.Parameter(torch.full((n,), float(init) / scale))
        self.register_buffer("keep", torch.ones(n))
        self.frozen = False
        self.s = 1.0

    @property
    def e(self):
        return self.scale * self.u

    def reset(self):
        with torch.no_grad():
            self.u.fill_(self.init / self.scale)
            self.keep.fill_(1.0)
        self.u.requires_grad_(True)
        self.frozen = False

    def current(self):
        if self.frozen:
            return self.keep
        if self.training:
            return _SoftMask.apply(self.e, self.s)
        return (self.e > 0).to(self.u.dtype)

    def freeze(self):
        with torch.no_grad():
            keep = (self.e > 0).to(self.keep.dtype)
            if keep.sum() == 0:  # nunca poda a camada inteira
                keep[self.e.argmax()] = 1.0
            self.keep.copy_(keep)
        self.u.requires_grad_(False)
        self.frozen = True

    def kept(self) -> int:
        if self.frozen:
            return int(self.keep.sum().item())
        return int((self.e > 0).sum().item())

    def forward(self, x):
        m = self.current()
        return x * m.view(1, -1, *([1] * (x.dim() - 2)))


class ExpertCNN(nn.Module):
    """Extrator de características do artigo (mesma arquitetura de Model_CNN_Cifar,
    sem o classificador): 5 blocos conv+BN+ReLU, flatten (256*4*4), FC(400), FC(400).
    Cada etapa do DER/Orth-DER cria uma cópia deste bloco inteiro.

    Com prune=True há uma máscara de canais após a ReLU de cada uma das 7 camadas
    (5 conv + 2 FC); a poda é "virtual": canais podados ficam zerados para sempre e
    a contagem de parâmetros efetivos (effective_params) só conta o que sobra."""

    feat_dim = 400
    DIMS = (16, 32, 64, 128, 256, 400, 400)

    def __init__(self, prune: bool = False, mask_init: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 400)
        self.fc2 = nn.Linear(400, 400)
        self.masks = (
            nn.ModuleList([ChannelMask(d, mask_init) for d in self.DIMS])
            if prune
            else None
        )

    def _m(self, i, x):
        return x if self.masks is None else self.masks[i](x)

    def forward(self, x):
        x = self._m(0, F.relu(self.batchnorm1(self.conv1(x))))
        x = self._m(1, F.relu(self.batchnorm2(self.conv2(x))))
        x = self._m(2, F.relu(self.batchnorm3(self.conv3(x))))
        x = self._m(3, F.relu(self.batchnorm4(self.conv4(x))))
        x = self._m(4, F.relu(self.batchnorm5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self._m(5, F.relu(self.fc1(x)))
        x = self._m(6, F.relu(self.fc2(x)))
        return x

    # ---- poda ----
    def set_scale(self, s: float):
        if self.masks is not None:
            for m in self.masks:
                m.s = s

    def reset_masks(self):
        if self.masks is not None:
            for m in self.masks:
                m.reset()

    def freeze_masks(self):
        if self.masks is not None:
            for m in self.masks:
                m.freeze()

    def kept(self):
        if self.masks is None:
            return list(self.DIMS)
        return [m.kept() for m in self.masks]

    def effective_params(self) -> int:
        k = self.kept()
        cin = [3] + k[:4]
        p = 0
        for i in range(5):  # conv (bias) + BatchNorm (peso e bias)
            p += k[i] * cin[i] * 9 + k[i] + 2 * k[i]
        p += k[5] * (16 * k[4]) + k[5]  # fc1
        p += k[6] * k[5] + k[6]  # fc2
        return p

    def sparsity_loss(self):
        n = [m.current().sum() for m in self.masks]
        ins = [3.0, n[0], n[1], n[2], n[3], 16.0 * n[4], n[5]]
        ks = [9, 9, 9, 9, 9, 1, 1]
        full_in = [3, 16, 32, 64, 128, 4096, 400]
        used = sum(k * i * o for k, i, o in zip(ks, ins, n))
        full = float(sum(k * i * o for k, i, o in zip(ks, full_in, self.DIMS)))
        return used / full


class DERNet(models.DynamicModule):
    """Representação expansível no estilo DER (Yan et al., CVPR 2021) sobre a CNN do artigo.

    - A cada nova etapa (experiência de treino) um novo ExpertCNN é adicionado,
      inicializado com os pesos do anterior; os anteriores são congelados
      (requires_grad=False e BatchNorm em eval).
    - O classificador único recebe a concatenação das características de todos os
      extratores (Class-IL padrão, sem task-id na inferência).
    - Um classificador auxiliar (n_novas + 1 saídas, classes antigas fundidas em
      uma só) supervisiona apenas o extrator mais novo.
    - A cabeça mascara classes ainda não vistas (mesma convenção do
      IncrementalClassifier do Avalanche).
    - prune=True ativa a poda por máscara de canais do DER (use junto com
      DERStrategy, que aprende as máscaras e as congela ao fim de cada etapa).

    Premissa: as classes chegam contíguas (fixed_class_order=range(N)).
    """

    def __init__(
        self,
        initial_out_features: int = 10,
        mask_value: float = -1000,
        init_from_previous: bool = True,
        prune: bool = False,
    ):
        super().__init__()
        self.mask_value = mask_value
        self.init_from_previous = init_from_previous
        self.prune = prune

        self.experts = nn.ModuleList([ExpertCNN(prune=prune)])
        self.fc = nn.Linear(ExpertCNN.feat_dim, initial_out_features)
        self.aux_fc = None
        self.register_buffer(
            "active_units", torch.zeros(initial_out_features, dtype=torch.int8)
        )
        # etapa (= índice do extrator) em que cada classe foi aprendida; -1 = não vista
        self.register_buffer(
            "class_task", torch.full((initial_out_features,), -1, dtype=torch.long)
        )

        self.known_classes = 0  # classes aprendidas ANTES da etapa corrente
        self._steps_seen = 0
        self._last_train_exp = -1

    # ------------------------------------------------------------------
    # Adaptação dinâmica (chamada pelo Avalanche antes de cada experiência)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def adaptation(self, experience):
        super().adaptation(experience)
        device = self._adaptation_device
        curr = sorted(experience.classes_in_this_experience)

        # a cabeça cresce também na avaliação (para cobrir classes do test_stream)
        self._ensure_width(max(curr) + 1, device)

        # a expansão só acontece no treino, e uma única vez por experiência
        if not self.training:
            return
        exp_id = experience.current_experience
        if exp_id <= self._last_train_exp:
            return
        self._last_train_exp = exp_id

        self.known_classes = int(self.active_units.sum().item())
        expected = list(range(self.known_classes, self.known_classes + len(curr)))
        if curr != expected:
            raise ValueError(
                "DERNet assume classes contíguas por etapa (use fixed_class_order="
                f"range(N)); recebido {curr}, esperado {expected}."
            )
        self.active_units[curr] = 1
        self.class_task[curr] = self._steps_seen

        if self._steps_seen > 0:
            self._add_expert(len(curr), device)
        self._steps_seen += 1

    def _ensure_width(self, n_cls: int, device):
        old_n = self.fc.out_features
        if n_cls <= old_n:
            return
        new_fc = nn.Linear(self.fc.in_features, n_cls).to(device)
        new_fc.weight[:old_n] = self.fc.weight
        new_fc.bias[:old_n] = self.fc.bias
        self.fc = new_fc
        au = torch.zeros(n_cls, dtype=torch.int8, device=device)
        au[:old_n] = self.active_units
        self.active_units = au
        ct = torch.full((n_cls,), -1, dtype=torch.long, device=device)
        ct[:old_n] = self.class_task
        self.class_task = ct

    def _add_expert(self, n_new: int, device):
        prev = self.experts[-1]
        prev.freeze_masks()
        prev.eval()
        prev.requires_grad_(False)

        new = ExpertCNN(prune=self.prune).to(device)
        if self.init_from_previous:
            new.load_state_dict(prev.state_dict())
            new.reset_masks()  # máscaras novas (todos os canais ligados)
        self.experts.append(new)

        old_fc = self.fc
        fc = nn.Linear(ExpertCNN.feat_dim * len(self.experts), old_fc.out_features)
        fc = fc.to(device)
        fc.weight[:, : old_fc.in_features] = old_fc.weight
        fc.bias.copy_(old_fc.bias)
        self.fc = fc

        self.aux_fc = nn.Linear(ExpertCNN.feat_dim, n_new + 1).to(device)

    # ------------------------------------------------------------------
    # Poda
    # ------------------------------------------------------------------
    def set_mask_scale(self, s: float):
        self.experts[-1].set_scale(s)

    def finalize_masks(self):
        """Binariza e congela as máscaras do extrator mais novo (fim da etapa)."""
        self.experts[-1].freeze_masks()

    def sparsity_loss(self):
        return self.experts[-1].sparsity_loss()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        for e in list(self.experts)[:-1]:  # extratores congelados: BN sempre em eval
            e.eval()
        return self

    def head(self, feats):
        out = self.fc(feats)
        return out.masked_fill(torch.logical_not(self.active_units), self.mask_value)

    def forward_all(self, x):
        feats = []
        n = len(self.experts)
        for i, e in enumerate(self.experts):
            if i < n - 1:
                with torch.no_grad():
                    feats.append(e(x))
            else:
                feats.append(e(x))
        f_new = feats[-1]
        logits = self.head(torch.cat(feats, dim=1))
        aux = (
            self.aux_fc(f_new) if (self.aux_fc is not None and self.training) else None
        )
        return {"logits": logits, "aux": aux, "feat_new": f_new, "feats": feats}

    def forward(self, x):
        return self.forward_all(x)["logits"]

    @torch.no_grad()
    def features(self, x):
        return torch.cat([e(x) for e in self.experts], dim=1)

    @torch.no_grad()
    def old_features(self, x):
        return torch.cat([e(x) for e in list(self.experts)[:-1]], dim=1)

    # ------------------------------------------------------------------
    # Contagem de parâmetros
    # ------------------------------------------------------------------
    def count_params(self, effective: bool = True) -> int:
        """effective=True: parâmetros de inferência após a poda (extratores podados +
        classificador sobre as características que sobram; sem o auxiliar nem as
        máscaras, que só existem no treino). effective=False: todos os parâmetros
        alocados no torch."""
        if not effective:
            return sum(p.numel() for p in self.parameters())
        n_cls = int(self.active_units.sum().item())
        feat = sum(e.kept()[6] for e in self.experts)
        return sum(e.effective_params() for e in self.experts) + n_cls * feat + n_cls

    def prune_report(self) -> dict:
        return {
            "kept_channels": [e.kept() for e in self.experts],
            "feature_dims": [e.kept()[6] for e in self.experts],
            "params_effective": self.count_params(True),
            "params_allocated": self.count_params(False),
        }
