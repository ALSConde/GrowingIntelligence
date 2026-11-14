import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models


class _SoftMask(torch.autograd.Function):
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

    # ---- pruning ----
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
        for i in range(5):
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


class ResExpert(nn.Module):
    STREAM = (16, 32, 64, 128, 256)
    STRIDES = (1, 2, 2, 2, 2)
    PADDINGS = (1, 2, 2, 2, 2)
    SPATIAL = 16

    def __init__(
        self,
        prune: bool = False,
        mask_init: float = 1.0,
        inner_ratio: float = 0.5,
        width_mult: float = 1.0,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        s_ = [max(1, int(round(b * width_mult))) for b in self.STREAM]
        fc = max(1, int(round(400 * width_mult)))
        m_ = [max(1, int(round(c * inner_ratio))) for c in s_]
        self.feat_dim = fc
        self.dims = tuple(s_) + tuple(m_) + (fc, fc)
        cin = [in_channels] + s_[:4]
        self.down = nn.ModuleList(
            [
                nn.Conv2d(
                    cin[i], s_[i], 3, stride=self.STRIDES[i], padding=self.PADDINGS[i]
                )
                for i in range(5)
            ]
        )
        self.bn_down = nn.ModuleList([nn.BatchNorm2d(s_[i]) for i in range(5)])
        self.conv_a = nn.ModuleList(
            [nn.Conv2d(s_[i], m_[i], 3, padding=1) for i in range(5)]
        )
        self.bn_a = nn.ModuleList([nn.BatchNorm2d(m_[i]) for i in range(5)])
        self.conv_b = nn.ModuleList(
            [nn.Conv2d(m_[i], s_[i], 3, padding=1) for i in range(5)]
        )
        self.bn_b = nn.ModuleList([nn.BatchNorm2d(s_[i]) for i in range(5)])
        self.fc1 = nn.Linear(s_[4] * self.SPATIAL, fc)
        self.fc2 = nn.Linear(fc, fc)
        self.masks = (
            nn.ModuleList([ChannelMask(d, mask_init) for d in self.dims])
            if prune
            else None
        )

    def _m(self, idx, x):
        return x if self.masks is None else self.masks[idx](x)

    def forward(self, x):
        for i in range(5):
            x = self._m(i, F.relu(self.bn_down[i](self.down[i](x))))
            h = self._m(5 + i, F.relu(self.bn_a[i](self.conv_a[i](x))))
            h = self._m(i, self.bn_b[i](self.conv_b[i](h)))
            x = F.relu(x + h)
        x = x.view(x.size(0), -1)
        x = self._m(10, F.relu(self.fc1(x)))
        x = self._m(11, F.relu(self.fc2(x)))
        return x

    # ---- pruning ----
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
            return list(self.dims)
        return [m.kept() for m in self.masks]

    def effective_params(self) -> int:
        k = self.kept()
        ks, km, f1, f2 = k[:5], k[5:10], k[10], k[11]
        cin = [self.in_channels] + ks[:4]
        p = 0
        for i in range(5):
            p += ks[i] * cin[i] * 9 + 3 * ks[i]
            p += km[i] * ks[i] * 9 + 3 * km[i]
            p += ks[i] * km[i] * 9 + 3 * ks[i]
        p += f1 * (self.SPATIAL * ks[4]) + f1
        p += f2 * f1 + f2
        return p

    def sparsity_loss(self):
        n = [m.current().sum() for m in self.masks]
        ns, nm, n1, n2 = n[:5], n[5:10], n[10], n[11]
        used = full = 0.0
        d_s, d_m = self.dims[:5], self.dims[5:10]
        for i in range(5):
            prev_n, prev_d = (
                (float(self.in_channels), self.in_channels)
                if i == 0
                else (ns[i - 1], d_s[i - 1])
            )
            used = used + 9 * prev_n * ns[i] + 9 * ns[i] * nm[i] + 9 * nm[i] * ns[i]
            full += 9 * prev_d * d_s[i] + 9 * d_s[i] * d_m[i] + 9 * d_m[i] * d_s[i]
        used = used + self.SPATIAL * ns[4] * n1 + n1 * n2
        full += self.SPATIAL * d_s[4] * self.dims[10] + self.dims[10] * self.dims[11]
        return used / float(full)


class MLPExpert(nn.Module):
    feat_dim = 400
    DIMS = (400, 400)

    def __init__(
        self, prune: bool = False, mask_init: float = 1.0, in_features: int = 784
    ):
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, 400)
        self.fc2 = nn.Linear(400, 400)
        self.masks = (
            nn.ModuleList([ChannelMask(d, mask_init) for d in self.DIMS])
            if prune
            else None
        )

    def _m(self, i, x):
        return x if self.masks is None else self.masks[i](x)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self._m(0, F.relu(self.fc1(x)))
        x = self._m(1, F.relu(self.fc2(x)))
        return x

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
        return list(self.DIMS) if self.masks is None else [m.kept() for m in self.masks]

    def effective_params(self) -> int:
        k0, k1 = self.kept()
        return k0 * self.in_features + k0 + k1 * k0 + k1

    def sparsity_loss(self):
        n0, n1 = [m.current().sum() for m in self.masks]
        used = self.in_features * n0 + n0 * n1
        full = float(self.in_features * 400 + 400 * 400)
        return used / full


class DERNet(models.DynamicModule):
    """Expandable representation in the style of DER (Yan et al., CVPR 2021) over the CNN of the article.

    - At each new step (training experience) a new ExpertCNN is added,
      initialized with the weights of the previous; the previous are frozen
      (requires_grad=False and BatchNorm in eval mode).
    - The single classifier receives the concatenation of the features of all the
      extratores (the previous are detached, the last is not).
    """

    def __init__(
        self,
        initial_out_features: int = 10,
        mask_value: float = -1000,
        init_from_previous: bool = True,
        prune: bool = False,
        backbone: str = "plain",
        inner_ratio: float = 0.5,
        width_mult: float = 1.0,
        in_channels: int = 3,
    ):
        super().__init__()
        assert backbone in ("plain", "res", "mlp")
        assert (
            backbone == "res" or in_channels == 3
        ), "the plain backbone only accepts 3 channels; use --backbone res"
        self.in_channels = in_channels
        self.mask_value = mask_value
        self.init_from_previous = init_from_previous
        self.prune = prune
        self.backbone = backbone
        self.inner_ratio = inner_ratio
        self.width_mult = width_mult

        self.experts = nn.ModuleList([self._make_expert()])
        self.feat_dim = self.experts[0].feat_dim
        self.fc = nn.Linear(self.feat_dim, initial_out_features)
        self.aux_fc = None
        self.register_buffer(
            "active_units", torch.zeros(initial_out_features, dtype=torch.int8)
        )

        self.register_buffer(
            "class_task", torch.full((initial_out_features,), -1, dtype=torch.long)
        )

        self.register_buffer(
            "step_rank", torch.full((initial_out_features,), -1, dtype=torch.long)
        )
        self.known_classes = 0
        self.cur_step = -1
        self._steps_seen = 0
        self._last_train_exp = -1

    def _make_expert(self):
        if self.backbone == "mlp":
            return MLPExpert(prune=self.prune)
        if self.backbone == "res":
            return ResExpert(
                prune=self.prune,
                inner_ratio=self.inner_ratio,
                width_mult=self.width_mult,
                in_channels=self.in_channels,
            )
        return ExpertCNN(prune=self.prune)

    # ----------------------------
    # Dynamic Adaptation
    # ----------------------------
    @torch.no_grad()
    def adaptation(self, experience):
        super().adaptation(experience)
        device = self._adaptation_device
        curr = sorted(experience.classes_in_this_experience)

        self._ensure_width(max(curr) + 1, device)

        if not self.training:
            return
        exp_id = experience.current_experience
        if exp_id <= self._last_train_exp:
            return
        self._last_train_exp = exp_id

        self.known_classes = int(self.active_units.sum().item())
        self.active_units[curr] = 1
        self.class_task[curr] = self._steps_seen
        self.step_rank[curr] = torch.arange(len(curr), device=self.step_rank.device)
        self.cur_step = self._steps_seen

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
        sr = torch.full((n_cls,), -1, dtype=torch.long, device=device)
        sr[:old_n] = self.step_rank
        self.step_rank = sr

    def _add_expert(self, n_new: int, device):
        prev = self.experts[-1]
        prev.freeze_masks()
        prev.eval()
        prev.requires_grad_(False)

        new = self._make_expert().to(device)
        if self.init_from_previous:
            new.load_state_dict(prev.state_dict())
            new.reset_masks()
        self.experts.append(new)

        old_fc = self.fc
        fc = nn.Linear(self.feat_dim * len(self.experts), old_fc.out_features)
        fc = fc.to(device)
        fc.weight[:, : old_fc.in_features] = old_fc.weight
        fc.bias.copy_(old_fc.bias)
        self.fc = fc

        self.aux_fc = nn.Linear(self.feat_dim, n_new + 1).to(device)

    # ----------------------------
    # Classes in the current step
    # ----------------------------
    def is_new(self, y):
        """True para amostras de classes da etapa corrente."""
        return self.class_task[y] == self.cur_step

    def aux_target(self, y):
        """Rótulo do classificador auxiliar: 0 = classe antiga; 1..N = classe nova."""
        return torch.where(self.is_new(y), self.step_rank[y] + 1, torch.zeros_like(y))

    def new_classes(self):
        return (self.class_task == self.cur_step).nonzero().squeeze(1)

    # ----------------------------
    # Prune
    # ----------------------------
    def set_mask_scale(self, s: float):
        self.experts[-1].set_scale(s)

    def finalize_masks(self):
        self.experts[-1].freeze_masks()

    def sparsity_loss(self):
        return self.experts[-1].sparsity_loss()

    # ----------------------------
    # Forward
    # ----------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        for e in list(self.experts)[:-1]:
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

    # --------------------------------------
    # Parameter Counting and Pruning Report
    # --------------------------------------
    def count_params(self, effective: bool = True) -> int:
        if not effective:
            return sum(p.numel() for p in self.parameters())
        n_cls = int(self.active_units.sum().item())
        feat = sum(e.kept()[-1] for e in self.experts)
        return sum(e.effective_params() for e in self.experts) + n_cls * feat + n_cls

    def prune_report(self) -> dict:
        return {
            "kept_channels": [e.kept() for e in self.experts],
            "feature_dims": [e.kept()[-1] for e in self.experts],
            "params_effective": self.count_params(True),
            "params_allocated": self.count_params(False),
        }
