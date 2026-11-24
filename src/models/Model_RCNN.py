import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier

from models.layers.LinearAttention import LinearAttention
from models.Model_ECNN import ExpandableCNN, ExpandableCNNAttention


class ResExpandableCNN(models.DynamicModule):
    STREAM = (16, 32, 64, 128, 256)
    STRIDES = (1, 2, 2, 2, 2)
    PADDINGS = (1, 2, 2, 2, 2)
    SPATIAL = 16

    _grow_in = staticmethod(ExpandableCNN._grow_in)
    _grow_bn = staticmethod(ExpandableCNN._grow_bn)
    _grow_out = staticmethod(ExpandableCNNAttention._grow_out)
    _grow_att_in = ExpandableCNNAttention._grow_att_in

    def __init__(
        self,
        width_mult: float = 1.0,
        initial_classes: int = 10,
        inner_ratio: float = 0.5,
        attention: bool = False,
        d_att: int = 256,
        mem_size: int = 64,
        in_channels: int = 3,
        pool: str = "flatten",
        emb_dim: int = 0,
        mem_mode: str = "append",
        mem_gain: float = 1.0,
    ):
        super().__init__()
        assert pool in ("flatten", "gap")
        self.pool = pool
        # posições espaciais que entram no fc1 (gap = 1 -> o fc1 não depende da resolução)
        self.spatial = 1 if pool == "gap" else self.SPATIAL
        self.emb_dim = int(emb_dim)
        self.in_channels = in_channels
        s = [max(1, int(round(b * width_mult))) for b in self.STREAM]
        fc = max(1, int(round(400 * width_mult)))
        inner = [max(1, int(round(c * inner_ratio))) for c in s]
        self.stream = s
        self.attention_on = attention
        self.d_att = d_att
        n_fc = 3 if attention else 2
        self.widths = inner + [fc] * n_fc
        self.NAMES = ("blk1", "blk2", "blk3", "blk4", "blk5", "fc1", "fc2") + (
            ("proj3",) if attention else ()
        )
        self.FC_IDS = (5, 6, 7) if attention else (5, 6)

        cin = [in_channels] + s[:4]
        self.down = nn.ModuleList(
            [
                nn.Conv2d(
                    cin[i], s[i], 3, stride=self.STRIDES[i], padding=self.PADDINGS[i]
                )
                for i in range(5)
            ]
        )
        self.bn_down = nn.ModuleList([nn.BatchNorm2d(s[i]) for i in range(5)])
        self.conv_a = nn.ModuleList(
            [nn.Conv2d(s[i], inner[i], 3, padding=1) for i in range(5)]
        )
        self.bn_a = nn.ModuleList([nn.BatchNorm2d(inner[i]) for i in range(5)])
        self.conv_b = nn.ModuleList(
            [nn.Conv2d(inner[i], s[i], 3, padding=1) for i in range(5)]
        )
        self.bn_b = nn.ModuleList([nn.BatchNorm2d(s[i]) for i in range(5)])
        if attention:
            self.spatial_attention = LinearAttention(
                s[4],
                s[4],
                d_att,
                mem_size=mem_size,
                mem_mode=mem_mode,
                mem_gain=mem_gain,
            )
            self.att_out = nn.Linear(d_att, s[4], bias=False)
            with torch.no_grad():
                self.att_out.weight.copy_(torch.eye(s[4], d_att))
            self.bn_att = nn.BatchNorm2d(s[4])
        self.fc1 = nn.Linear(s[4] * self.spatial, fc)
        self.fc2 = nn.Linear(fc, fc)
        if attention:
            self.attention = LinearAttention(
                fc, fc, d_att, mem_size=mem_size, mem_mode=mem_mode, mem_gain=mem_gain
            )
            self.proj3 = nn.Linear(d_att, fc)

        if self.emb_dim > 0:
            self.embed = nn.Linear(fc, self.emb_dim)
        self.classifier = IncrementalClassifier(
            self.emb_dim if self.emb_dim > 0 else fc,
            initial_out_features=initial_classes,
        )

        self.proto_mode = False
        self.proto_scale = 16.0
        self.report_head = "linear"
        self.last_emb = None
        if self.emb_dim > 0:
            self.register_buffer("proto", torch.zeros(initial_classes, self.emb_dim))
            self.register_buffer(
                "proto_valid", torch.zeros(initial_classes, dtype=torch.bool)
            )

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)
        self._sync_proto()

    @torch.no_grad()
    def _sync_proto(self):
        if self.emb_dim <= 0:
            return
        n = self.classifier.classifier.out_features
        have = self.proto.size(0)
        if have < n:
            dev = self.proto.device
            self.proto = torch.cat(
                [self.proto, torch.zeros(n - have, self.emb_dim, device=dev)]
            )
            self.proto_valid = torch.cat(
                [self.proto_valid, torch.zeros(n - have, dtype=torch.bool, device=dev)]
            )

    def forward_acts(self, x):
        acts = []
        for i in range(5):
            x = F.relu(self.bn_down[i](self.down[i](x)))
            h = F.relu(self.bn_a[i](self.conv_a[i](x)))
            acts.append(h)
            h = self.bn_b[i](self.conv_b[i](h))
            x = F.relu(x + h)
        B, C, H, W = x.shape
        if self.attention_on:
            flat = x.reshape(B, C, H * W).permute(0, 2, 1)
            res = self.att_out(self.spatial_attention(flat, flat, flat))
            x = self.bn_att(res.permute(0, 2, 1).reshape(B, C, H, W) + x)
        x = x.mean(dim=(2, 3)) if self.pool == "gap" else x.reshape(B, -1)
        h1 = F.relu(self.fc1(x))
        acts.append(h1)
        h2 = F.relu(self.fc2(h1))
        acts.append(h2)
        if self.attention_on:
            h_att = self.attention(
                h2.unsqueeze(1), h1.unsqueeze(1), h1.unsqueeze(1)
            ).squeeze(1)
            h3 = self.proj3(h_att)
            acts.append(h3)
            return self._head(h3), acts
        return self._head(h2), acts

    def _head(self, f):
        if self.emb_dim <= 0:
            return self.classifier(f)
        e = self.embed(f)
        self.last_emb = e
        if self.proto_mode and not self.training:
            en = F.normalize(e, dim=1)
            pn = F.normalize(self.proto, dim=1)
            logits = self.proto_scale * en @ pn.t()
            return logits.masked_fill(~self.proto_valid.unsqueeze(0), -1e4)
        return self.classifier(e)

    def forward(self, x):
        return self.forward_acts(x)[0]

    def extractor_params(self) -> int:
        return sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith("classifier")
        )

    def unit_cost(self, i: int) -> int:
        w, d = self.widths, self.d_att
        n_cls = self.classifier.classifier.out_features
        if i < 5:
            c = self.stream[i]
            return 9 * c + 1 + 2 + 9 * c
        if i == 5:
            own = self.stream[4] * self.spatial + 1
            down = w[6] + (2 * d if self.attention_on else 0)
        elif i == 6:
            own = w[5] + 1
            head_out = self.emb_dim if self.emb_dim > 0 else n_cls
            down = d if self.attention_on else head_out
        else:
            own = d + 1
            down = self.emb_dim if self.emb_dim > 0 else n_cls
        return own + down

    @torch.no_grad()
    def expand(self, i: int, n: int):
        if n <= 0:
            return
        dev = next(self.parameters()).device
        if i < 5:
            a = self.conv_a[i]
            new = nn.Conv2d(a.in_channels, a.out_channels + n, 3, padding=1).to(dev)
            new.weight[: a.out_channels] = a.weight
            new.bias[: a.out_channels] = a.bias
            self.conv_a[i] = new
            self.bn_a[i] = self._grow_bn(self.bn_a[i], n)
            b = self.conv_b[i]
            self.conv_b[i] = self._grow_in(
                b, n, lambda: nn.Conv2d(b.in_channels + n, b.out_channels, 3, padding=1)
            )
        elif i == 5:
            self.fc1 = self._grow_out(self.fc1, n)
            g = self.fc2
            self.fc2 = self._grow_in(
                g, n, lambda: nn.Linear(g.in_features + n, g.out_features)
            )
            if self.attention_on:
                self._grow_att_in(self.attention, ("proj_K", "proj_V"), n)
        elif i == 6:
            self.fc2 = self._grow_out(self.fc2, n)
            if self.attention_on:
                self._grow_att_in(self.attention, ("proj_Q",), n)
            else:
                self._grow_head_in(n)
        else:
            self.proj3 = self._grow_out(self.proj3, n)
            self._grow_head_in(n)
        self.widths[i] += n

    def _grow_head_in(self, n):
        if self.emb_dim > 0:
            e = self.embed
            self.embed = self._grow_in(
                e, n, lambda: nn.Linear(e.in_features + n, e.out_features)
            )
            return
        c = self.classifier.classifier
        self.classifier.classifier = self._grow_in(
            c, n, lambda: nn.Linear(c.in_features + n, c.out_features)
        )
