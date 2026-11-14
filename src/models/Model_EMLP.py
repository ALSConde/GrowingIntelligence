import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier

from models.layers.LinearAttention import LinearAttention
from models.Model_ECNN import ExpandableCNN, ExpandableCNNAttention


class ExpandableMLP(models.DynamicModule):
    _grow_in = staticmethod(ExpandableCNN._grow_in)
    _grow_out = staticmethod(ExpandableCNNAttention._grow_out)
    _grow_att_in = ExpandableCNNAttention._grow_att_in

    def __init__(
        self,
        width_mult: float = 1.0,
        initial_classes: int = 2,
        attention: bool = False,
        d_att: int = 128,
        mem_size: int = 32,
        in_features: int = 784,
    ):
        super().__init__()
        w = max(1, int(round(400 * width_mult)))
        self.in_features = in_features
        self.attention_on = attention
        self.d_att = d_att
        self.widths = [w, w] + ([w] if attention else [])
        self.NAMES = ("fc1", "fc2") + (("proj3",) if attention else ())
        self.FC_IDS = tuple(range(len(self.widths)))
        self.fc1 = nn.Linear(in_features, w)
        self.fc2 = nn.Linear(w, w)
        if attention:
            self.attention = LinearAttention(w, w, d_att, mem_size=mem_size)
            self.proj3 = nn.Linear(d_att, w)
        self.classifier = IncrementalClassifier(w, initial_out_features=initial_classes)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward_acts(self, x):
        x = x.reshape(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        acts = [h1, h2]
        if self.attention_on:
            h_att = self.attention(
                h2.unsqueeze(1), h1.unsqueeze(1), h1.unsqueeze(1)
            ).squeeze(1)
            h3 = self.proj3(h_att)
            acts.append(h3)
            return self.classifier(h3), acts
        return self.classifier(h2), acts

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
        if i == 0:
            return self.in_features + 1 + w[1] + (2 * d if self.attention_on else 0)
        if i == 1:
            return w[0] + 1 + (d if self.attention_on else n_cls)
        return d + 1 + n_cls

    def _grow_classifier_in(self, n):
        c = self.classifier.classifier
        self.classifier.classifier = self._grow_in(
            c, n, lambda: nn.Linear(c.in_features + n, c.out_features)
        )

    @torch.no_grad()
    def expand(self, i: int, n: int):
        if n <= 0:
            return
        if i == 0:
            self.fc1 = self._grow_out(self.fc1, n)
            g = self.fc2
            self.fc2 = self._grow_in(
                g, n, lambda: nn.Linear(g.in_features + n, g.out_features)
            )
            if self.attention_on:
                self._grow_att_in(self.attention, ("proj_K", "proj_V"), n)
        elif i == 1:
            self.fc2 = self._grow_out(self.fc2, n)
            if self.attention_on:
                self._grow_att_in(self.attention, ("proj_Q",), n)
            else:
                self._grow_classifier_in(n)
        else:
            self.proj3 = self._grow_out(self.proj3, n)
            self._grow_classifier_in(n)
        self.widths[i] += n
