import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier


class ExpandableCNN(models.DynamicModule):
    """Mesma arquitetura de Model_CNN_Cifar (5 blocos conv+BN+ReLU, FC400, FC400,
    IncrementalClassifier), mas com largura de TODAS as camadas ajustável:

      - width_mult escala as larguras iniciais (baseline de capacidade igualada);
      - expand(i, n) acrescenta n canais (conv) ou neurônios (FC) à camada i,
        preservando a função: pesos de saída das unidades novas começam em zero, a
        BatchNorm nova começa neutra e a camada seguinte recebe colunas zeradas.

    Camadas: 0..4 = conv1..conv5, 5 = fc1, 6 = fc2.
    """

    BASE = (16, 32, 64, 128, 256, 400, 400)
    NAMES = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2")
    STRIDES = (1, 2, 2, 2, 2)
    PADDINGS = (1, 2, 2, 2, 2)
    SPATIAL = 16  # posições espaciais (4x4) da última conv, achatadas antes da fc1

    def __init__(self, width_mult: float = 1.0, initial_classes: int = 10):
        super().__init__()
        w = [max(1, int(round(b * width_mult))) for b in self.BASE]
        self.widths = list(w)
        cin = [3] + w[:4]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    cin[i], w[i], 3, stride=self.STRIDES[i], padding=self.PADDINGS[i]
                )
                for i in range(5)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(w[i]) for i in range(5)])
        self.fc1 = nn.Linear(w[4] * self.SPATIAL, w[5])
        self.fc2 = nn.Linear(w[5], w[6])
        self.classifier = IncrementalClassifier(
            w[6], initial_out_features=initial_classes
        )

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    # ------------------------------------------------------------------ forward
    def forward_acts(self, x):
        """Logits e as ativações (após ReLU) das 7 camadas."""
        acts = []
        for i in range(5):
            x = F.relu(self.bns[i](self.convs[i](x)))
            acts.append(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        acts.append(x)
        x = F.relu(self.fc2(x))
        acts.append(x)
        return self.classifier(x), acts

    def forward(self, x):
        return self.forward_acts(x)[0]

    # ------------------------------------------------------------------ custo
    def extractor_params(self) -> int:
        return sum(
            p.numel()
            for n, p in self.named_parameters()
            if not n.startswith("classifier")
        )

    def unit_cost(self, i: int) -> int:
        """Parâmetros acrescentados por UMA unidade nova na camada i, incluindo os pesos
        que ela recebe, os da BatchNorm e as colunas que entram na camada seguinte."""
        w = self.widths
        n_cls = self.classifier.classifier.out_features
        if i < 5:
            cin = 3 if i == 0 else w[i - 1]
            own = cin * 9 + 1 + 2
            down = w[i + 1] * 9 if i < 4 else w[5] * self.SPATIAL
        elif i == 5:
            own = w[4] * self.SPATIAL + 1
            down = w[6]
        else:
            own = w[5] + 1
            down = n_cls
        return own + down

    # ------------------------------------------------------------------ expansão
    @staticmethod
    def _grow_in(layer, n_in, make):
        """Nova camada com n_in entradas a mais; pesos antigos copiados, novos = 0."""
        new = make().to(layer.weight.device)
        old_in = layer.weight.shape[1]
        new.weight.zero_()
        new.weight[:, :old_in] = layer.weight
        new.bias.copy_(layer.bias)
        return new

    @torch.no_grad()
    def expand(self, i: int, n: int):
        if n <= 0:
            return
        dev = next(self.parameters()).device
        if i < 5:
            old = self.convs[i]
            new = nn.Conv2d(
                old.in_channels,
                old.out_channels + n,
                3,
                stride=self.STRIDES[i],
                padding=self.PADDINGS[i],
            ).to(dev)
            new.weight[: old.out_channels] = old.weight
            new.bias[: old.out_channels] = old.bias
            self.convs[i] = new

            ob = self.bns[i]
            nb = nn.BatchNorm2d(ob.num_features + n).to(dev)
            k = ob.num_features
            nb.weight[:k] = ob.weight
            nb.bias[:k] = ob.bias
            nb.running_mean[:k] = ob.running_mean
            nb.running_var[:k] = ob.running_var
            nb.num_batches_tracked.copy_(ob.num_batches_tracked)
            nb.train(ob.training)  # herda o modo (módulo novo nasce em treino)
            self.bns[i] = nb

            if i < 4:
                nxt = self.convs[i + 1]
                self.convs[i + 1] = self._grow_in(
                    nxt,
                    n,
                    lambda: nn.Conv2d(
                        nxt.in_channels + n,
                        nxt.out_channels,
                        3,
                        stride=self.STRIDES[i + 1],
                        padding=self.PADDINGS[i + 1],
                    ),
                )
            else:  # a fc1 recebe os canais novos ao fim do vetor achatado (ordem canal-major)
                f = self.fc1
                self.fc1 = self._grow_in(
                    f,
                    n * self.SPATIAL,
                    lambda: nn.Linear(f.in_features + n * self.SPATIAL, f.out_features),
                )
        elif i == 5:
            f = self.fc1
            new = nn.Linear(f.in_features, f.out_features + n).to(dev)
            new.weight[: f.out_features] = f.weight
            new.bias[: f.out_features] = f.bias
            self.fc1 = new
            g = self.fc2
            self.fc2 = self._grow_in(
                g, n, lambda: nn.Linear(g.in_features + n, g.out_features)
            )
        else:
            f = self.fc2
            new = nn.Linear(f.in_features, f.out_features + n).to(dev)
            new.weight[: f.out_features] = f.weight
            new.bias[: f.out_features] = f.bias
            self.fc2 = new
            c = self.classifier.classifier
            self.classifier.classifier = self._grow_in(
                c, n, lambda: nn.Linear(c.in_features + n, c.out_features)
            )
        self.widths[i] += n
