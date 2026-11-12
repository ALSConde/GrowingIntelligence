import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier

from models.layers.LinearAttention import LinearAttention


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
    FC_IDS = (5, 6)  # camadas que o modo "fc" do plugin de expansão pode alargar

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

    @staticmethod
    def _grow_bn(ob, n):
        """BatchNorm com n canais novos neutros (γ=1, β=0, média 0, var 1)."""
        nb = nn.BatchNorm2d(ob.num_features + n).to(ob.weight.device)
        k = ob.num_features
        with torch.no_grad():
            nb.weight[:k] = ob.weight
            nb.bias[:k] = ob.bias
            nb.running_mean[:k] = ob.running_mean
            nb.running_var[:k] = ob.running_var
            nb.num_batches_tracked.copy_(ob.num_batches_tracked)
        nb.train(ob.training)  # herda o modo (módulo novo nasce em treino)
        return nb

    @torch.no_grad()
    def _grow_conv_out(self, i: int, n: int):
        """Acrescenta n canais de saída à conv i e à sua BatchNorm (unidades novas neutras)."""
        dev = next(self.parameters()).device
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

        nb = self._grow_bn(self.bns[i], n)
        self.bns[i] = nb

    @torch.no_grad()
    def expand(self, i: int, n: int):
        if n <= 0:
            return
        dev = next(self.parameters()).device
        if i < 5:
            self._grow_conv_out(i, n)

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


class ExpandableCNNAttention(ExpandableCNN):
    """Model_CNN_Cifar_attention com larguras ajustáveis e expansão de canais.

    Atenção espacial após a conv5 e atenção entre fc1/fc2, com a dimensão de atenção
    `d_att` FIXA (e a memória persistente k_mem/v_mem, em R^{mem_size x d_att}, intacta).
    Para a expansão da conv5 funcionar, a atenção ganha uma projeção de saída
    `att_out: d_att -> C`. No modelo original ela não existe porque d_att == C == 256 e a
    saída da atenção é somada direto ao mapa de features; aqui ela começa como identidade
    (retangular se C != d_att), o que reproduz o modelo original quando width_mult = 1.

    Camadas: 0..4 = conv1..conv5, 5 = fc1, 6 = fc2, 7 = proj3 (linear_3, que alimenta o
    classificador). Expandir:
      conv5 (+K canais): proj_Q/K/V da atenção espacial ganham K colunas de entrada zeradas,
          att_out ganha K linhas zeradas, a BatchNorm pós-atenção ganha K canais neutros e a
          fc1 ganha K*16 colunas zeradas;
      fc1 (+K): fc2 e proj_K/proj_V da atenção de features ganham K colunas zeradas;
      fc2 (+K): proj_Q da atenção de features ganha K colunas zeradas;
      proj3 (+K): o classificador ganha K colunas zeradas.
    Como d_att não muda e as unidades novas entram com pesos zero, a função é preservada
    exatamente (a feature map ELU+1 só importa se d_att crescesse: ELU(0)+1 = 1 alteraria
    a normalização da atenção linear).
    """

    BASE = (16, 32, 64, 128, 256, 400, 400, 400)
    NAMES = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "proj3")
    FC_IDS = (5, 6, 7)

    def __init__(
        self,
        width_mult: float = 1.0,
        initial_classes: int = 10,
        d_att: int = 256,
        mem_size: int = 64,
    ):
        models.DynamicModule.__init__(self)
        w = [max(1, int(round(b * width_mult))) for b in self.BASE]
        self.widths = list(w)
        self.d_att = d_att
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
        self.spatial_attention = LinearAttention(w[4], w[4], d_att, mem_size=mem_size)
        self.att_out = nn.Linear(d_att, w[4], bias=False)
        with torch.no_grad():
            self.att_out.weight.copy_(torch.eye(w[4], d_att))
        self.bn_att = nn.BatchNorm2d(w[4])
        self.fc1 = nn.Linear(w[4] * self.SPATIAL, w[5])
        self.fc2 = nn.Linear(w[5], w[6])
        self.attention = LinearAttention(w[6], w[5], d_att, mem_size=mem_size)
        self.proj3 = nn.Linear(d_att, w[7])
        self.classifier = IncrementalClassifier(
            w[7], initial_out_features=initial_classes
        )

    def forward_acts(self, x):
        acts = []
        for i in range(5):
            x = F.relu(self.bns[i](self.convs[i](x)))
            acts.append(x)
        B, C, H, W = x.shape
        flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        res = self.att_out(self.spatial_attention(flat, flat, flat))  # (B, H*W, C)
        out = res.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.bn_att(out + x).reshape(B, -1)
        h1 = F.relu(self.fc1(x))
        acts.append(h1)
        h2 = F.relu(self.fc2(h1))
        acts.append(h2)
        h_att = self.attention(
            h2.unsqueeze(1), h1.unsqueeze(1), h1.unsqueeze(1)
        ).squeeze(1)
        h3 = self.proj3(h_att)
        acts.append(h3)
        return self.classifier(h3), acts

    # ------------------------------------------------------------------ custo
    def unit_cost(self, i: int) -> int:
        w, d = self.widths, self.d_att
        if i < 4:
            return super().unit_cost(i)
        if i == 4:
            own = w[3] * 9 + 1 + 2
            down = (
                3 * d + d + 2 + w[5] * self.SPATIAL
            )  # Q,K,V; linha de att_out; BN; fc1
        elif i == 5:
            own = w[4] * self.SPATIAL + 1
            down = w[6] + 2 * d  # fc2; proj_K e proj_V
        elif i == 6:
            own = w[5] + 1
            down = d  # proj_Q
        else:
            own = d + 1
            down = self.classifier.classifier.out_features
        return own + down

    # ------------------------------------------------------------------ expansão
    def _grow_att_in(self, att, names, n):
        for name in names:
            old = getattr(att, name)
            setattr(
                att,
                name,
                self._grow_in(
                    old, n, lambda: nn.Linear(old.in_features + n, old.out_features)
                ),
            )

    @staticmethod
    def _grow_out(layer, n, bias=True):
        new = nn.Linear(layer.in_features, layer.out_features + n, bias=bias).to(
            layer.weight.device
        )
        with torch.no_grad():
            new.weight[: layer.out_features] = layer.weight
            if bias:
                new.bias[: layer.out_features] = layer.bias
        return new

    @torch.no_grad()
    def expand(self, i: int, n: int):
        if n <= 0:
            return
        if i < 4:
            return super().expand(i, n)
        if i == 4:
            self._grow_conv_out(4, n)
            self._grow_att_in(self.spatial_attention, ("proj_Q", "proj_K", "proj_V"), n)
            o = self.att_out
            new = self._grow_out(o, n, bias=False)
            new.weight[o.out_features :] = 0
            self.att_out = new
            self.bn_att = self._grow_bn(self.bn_att, n)
            f = self.fc1
            self.fc1 = self._grow_in(
                f,
                n * self.SPATIAL,
                lambda: nn.Linear(f.in_features + n * self.SPATIAL, f.out_features),
            )
        elif i == 5:
            self.fc1 = self._grow_out(self.fc1, n)
            g = self.fc2
            self.fc2 = self._grow_in(
                g, n, lambda: nn.Linear(g.in_features + n, g.out_features)
            )
            self._grow_att_in(self.attention, ("proj_K", "proj_V"), n)
        elif i == 6:
            self.fc2 = self._grow_out(self.fc2, n)
            self._grow_att_in(self.attention, ("proj_Q",), n)
        else:
            self.proj3 = self._grow_out(self.proj3, n)
            c = self.classifier.classifier
            self.classifier.classifier = self._grow_in(
                c, n, lambda: nn.Linear(c.in_features + n, c.out_features)
            )
        self.widths[i] += n
