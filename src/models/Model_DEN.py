import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier
from .layers.DENLayer import DENLayer
from .layers.LinearAttention import LinearAttention


class Model_DEN_CIL(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.den_1 = DENLayer(28 * 28, 400)
        self.den_2 = DENLayer(self.den_1.out_features, 400)
        self.classifier = IncrementalClassifier(self.den_2.out_features)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = x.reshape((x.size(0), -1))
        x = self.den_1(x)
        x = self.den_2(x)
        x = self.classifier(x)
        return x


class Model_DEN_CIL_attention(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.den_1: nn.Module = DENLayer(784, 400)
        self.den_2: nn.Module = DENLayer(400, 400)
        self.attention: nn.Module = LinearAttention(
            d_q=self.den_2.out_features,
            d_kv=self.den_1.out_features,
            d_att=128,
            mem_size=32,
        )
        self.den_3 = DENLayer(128, 400)
        self.classifier = IncrementalClassifier(self.den_3.out_features)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        h1 = F.relu(self.den_1(x))  # (N, D1)
        h2 = F.relu(self.den_2(h1))  # (N, D2)

        # ---- Atenção ----
        # Q: batch atual
        Q = h2.unsqueeze(1)  # (N, 1, D)
        # K: memória curta
        K = h1.unsqueeze(1)  # (N, 1, D)
        # V: memória curta
        V = K

        # Atenção linear com memoria interna (k_mem, v_mem) como pesos treinaveis adicionados a K e V
        h_att = self.attention(Q, K, V).squeeze(1)  # (N, D)
        h3 = self.den_3(h_att)

        out = self.classifier(h3)
        return out


class Model_DEN_CIL_CIFAR(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # 32x32x3 -> 32x32x16
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=2
        )  # 32x32x16 -> 17x17x32
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=2
        )  # 17x17x32 -> 10x10x64
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=2
        )  # 10x10x64 -> 6x6x128
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=2
        )  # 6x6x128 -> 4x4x256
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.den1 = DENLayer(256 * 4 * 4, 400)
        self.den2 = DENLayer(self.den1.out_features, 400)
        self.classifier = IncrementalClassifier(
            self.den2.out_features, initial_out_features=10
        )

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.den1(x))
        x = F.relu(self.den2(x))
        x = self.classifier(x)
        return x


class Model_DEN_CIL_Cifar_attention(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # 32x32x3 -> 32x32x16
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=2
        )  # 32x32x16 -> 17x17x32
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=2
        )  # 17x17x32 -> 10x10x64
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=2
        )  # 10x10x64 -> 6x6x128
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=2
        )  # 6x6x128 -> 4x4x256
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.spatial_attention = LinearAttention(
            d_q=self.conv5.out_channels,
            d_kv=self.conv5.out_channels,
            d_att=256,
            mem_size=64,
        )
        self.batchnorm_att = nn.BatchNorm2d(256)
        self.den_1 = DENLayer(256 * 4 * 4, 400)
        self.den_2 = DENLayer(self.den_1.out_features, 400)
        self.attention = LinearAttention(
            d_q=self.den_2.out_features,
            d_kv=self.den_1.out_features,
            d_att=256,
            mem_size=64,
        )
        self.den_3 = DENLayer(256, 400)
        self.classifier = IncrementalClassifier(
            self.den_3.out_features, initial_out_features=10
        )

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        # ---- CNN ----
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.relu(self.batchnorm5(self.conv5(x)))  # (B, 256, 4, 4)

        # --------- Spatial Attention ---------
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        flat = x.view(B, C, H * W).permute(0, 2, 1)
        res = self.spatial_attention(flat, flat, flat)  # (B, H*W, C)
        # (B, C, H*W) -> (B, C, H, W)
        out = res.permute(0, 2, 1).view(B, C, H, W)
        x = self.batchnorm_att(out + x)

        # ---- DEN ----
        x = x.reshape(x.size(0), -1)  # (B, C*H*W)
        h1 = F.relu(self.den_1(x))
        h2 = F.relu(self.den_2(h1))

        # ---- Atenção como memória curta ----
        Q = h2.unsqueeze(1)
        K = h1.unsqueeze(1)
        V = K
        h_att = self.attention(Q, K, V).squeeze(1)

        # ---- Classificador ----
        h_att = self.den_3(h_att)
        out = self.classifier(h_att)
        return out
