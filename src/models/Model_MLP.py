import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier, MultiHeadClassifier
from .layers.LinearAttention import LinearAttention


class Model_MLP(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_1 = nn.Linear(28 * 28, 400)
        self.linear_2 = nn.Linear(self.linear_1.out_features, 400)
        self.classifier = IncrementalClassifier(self.linear_2.out_features, 2)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.classifier(x)
        return x


class model_MLP_attention(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_1 = nn.Linear(784, 400)
        self.linear_2 = nn.Linear(400, 400)
        self.attention = LinearAttention(
            d_q=self.linear_2.out_features,
            d_kv=self.linear_1.out_features,
            d_att=128,
            mem_size=32,
        )
        self.linear_3 = nn.Linear(128, 400)
        self.classifier = IncrementalClassifier(self.linear_2.out_features)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        h1 = F.relu(self.linear_1(x))
        h2 = F.relu(self.linear_2(h1))

        Q = h2.unsqueeze(1)  # (N, 1, D)

        K = h1.unsqueeze(1)  # (N, 1, D)
        V = h1.unsqueeze(1)  # (N, 1, D)

        h_att = self.attention(Q, K, V).squeeze(1)  # (N, D)
        h3 = self.linear_3(h_att)

        x = self.classifier(h3)
        return x


class Model_MLP_TIL(models.MultiTaskModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_1 = nn.Linear(28 * 28, 400)
        self.linear_2 = nn.Linear(self.linear_1.out_features, 400)
        self.classifier = MultiHeadClassifier(self.linear_2.out_features, 2)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x, task_labels):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.classifier(x, task_labels)
        return x


class Model_MLP_Cifar(models.DynamicModule):
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
        self.fc1 = nn.Linear(256 * 4 * 4, 400)
        self.fc2 = nn.Linear(400, 400)
        self.classifier = IncrementalClassifier(
            self.fc2.out_features, initial_out_features=10
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


class Model_MLP_CIL_Cifar_attention(models.DynamicModule):
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
        self.linear_1 = nn.Linear(256 * 4 * 4, 400)
        self.linear_2 = nn.Linear(self.linear_1.out_features, 400)
        self.attention = LinearAttention(
            d_q=self.linear_2.out_features,
            d_kv=self.linear_1.out_features,
            d_att=256,
            mem_size=256,
        )
        self.linear_3 = nn.Linear(256, 400)
        self.classifier = IncrementalClassifier(
            self.linear_3.out_features, initial_out_features=10
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

        # ---- MLP ----
        x = x.reshape(x.size(0), -1)  # (B, C*H*W)
        h1 = F.relu(self.linear_1(x))
        h2 = F.relu(self.linear_2(h1))

        # ---- Atenção como memória curta ----
        Q = h2.unsqueeze(1)
        K = h1.unsqueeze(1)
        V = K
        h_att = self.attention(Q, K, V).squeeze(1)

        # ---- Classificador ----
        h_att = self.linear_3(h_att)
        out = self.classifier(h_att)
        return out
