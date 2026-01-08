import torch
import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import (
    MultiHeadClassifier,
    IncrementalClassifier,
)
from DENLayer import DENLayer


class Model_DEN_TIL(models.MultiTaskModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.den_1 = DENLayer((28 * 28), 80)
        self.linear_1 = nn.Linear(self.den_1.out_features, 400)
        self.classifier = MultiHeadClassifier(self.linear_1.out_features, 2)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x, task_labels):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.den_1(x))
        x = F.relu(self.linear_1(x))
        x = self.classifier(x, task_labels)
        return x


class Model_DEN_CIL(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.den_1 = DENLayer(28 * 28, 80)
        self.den_2 = DENLayer(self.den_1.out_features, 80)
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
        self.den1 = DENLayer(256 * 4 * 4, 100)
        self.den2 = DENLayer(self.den1.out_features, 100)
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
