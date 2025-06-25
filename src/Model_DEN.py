import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier, MultiHeadClassifier
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

    def forward(self, x, task_label):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.den_1(x))
        x = F.relu(self.linear_1(x))
        x = self.classifier(x, task_label)
        return x


class Model_DEN(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.den_1 = DENLayer(28 * 28, 80)
        self.linear_1 = nn.Linear(self.den_1.out_features, 400)
        self.classifier = IncrementalClassifier(self.linear_1.out_features, 2)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.den_1(x))
        x = F.relu(self.linear_1(x))
        x = self.classifier(x)
        return x
