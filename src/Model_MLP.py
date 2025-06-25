import torch.nn as nn
import torch.nn.functional as F
import avalanche.models as models
from avalanche.models import IncrementalClassifier, MultiHeadClassifier


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


class Model_MLP_TIL(models.MultiTaskModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_1 = nn.Linear(28 * 28, 400)
        self.linear_2 = nn.Linear(self.linear_1.out_features, 400)
        self.classifier = MultiHeadClassifier(self.linear_2.out_features, 2)

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x, task_label):
        x = x.reshape((x.size(0), -1))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.classifier(x, task_label)
        return x
