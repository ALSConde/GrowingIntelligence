import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import avalanche.models as models
from avalanche.models import IncrementalClassifier
from models.layers.WDLayer import WDLayer
from models.layers.LinearAttention import LinearAttention


class model_cnn_train(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
                    ("batchnorm1", nn.BatchNorm2d(16)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2)),
                    ("batchnorm2", nn.BatchNorm2d(32)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2)),
                    ("batchnorm3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU()),
                    ("conv4", nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2)),
                    ("batchnorm4", nn.BatchNorm2d(128)),
                    ("relu4", nn.ReLU()),
                    ("conv5", nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)),
                    ("batchnorm5", nn.BatchNorm2d(256)),
                    ("relu5", nn.ReLU()),
                ]
            )
        )

        self.fc1 = nn.Linear(256 * 4 * 4, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.classifier = nn.Linear(self.fc2.out_features, 10)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


class model_cifar_we(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = model_cnn_train()
        self.extractor.load_state_dict(torch.load("./model_cnn.pth"))
        self.cnn = self.extractor.cnn

        self.cnn.eval()  # Set the pre-trained CNN to evaluation mode
        self.den1 = WDLayer(256 * 4 * 4, 400)
        self.den2 = WDLayer(self.den1.out_features, 400)
        self.classifier = IncrementalClassifier(
            self.den2.out_features, initial_out_features=10
        )

    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation for the pre-trained CNN
            x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.den1(x))
        x = F.relu(self.den2(x))
        x = self.classifier(x)
        return x


class model_cifar_mlp(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = model_cnn_train()
        self.extractor.load_state_dict(torch.load("./model_cnn.pth"))
        self.cnn = self.extractor.cnn

        self.cnn.eval()  # Set the pre-trained CNN to evaluation mode
        self.linear_1 = nn.Linear(256 * 4 * 4, 400)
        self.linear_2 = nn.Linear(self.linear_1.out_features, 400)
        self.classifier = IncrementalClassifier(
            self.linear_2.out_features, initial_out_features=10
        )

    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation for the pre-trained CNN
            x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.classifier(x)
        return x


class model_cifar_mlp_attention(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = model_cnn_train()
        self.extractor.load_state_dict(torch.load("./model_cnn.pth"))
        self.cnn = self.extractor.cnn
        self.cnn.eval()  # Set the pre-trained CNN to evaluation mode

        self.spatial_attention = LinearAttention(
            d_q=self.cnn[-1].out_channels,
            d_kv=self.cnn[-1].out_channels,
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
            mem_size=64,
        )
        self.linear_3 = nn.Linear(256, 400)
        self.classifier = IncrementalClassifier(
            self.linear_3.out_features, initial_out_features=10
        )

    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)

    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation for the pre-trained CNN
            x = self.cnn(x)

        # ------------------- Spatial Attention -------------------
        B, C, H, W = x.shape
        flat = x.view(B, C, H * W).permute(0, 2, 1)
        res = self.spatial_attention(flat, flat, flat)
        out = res.permute(0, 2, 1).view(B, C, H, W)
        x = self.batchnorm_att(out + x)

        # ------------ MLP ---------------
        x = x.reshape(x.size(0), -1)
        h1 = F.relu(self.linear_1(x))
        h2 = F.relu(self.linear_2(h1))

        # -------------- Attention ---------------
        Q = h2.unsqueeze(1)
        K = h1.unsqueeze(1)
        V = K
        h_att = self.attention(Q, K, V).squeeze(1)

        # -------------- Classificador ---------------
        h_att = self.linear_3(h_att)
        out = self.classifier(h_att)
        return out


class model_cifar_we_attention(models.DynamicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = model_cnn_train()
        self.extractor.load_state_dict(torch.load("./model_cnn.pth"))
        self.cnn = self.extractor.cnn
        self.cnn.eval()  # Set the pre-trained CNN to evaluation mode

        self.spatial_attention = LinearAttention(
            d_q=self.cnn[-1].out_channels,
            d_kv=self.cnn[-1].out_channels,
            d_att=256,
            mem_size=64,
        )
        self.batchnorm_att = nn.BatchNorm2d(256)
        self.den_1 = WDLayer(256 * 4 * 4, 400)
        self.den_2 = WDLayer(self.den_1.out_features, 400)
        self.attention = LinearAttention(
            d_q=self.den_2.out_features,
            d_kv=self.den_1.out_features,
            d_att=256,
            mem_size=64,
        )
        self.den_3 = WDLayer(256, 400)
        self.classifier = IncrementalClassifier(
            self.den_3.out_features, initial_out_features=10
        )
        
    def adaptation(self, experience):
        super().adaptation(experience)
        self.classifier.adaptation(experience=experience)
        
    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation for the pre-trained CNN
            x = self.cnn(x)

        # ------------------- Spatial Attention -------------------
        B, C, H, W = x.shape
        flat = x.view(B, C, H * W).permute(0, 2, 1)
        res = self.spatial_attention(flat, flat, flat)
        out = res.permute(0, 2, 1).view(B, C, H, W)
        x = self.batchnorm_att(out + x)

        # ------------ DEN ---------------
        x = x.reshape(x.size(0), -1)
        h1 = F.relu(self.den_1(x))
        h2 = F.relu(self.den_2(h1))

        # -------------- Attention ---------------
        Q = h2.unsqueeze(1)
        K = h1.unsqueeze(1)
        V = K
        h_att = self.attention(Q, K, V).squeeze(1)

        # -------------- Classificador ---------------
        h_att = self.den_3(h_att)
        out = self.classifier(h_att)
        return out
