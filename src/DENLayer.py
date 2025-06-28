import torch
import torch.nn as nn
import torch.nn.functional as F


class DENLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.__activation_sum = None
        self.__counter = 0

    def forward(self, x):
        x = F.linear(x, self.weights, self.bias)
        x = F.relu(x)

        self.compute_sum_activation(x)

        return x

    def compute_average_gradient_norm(self):
        total_norm = 0.0
        total_elements = 0

        for p in self.parameters():
            if p.grad is not None:
                grad_flat = p.grad.data.view(-1)

                total_norm += grad_flat.abs().sum().item()

                total_elements += grad_flat.numel()

        if total_elements == 0:
            return 0.0

        return total_norm / total_elements

    def compute_sum_activation(self, x):
        batch_sum = x.detach().sum(dim=0)

        if self.__activation_sum is None:
            self.__activation_sum = torch.zeros_like(batch_sum)

        self.__activation_sum += batch_sum
        self.__counter += x.shape[0]

    def reset_state(self):
        self.__activation_sum = None
        self.__counter = 0

    def get_activation_mean(self):
        if self.__counter == 0:
            raise ValueError("Counter is zero, cannot compute mean activation.")
        return self.__activation_sum / self.__counter
