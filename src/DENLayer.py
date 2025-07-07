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
    
    def expand(self, n_neurons: int):
        device = self.weights.device

        n_neurons = n_neurons * self.weights.shape[0]

        mean_weight = self.weights.mean(dim=0, keepdim=True)
        new_weights = mean_weight.repeat(n_neurons, 1)
        new_weights += torch.randn_like(new_weights) * 0.01

        new_bias = torch.zeros(n_neurons, device=device)

        new_weight_param = nn.Parameter(new_weights)
        new_bias_param = nn.Parameter(new_bias)

        with torch.no_grad():

            old_weight = self.weights.detach()
            old_weight.requires_grad = False

            old_bias = self.bias.detach()
            old_bias.requires_grad = False

            combined_weight = torch.cat([old_weight, new_weights], dim=0)
            combined_bias = torch.cat([old_bias, new_bias], dim=0)

            self.weights = nn.Parameter(combined_weight)
            self.bias = nn.Parameter(combined_bias)
            self.out_features += n_neurons
            self.active_neurons = torch.cat(
                [self.active_neurons, torch.ones(n_neurons, device=device).bool()]
            )

            self.new_parameters = [new_weight_param, new_bias_param]

        self._parameters["weight"] = self.weights
        self._parameters["bias"] = self.bias

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
