import torch
import torch.nn as nn
import torch.nn.functional as F


class DENLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.active_neurons = torch.ones(out_features).bool()

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = F.relu(out)

        self.last_activation = out.detach()
        return out

    def expand_den_layer(self, n_new_neurons):
        device = self.weight.device

        new_weights = torch.randn(n_new_neurons, self.in_features, device=device) * 0.01
        new_bias = torch.zeros(n_new_neurons, device=device)

        new_weight_param = nn.Parameter(new_weights)
        new_bias_param = nn.Parameter(new_bias)

        with torch.no_grad():

            old_weight = self.weight.detach()
            old_weight.requires_grad = False

            old_bias = self.bias.detach()
            old_bias.requires_grad = False

            combined_weight = torch.cat([old_weight, new_weights], dim=0)
            combined_bias = torch.cat([old_bias, new_bias], dim=0)

            self.weight = nn.Parameter(combined_weight)
            self.bias = nn.Parameter(combined_bias)
            self.out_features += n_new_neurons
            self.active_neurons = torch.cat(
                [self.active_neurons, torch.ones(n_new_neurons, device=device).bool()]
            )

            self.new_parameters = [new_weight_param, new_bias_param]

        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias

    def get_new_parameters(self):
        return getattr(self, "new_parameters", [])

    def get_old_parameters(self):
        if not hasattr(self, "new_parameters"):
            return [self.weight, self.bias]
        new_w, new_b = self.new_parameters
        self._old_w = self.weight[: -new_w.shape[0]]
        self._old_b = self.bias[: -new_b.shape[0]]

        return self._old_w, self._old_b

    def get_saturation(self, threshold=1e-4):
        if not hasattr(self, "last_activation"):
            return None

        zeros = (self.last_activation < threshold).float()
        saturation_per_neuron = zeros.mean()

        return saturation_per_neuron

    def conditional_expand(self, n_new_neurons=20, threshold=0.95, neuron_fraction=0.5):
        saturation = self.get_saturation()
        if saturation is None:
            return False

        saturated_count = (saturation > threshold).sum().item()
        total = saturation.numel()
        frac = saturated_count / total

        if frac > neuron_fraction:
            print(
                f"[DENLayer] Expandindo camada com {n_new_neurons} neurônios (saturação: {frac:.2f})"
            )
            self.expand_den_layer(n_new_neurons)
            return True

        return False

    def remove_last_expansion(self):
        if not hasattr(self, "new_parameters"):
            return

        n_new = len(self.new_parameters[0])
        self.weight = nn.Parameter(self.weight[:-n_new])
        self.bias = nn.Parameter(self.bias[:-n_new])
        self.out_features -= n_new
        self.active_neurons = self.active_neurons[:-n_new]
        del self.new_parameters
