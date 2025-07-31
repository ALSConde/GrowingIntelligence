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

        self.current_time = 0
        self.added_times = torch.zeros(out_features)
        self.lambda_decay = 0.01

        self._activation_sum = None
        self._last_activation_sum = None
        self._activation_usage = None
        self._last_activation_usage = None
        self._counter = 0

    def forward(self, x):
        x = F.linear(x, self.weights, self.bias)
        x = F.layer_norm(x, x.shape[1:])
        x = F.relu(x)

        self.compute_sum_activation(x)
        self.compute_activation_usage(x)
        self.current_time += 1

        return x

    def expand(self, n_neurons: int):
        device = self.weights.device

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

            self.old_parameters = [old_weight, old_bias]
            self.new_parameters = [new_weight_param, new_bias_param]

            self.added_times = torch.cat(
                [
                    self.added_times,
                    torch.full((n_neurons,), self.current_time, device=device),
                ]
            )

        self._parameters["weight"] = self.weights
        self._parameters["bias"] = self.bias

    def prune(self, indices):
        if len(indices) == 0:
            return

        device = self.weights.device
        indices = torch.tensor(indices, dtype=torch.long, device=device)

        mask = torch.ones(self.out_features, dtype=torch.bool, device=device)
        mask[indices] = False

        with torch.no_grad():
            self.weights = nn.Parameter(self.weights[mask])
            self.bias = nn.Parameter(self.bias[mask])
            self.added_times = self.added_times[mask]

            self.out_features = self.weights.shape[0]

        self._parameters["weight"] = self.weights
        self._parameters["bias"] = self.bias

        if self._activation_usage is not None:
            self._activation_usage = self._activation_usage[mask]
        if self._activation_sum is not None:
            self._activation_sum = self._activation_sum[mask]
        if self._last_activation_usage is not None:
            self._last_activation_usage = self._last_activation_usage[mask]
        if self._last_activation_sum is not None:
            self._last_activation_sum = self._last_activation_sum[mask]

    def compute_temporal_weights(self):
        device = self.added_times.device
        t = torch.full_like(
            self.added_times, self.current_time, dtype=torch.float32, device=device
        )
        decay = torch.exp(-self.lambda_decay * (t - self.added_times))
        return decay

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

        if self._activation_sum is None:
            self._activation_sum = torch.zeros_like(batch_sum)

        self._activation_sum += batch_sum
        self._counter += x.shape[0]

    def compute_activation_usage(self, x, threshold=1e-3):
        activated = (x.detach() > threshold).float().sum(dim=0)

        if self._activation_usage is None:
            self._activation_usage = torch.zeros_like(activated)

        self._activation_usage += activated

    def reset_state(self):
        self._last_activation_sum = self._activation_sum
        self._activation_sum = None
        self._last_activation_usage = self._activation_usage
        self._activation_usage = None
        self._last_counter = self._counter
        self._counter = 0

    def get_activation_mean(self):
        if self._activation_sum is not None and self._counter > 0:
            return self._activation_sum / self._counter
        elif self._last_activation_sum is not None and self._last_counter > 0:
            return self._last_activation_sum / self._last_counter
        else:
            return torch.zeros_like(self._activation_sum)

    def get_activation_usage_ratio(self):
        if self._activation_usage is not None and self._counter > 0:
            return self._activation_usage / self._counter
        elif self._last_activation_usage is not None and self._last_counter > 0:
            return self._last_activation_usage / self._last_counter
        else:
            return torch.zeros_like(self._activation_usage)
