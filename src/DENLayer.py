import torch
import torch.nn as nn
import torch.nn.functional as F


class DENLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01, requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

        self.layer_norm = nn.LayerNorm(out_features)

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
        x = self.layer_norm(x)
        x = F.relu(x)

        self.compute_sum_activation(x)
        self.compute_activation_usage(x)
        self.current_time += 1

        return x

    def expand(self, n_neurons: int):
        device = self.weights.device

        mean_weight = self.weights.mean(dim=0, keepdim=True)
        new_weights = (
            mean_weight.repeat(n_neurons, 1)
            + torch.randn(n_neurons, self.in_features, device=device) * 0.01
        )
        new_bias = torch.zeros(n_neurons, device=device)

        with torch.no_grad():
            combined_weight = torch.cat([self.weights, new_weights], dim=0)
            combined_bias = torch.cat([self.bias, new_bias], dim=0)

        old_weights_grad = (
            self.weights.grad.clone() if self.weights.grad is not None else None
        )
        old_bias_grad = self.bias.grad.clone() if self.bias.grad is not None else None

        weights = nn.Parameter(combined_weight)
        bias = nn.Parameter(combined_bias)
        self.out_features += n_neurons

        if "weights" in self._parameters:
            self._parameters.pop("weights")
        if "bias" in self._parameters:
            self._parameters.pop("bias")

        self.register_parameter("weights", weights)
        self.register_parameter("bias", bias)

        if old_weights_grad is not None:
            new_grad = torch.zeros_like(self.weights)
            new_grad[: old_weights_grad.shape[0]] = old_weights_grad
            self.weights.grad = new_grad
        if old_bias_grad is not None:
            new_grad = torch.zeros_like(self.bias)
            new_grad[: old_bias_grad.shape[0]] = old_bias_grad
            self.bias.grad = new_grad

        self.added_times = torch.cat(
            [
                self.added_times,
                torch.full((n_neurons,), self.current_time, device=device),
            ]
        )

    def expand_input(self, n_features: int):
        device = self.weights.device

        mean_weight = self.weights.mean(dim=0, keepdim=True)
        new_weights = mean_weight.repeat(n_features, 1) + torch.randn(
            n_features, self.in_features, device=device
        )
        new_bias = torch.zeros(n_features, device=device)

        with torch.no_grad():
            combined_weights = torch.cat([self.weights, new_weights], dim=1)
            combined_bias = torch.cat([self.bias, new_bias], dim=1)

        old_weights_grad = (
            self.weights.grad.clone() if self.weights.grad is not None else None
        )
        old_bias_grad = self.bias.grad.clone() if self.bias.grad is not None else None

        weights = nn.Parameter(combined_weights)
        bias = nn.Parameter(combined_bias)
        self.in_features += n_features

        if "weights" in self._parameters:
            self._parameters.pop("weights")
        if "bias" in self._parameters:
            self._parameters.pop("bias")

        self.register_parameter("weights", weights)
        self.register_parameter("bias", bias)

        if old_weights_grad is not None:
            new_grad = torch.zeros_like(self.weights)
            new_grad[: old_weights_grad.shape[0]] = old_weights_grad
            self.weights.grad = new_grad
        if old_bias_grad is not None:   
            new_grad = torch.zeros_like(self.bias)
            new_grad[: old_bias_grad.shape[0]] = old_bias_grad
            self.bias.grad = new_grad

        self.added_times = torch.cat(
            [
                self.added_times,
                torch.full((n_features,), self.current_time, device=device),
            ]
        )

    def prune(self, indices):
        if len(indices) == 0:
            return

        device = self.weights.device
        indices = torch.tensor(indices, dtype=torch.long, device=device)

        mask = torch.ones(self.out_features, dtype=torch.bool, device=device)
        mask[indices] = False

        old_weights_grad = (
            self.weights.grad.clone() if self.weights.grad is not None else None
        )
        old_bias_grad = self.bias.grad.clone() if self.bias.grad is not None else None

        with torch.no_grad():
            weights = nn.Parameter(self.weights[mask])
            bias = nn.Parameter(self.bias[mask])
            self.added_times = self.added_times[mask]

            self.out_features = self.out_features - len(indices)

        if "weights" in self._parameters:
            self._parameters.pop("weights")
        if "bias" in self._parameters:
            self._parameters.pop("bias")

        self.register_parameter("weights", weights)
        self.register_parameter("bias", bias)

        if old_weights_grad is not None:
            self.weights.grad = old_weights_grad[mask].clone()
        if old_bias_grad is not None:
            self.bias.grad = old_bias_grad[mask].clone()

        if self._activation_usage is not None:
            self._activation_usage = self._activation_usage[mask]
        if self._activation_sum is not None:
            self._activation_sum = self._activation_sum[mask]
        if self._last_activation_usage is not None:
            self._last_activation_usage = self._last_activation_usage[mask]
        if self._last_activation_sum is not None:
            self._last_activation_sum = self._last_activation_sum[mask]

    @torch.no_grad()
    def compute_temporal_weights(self):
        device = self.added_times.device
        t = torch.full_like(
            self.added_times, self.current_time, dtype=torch.float32, device=device
        )
        decay = torch.exp(-self.lambda_decay * (t - self.added_times))
        return decay

    @torch.no_grad()
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

    @torch.no_grad()
    def compute_sum_activation(self, x):
        batch_sum = x.detach().sum(dim=0)

        if self._activation_sum is None:
            self._activation_sum = torch.zeros_like(batch_sum)

        self._activation_sum += batch_sum
        self._counter += x.shape[0]

    @torch.no_grad()
    def compute_activation_usage(self, x, threshold=1e-3):
        activated = (x.detach() > threshold).float().sum(dim=0)

        if self._activation_usage is None:
            self._activation_usage = torch.zeros_like(activated)

        self._activation_usage += activated

    @torch.no_grad()
    def reset_state(self):
        self._last_activation_sum = self._activation_sum
        self._activation_sum = None
        self._last_activation_usage = self._activation_usage
        self._activation_usage = None
        self._last_counter = self._counter
        self._counter = 0

    @torch.no_grad()
    def get_activation_mean(self):
        if self._activation_sum is not None and self._counter > 0:
            return self._activation_sum / self._counter
        elif self._last_activation_sum is not None and self._last_counter > 0:
            return self._last_activation_sum / self._last_counter
        else:
            return torch.zeros_like(self._activation_sum)

    @torch.no_grad()
    def get_activation_usage_ratio(self):
        if self._activation_usage is not None and self._counter > 0:
            return self._activation_usage / self._counter
        elif self._last_activation_usage is not None and self._last_counter > 0:
            return self._last_activation_usage / self._last_counter
        else:
            return torch.zeros_like(self._activation_usage)
