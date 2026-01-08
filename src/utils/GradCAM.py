import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

    def forward_hook(self, module, inp, out):
        self.activations = out
        self.activations.requires_grad_(True)

    def compute_cam(self, x, class_idx) -> torch.Tensor:
        """
        x: input batch [B, ...]
        class_idx: índice da classe de interesse [B]
        """
        handle = self.target_layer.register_forward_hook(self.forward_hook)

        outputs = self.model(x)  # [B, num_classes]

        batch_size = outputs.size(0)
        scores = outputs[torch.arange(batch_size), class_idx]  # [B]

        grads = torch.autograd.grad(
            outputs=scores,
            inputs=self.activations,
            grad_outputs=torch.ones_like(scores),
            create_graph=False,
            retain_graph=False,
        )[
            0
        ]  # [B, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        cam = F.relu((weights * self.activations).sum(dim=1))  # [B, H, W]

        handle.remove()
        return cam
