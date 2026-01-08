from typing import Optional
import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from utils.GradCAM import GradCAM
from avalanche.logging.tensorboard_logger import SummaryWriter


class LwMPlugin(SupervisedPlugin):
    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        temperature=2.0,
        target_layer_name=None,
        writter: Optional[SummaryWriter] = None,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.target_layer_name = target_layer_name
        self.teacher_model = None
        self.gradcam_t: Optional[GradCAM] = None
        self.gradcam_s: Optional[GradCAM] = None
        self.active_units = None
        self.writer = writter
        self.graph_logged = False

        # buffers de perdas
        self.losses_epoch = {"LC": [], "LD": [], "LAD": []}
        self.epoch = 0
        self.exp_counter = 0

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """Define o modelo anterior como professor e prepara GradCAM."""

        if not self.graph_logged and self.writer is not None:
            example_input = torch.randn(1, 3, 32, 32).to(strategy.device)
            self.writer.add_graph(strategy.model, example_input)
            self.graph_logged = True

        if hasattr(strategy, "model_old"):
            self.teacher_model = getattr(strategy, "model_old")
            self.teacher_model.eval()

            self.gradcam_t = self._create_gradcam(self.teacher_model)
            self.gradcam_s = self._create_gradcam(strategy.model)

            active_units = getattr(self.teacher_model.classifier, "active_units", None)

            if active_units is not None:
                active_units = torch.as_tensor(active_units, device=strategy.device)

                self.active_units = active_units.nonzero().squeeze(1)
            else:
                self.active_units = None
        else:
            self.teacher_model = None
            self.active_units = None

        # limpa buffers a cada nova experiência
        if self.writer is not None:
            self.losses_epoch = {"LC": [], "LD": [], "LAD": []}

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        """Calcula LwM = LC + βLD + γLAD."""
        if self.teacher_model is None:
            return

        x = strategy.mb_x
        out_t = self.teacher_model(x)

        # Calcula LD e LAD
        ld = self._distillation_loss(strategy.mb_output, out_t)
        lad = 0.0
        if self.gradcam_s is not None and self.gradcam_t is not None:
            cam_t = self.gradcam_t.compute_cam(x.detach(), self.active_units)
            cam_s = self.gradcam_s.compute_cam(x.detach(), self.active_units)
            lad = self._attention_distillation_loss(cam_t, cam_s)

        # Soma ponderada
        strategy.loss += self.beta * ld + self.gamma * lad

        # Armazena os valores das perdas
        if self.writer is not None:
            self.losses_epoch["LC"].append(strategy.loss.detach().clone().item())
            self.losses_epoch["LD"].append(ld.item())
            if isinstance(lad, torch.Tensor):
                self.losses_epoch["LAD"].append(lad.item())

    def after_training_epoch(self, strategy: "SupervisedTemplate", *args, **kwargs):
        super().after_training_epoch(strategy, *args, **kwargs)

        if self.writer is not None and self.teacher_model is not None:
            for key in self.losses_epoch:
                avg_loss = sum(self.losses_epoch[key]) / len(self.losses_epoch[key])
                self.writer.add_scalar(
                    f"Exp{self.exp_counter}/{key}", avg_loss, self.epoch
                )
            self.writer.flush()
        self.epoch += 1

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """Incrementa o contador de experiências."""
        self.exp_counter += 1

    # ----------------- AUX -----------------
    def _distillation_loss(self, out_s, out_t: torch.Tensor) -> torch.Tensor:
        T = self.temperature

        p = F.softmax(out_t / T, dim=1)
        q = F.log_softmax(out_s / T, dim=1)

        if self.active_units is not None:
            p_old = p[:, self.active_units]
            q_old = q[:, self.active_units]

        else:
            p_old, q_old = p, q

        return F.kl_div(q_old, p_old, reduction="batchmean") * (T**2)

    def _attention_distillation_loss(
        self, cam_t: torch.Tensor, cam_s: torch.Tensor
    ) -> torch.Tensor:
        cam_t = cam_t / (torch.norm(cam_t, p=2, dim=(1, 2), keepdim=True) + 1e-8)
        cam_s = cam_s / (torch.norm(cam_s, p=2, dim=(1, 2), keepdim=True) + 1e-8)
        return torch.abs(cam_t - cam_s).sum(dim=(1, 2)).mean()

    def _create_gradcam(self, model: torch.nn.Module) -> GradCAM:
        target_layer = self._find_target_layer(model)
        return GradCAM(model, target_layer)

    def _find_target_layer(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.target_layer_name is not None:
            layers = dict(model.named_modules())
            if self.target_layer_name not in layers:
                raise ValueError(
                    f"Camada {self.target_layer_name} não encontrada no modelo."
                )
            return layers[self.target_layer_name]

        last_conv = None
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("Nenhuma camada Conv2d encontrada para o GradCAM.")
        return last_conv
