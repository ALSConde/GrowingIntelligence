import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate


class LwFPlugin(SupervisedPlugin):
    def __init__(
        self,
        beta=1.0,
        temperature=2.0,
    ):
        super().__init__()
        self.beta = beta
        self.temperature = temperature
        self.teacher_model = None

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if hasattr(strategy, "model_old"):
            self.teacher_model = getattr(strategy, "model_old")
            self.teacher_model.eval()

            active_units = getattr(self.teacher_model.classifier, "active_units", None)

            if active_units is not None:
                active_units = torch.as_tensor(active_units, device=strategy.device)

                self.active_units = active_units.nonzero().squeeze(1)
            else:
                self.active_units = None

        else:
            self.teacher_model = None
            self.active_units = None

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.teacher_model is None:
            return

        x = strategy.mb_x
        out_t = self.teacher_model(x)

        ld = self._distillation_loss(strategy.mb_output, out_t)

        strategy.loss += self.beta * ld


    # ----------------- AUX -----------------
    def _distillation_loss(self, out_s, out_t):
        T = self.temperature

        q = F.softmax(out_t / T, dim=1)
        log_p = F.log_softmax(out_s / T, dim=1)

        if self.active_units is not None:
            q = q[:, self.active_units]
            log_p = log_p[:, self.active_units]

        else:
            q, log_p = q, log_p

        return F.kl_div(log_p, q, reduction="batchmean") * (T**2)
