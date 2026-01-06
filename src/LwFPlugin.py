import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging.tensorboard_logger import SummaryWriter


class LwFPlugin(SupervisedPlugin):
    def __init__(
        self,
        beta=1.0,
        temperature=2.0,
        writter: SummaryWriter = None,
    ):
        super().__init__()
        self.beta = beta
        self.temperature = temperature
        self.teacher_model = None
        self.writer = writter
        self.graph_logged = False

        # buffers de perdas
        self.losses_epoch = {"LC": [], "LD": []}
        self.epoch = 0
        self.exp_counter = 0

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if not self.graph_logged and self.writer is not None:
            example_input = torch.randn(1, 3, 32, 32).to(strategy.device)
            self.writer.add_graph(strategy.model, example_input)
            self.graph_logged = True

        if hasattr(strategy, "model_old"):
            self.teacher_model = strategy.model_old
            self.teacher_model.eval()

            for param in self.teacher_model.parameters():
                param.requires_grad = False

            active_units = getattr(self.teacher_model.classifier, "active_units", None)

            if active_units is not None:
                active_units = torch.as_tensor(active_units, device=strategy.device)

                self.active_units = active_units.nonzero().squeeze(1)
            else:
                self.active_units = None

        else:
            self.teacher_model = None
            self.active_units = None

        if self.writer is not None:
            self.losses_epoch = {"LC": [], "LD": []}

    def before_backward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.teacher_model is None:
            return

        x = strategy.mb_x
        out_t = self.teacher_model(x)

        ld = self._distillation_loss(strategy.mb_output, out_t)

        if self.writer is not None:
            self.losses_epoch["LC"].append(strategy.loss.detach().item())
            self.losses_epoch["LD"].append(ld.item())

        strategy.loss += self.beta * ld

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
    def _distillation_loss(self, out_s, out_t):
        T = self.temperature

        p = F.softmax(out_t / T, dim=1)
        q = F.log_softmax(out_s / T, dim=1)

        if self.active_units is not None:
            p_old = p[:, self.active_units]
            q_old = q[:, self.active_units]

            # renormaliza
            # p_old = p_old / p_old.sum(dim=1, keepdim=True)
            # q_old = q_old - torch.logsumexp(q_old, dim=1, keepdim=True)
        else:
            p_old, q_old = p, q

        return F.kl_div(q_old, p_old, reduction="batchmean") * (T ** 2)

