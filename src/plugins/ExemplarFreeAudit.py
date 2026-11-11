import warnings

from avalanche.core import SupervisedPlugin

# nomes de atributos que costumam guardar exemplares (ReplayPlugin, A-GEM, buffers próprios)
_MEMORY_ATTRS = ("storage_policy", "buffer", "buffers", "ext_mem", "memory")


class ExemplarFreeAudit(SupervisedPlugin):
    """Auditoria de 'sem exemplares', para rodar junto com qualquer estratégia que deva
    ser livre de memória (Naive, EWC, SI, LwF, LwM, WE, WE+Atenção).

    Verificação rígida: se algum minibatch de treino tiver rótulo de classe fora da
    experiência corrente, levanta RuntimeError (isso só acontece se amostras de
    experiências anteriores estiverem sendo reapresentadas).

    Verificação heurística: avisa (warning) se a estratégia ou algum plugin tiver
    atributo com nome típico de buffer de exemplares. Pode dar falso positivo; nesse
    caso inspecione o objeto apontado.

    Ao fim de cada experiência imprime quantas amostras de treino foram vistas e de
    quais classes, para registrar no log do experimento.
    """

    def __init__(self):
        super().__init__()
        self._seen = 0
        self._classes_seen = set()

    def before_training(self, strategy, *args, **kwargs):
        holders = [
            type(o).__name__
            for o in [strategy, *strategy.plugins]
            if o is not self and any(hasattr(o, a) for a in _MEMORY_ATTRS)
        ]
        if holders:
            warnings.warn(f"[audit] Possible to have memory buffers: {holders}")

    def before_training_exp(self, strategy, *args, **kwargs):
        self._seen = 0
        self._classes_seen = set()

    def before_training_iteration(self, strategy, *args, **kwargs):
        current = set(strategy.experience.classes_in_this_experience)
        labels = set(strategy.mb_y.unique().tolist())
        extra = labels - current
        if extra:
            raise RuntimeError(
                f"[audit] minibatch with classes outside the current experience: {sorted(extra)} "
                f"(current: {sorted(current)}). There is replay of old data."
            )
        self._seen += len(strategy.mb_y)
        self._classes_seen |= labels

    def after_training_exp(self, strategy, *args, **kwargs):
        n = len(strategy.experience.dataset)
        print(
            f"[audit] exp {strategy.experience.current_experience}: "
            f"{self._seen} training samples seen ({strategy.train_epochs} epochs x {n}), "
            f"classes {sorted(self._classes_seen)}"
        )
