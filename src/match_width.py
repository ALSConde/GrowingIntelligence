"""Acha o --width_mult da `ecnn` (CNN de largura fixa) cujo nº de parâmetros iguala o de um
método com expansão: controle de capacidade para saber se o ganho vem da expansão dinâmica
ou apenas de ter mais parâmetros.

    python match_width.py --results results/we2_lwf     # usa params finais médios das seeds
    python match_width.py --target 3200000              # ou um alvo explícito
    python match_width.py --target_model rcnn           # CNN sem saltos com os parâmetros da rcnn
    python match_width.py --results results/mnist/mwe_lwf --family mlp --n_classes 10   # MNIST

Receita v2 (pooling global + embedding fixo): informe a mesma arquitetura do método com expansão,
senão a contagem de parâmetros do baseline de largura fixa sai errada:
    python match_width.py --results results/cifar100/rwe_att_lwf_v2 --family rcnn_att --pool gap --emb_dim 128
"""

import argparse

from experiment.summary import aggregate
from models.Model_ECNN import ExpandableCNN, ExpandableCNNAttention
from models.Model_EMLP import ExpandableMLP
from models.Model_RCNN import ResExpandableCNN

FAMILIES = {
    "ecnn": lambda m, **kw: ExpandableCNN(width_mult=m),
    "ecnn_att": lambda m, **kw: ExpandableCNNAttention(width_mult=m),
    "rcnn": lambda m, **kw: ResExpandableCNN(width_mult=m, **kw),
    "rcnn_att": lambda m, **kw: ResExpandableCNN(width_mult=m, attention=True, **kw),
    "mlp": lambda m, **kw: ExpandableMLP(width_mult=m),
    "mlp_att": lambda m, **kw: ExpandableMLP(width_mult=m, attention=True),
}


def total_params(mult, n_classes=100, family="ecnn", pool="flatten", emb_dim=0):
    kw = {"pool": pool, "emb_dim": emb_dim} if family.startswith("rcnn") else {}
    m = FAMILIES[family](mult, **kw)
    head_in = emb_dim if (emb_dim > 0 and family.startswith("rcnn")) else m.widths[-1]
    # extractor_params inclui a camada de embedding; o classificador lê o embedding
    return m.extractor_params() + head_in * n_classes + n_classes


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--results", help="diretório de resultados de um método com expansão"
    )
    g.add_argument(
        "--target",
        type=float,
        help="nº de parâmetros alvo (com classificador de 100 classes)",
    )
    p.add_argument("--n_classes", type=int, default=100)
    p.add_argument("--pool", choices=["flatten", "gap"], default="flatten")
    p.add_argument("--emb_dim", type=int, default=0)
    p.add_argument(
        "--family",
        choices=list(FAMILIES),
        default="ecnn",
        help="modelo de largura fixa cuja largura será ajustada",
    )
    g.add_argument(
        "--target_model",
        choices=list(FAMILIES),
        help="usa como alvo os parâmetros deste modelo (largura 1.0), ex.: rcnn",
    )
    a = p.parse_args()
    target = a.target
    if a.target_model:
        target = total_params(1.0, a.n_classes, a.target_model, a.pool, a.emb_dim)
    if a.results:
        agg = aggregate(a.results)
        if agg is None:
            raise SystemExit(f"sem resultados em {a.results}")
        target = agg["params_final"][0]
    lo, hi = 0.25, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (
            (mid, hi)
            if total_params(mid, a.n_classes, a.family, a.pool, a.emb_dim) < target
            else (lo, mid)
        )
    mult = round((lo + hi) / 2, 3)
    print(
        f"alvo: {target:,.0f} parâmetros -> --width_mult {mult}  "
        f"({a.family} resultante: "
        f"{total_params(mult, a.n_classes, a.family, a.pool, a.emb_dim):,} parâmetros)"
    )


if __name__ == "__main__":
    main()
