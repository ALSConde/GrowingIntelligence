"""Acha o --width_mult da `ecnn` (CNN de largura fixa) cujo nº de parâmetros iguala o de um
método com expansão: controle de capacidade para saber se o ganho vem da expansão dinâmica
ou apenas de ter mais parâmetros.

    python match_width.py --results results/we2_lwf     # usa params finais médios das seeds
    python match_width.py --target 3200000              # ou um alvo explícito
"""
import argparse

from experiment.summary import aggregate
from models.Model_ECNN import ExpandableCNN


def total_params(mult, n_classes=100):
    m = ExpandableCNN(width_mult=mult)
    return m.extractor_params() + m.widths[6] * n_classes + n_classes


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--results", help="diretório de resultados de um método com expansão")
    g.add_argument("--target", type=float, help="nº de parâmetros alvo (com classificador de 100 classes)")
    p.add_argument("--n_classes", type=int, default=100)
    a = p.parse_args()
    target = a.target
    if a.results:
        agg = aggregate(a.results)
        if agg is None:
            raise SystemExit(f"sem resultados em {a.results}")
        target = agg["params_final"][0]
    lo, hi = 0.25, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if total_params(mid, a.n_classes) < target else (lo, mid)
    mult = round((lo + hi) / 2, 3)
    print(f"alvo: {target:,.0f} parâmetros -> --width_mult {mult}  "
          f"(ecnn resultante: {total_params(mult, a.n_classes):,} parâmetros)")


if __name__ == "__main__":
    main()
