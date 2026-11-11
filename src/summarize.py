"""Compara métodos a partir dos diretórios de resultados.

python summarize.py results/we_att_lwf results/cnn_lwf --baseline results/cnn_lwf
"""

import argparse
import csv
import os

from experiment.summary import aggregate, fmt, paired_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument(
        "--baseline",
        default=None,
        help="diretório de referência para a diferença pareada",
    )
    p.add_argument("--out", default="comparison.csv")
    a = p.parse_args()

    rows = []
    head = [
        "método",
        "runs",
        "acc final (%)",
        "acc média incr. (%)",
        "esquecimento (pp)",
        "BWT (pp)",
        "params antes",
        "params depois",
        "tempo (s)",
    ]
    if a.baseline:
        head += ["Δ acc final vs base (pp)", "vitórias", "Δ acc média incr. (pp)"]
    for d in a.dirs:
        agg = aggregate(d)
        if agg is None:
            print(f"(sem resultados em {d})")
            continue
        row = [
            os.path.basename(os.path.normpath(d)),
            agg["n_runs"],
            fmt(agg["final_acc_seen"]),
            fmt(agg["avg_incremental_acc"]),
            (
                fmt(agg["forgetting"], 100.0)
                if agg["forgetting"][0] == agg["forgetting"][0]
                else "-"
            ),
            fmt(agg["bwt"], 100.0) if agg["bwt"][0] == agg["bwt"][0] else "-",
            f"{agg['params_initial'][0]:,.0f}",
            f"{agg['params_final'][0]:,.0f}",
            f"{agg['seconds'][0]:.0f}",
        ]
        if a.baseline:
            d1 = paired_diff(d, a.baseline, "final_acc_seen")
            d2 = paired_diff(d, a.baseline, "avg_incremental_acc")
            row += [
                f"{d1['mean']:+.2f} ± {d1['std']:.2f}" if d1 else "-",
                f"{d1['wins']}/{d1['n']}" if d1 else "-",
                f"{d2['mean']:+.2f} ± {d2['std']:.2f}" if d2 else "-",
            ]
        rows.append(row)

    widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print(f"\nsalvo em {a.out}")


if __name__ == "__main__":
    main()
