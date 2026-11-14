"""Roda uma matriz de experimentos (arquiteturas x estratégias), 10 seeds cada, com retomada
automática (seeds já salvas são puladas). Argumentos desconhecidos são repassados ao
run_experiment.py (ex.: --seeds 1 2 3, --smoke, --class_order fixed, --budget_per_step 0.1).

    python run_matrix.py --dataset cifar100 --archs rcnn rwe rcnn_att rwe_att \
                         --strategies naive ewc si lwf lwm er agem
    python run_matrix.py --dataset cifar100 --strategies der orth_der --extra_prune
    python run_matrix.py --dataset mnist --archs mlp mwe mlp_att mwe_att --strategies naive lwf er agem

der/orth_der não dependem de --arch (usam --backbone), então rodam uma vez por execução.
"""

import argparse
import run_experiment
from experiment.builders import DER_FAMILY


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=["cifar100", "mnist"], default="cifar100")
    p.add_argument("--archs", nargs="*", default=[])
    p.add_argument("--strategies", nargs="+", required=True)
    p.add_argument(
        "--extra_prune", action="store_true", help="der/orth_der com --prune"
    )
    a, extra = p.parse_known_args()

    jobs = []
    for strat in a.strategies:
        if strat in DER_FAMILY:
            jobs.append(["--strategy", strat] + (["--prune"] if a.extra_prune else []))
        else:
            if not a.archs:
                raise SystemExit(
                    "informe --archs para as estratégias que não são der/orth_der"
                )
            jobs += [["--strategy", strat, "--arch", arch] for arch in a.archs]
    print(f"[matriz] {len(jobs)} configurações x seeds")
    for job in jobs:
        run_experiment.main(["--dataset", a.dataset] + job + extra)


if __name__ == "__main__":
    main()
