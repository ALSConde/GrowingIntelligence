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
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import run_experiment
from experiment.builders import DER_FAMILY


def _split_seeds(extra):
    """Separa '--seeds ...' dos demais argumentos repassados; sem --seeds -> 1..10."""
    if "--seeds" not in extra:
        return list(range(1, 11)), list(extra)
    i = extra.index("--seeds")
    j = i + 1
    while j < len(extra) and not extra[j].startswith("--"):
        j += 1
    return [int(s) for s in extra[i + 1 : j]], extra[:i] + extra[j:]


def _job_name(job):
    return "_".join(x.strip("-") for x in job)


def _run_parallel(dataset, jobs, extra, n_workers, log_dir):
    """Uma execução por (configuração, seed), até n_workers ao mesmo tempo na mesma GPU."""
    seeds, rest = _split_seeds(extra)
    if dataset == "cifar100" and "--smoke" not in rest:
        # baixa o CIFAR-100 uma vez, antes, para os processos não disputarem o download
        from torchvision.datasets import CIFAR100

        for train in (True, False):
            CIFAR100(root="./data/cifar100", train=train, download=True)
    os.makedirs(log_dir, exist_ok=True)
    cmds = []
    for job in jobs:
        for s in seeds:
            cmd = (
                [sys.executable, "run_experiment.py", "--dataset", dataset] + job + rest
            )
            cmd += ["--seeds", str(s), "--no_aggregate"]
            cmds.append((f"{_job_name(job)}_seed{s}", cmd))

    def go(item):
        name, cmd = item
        with open(os.path.join(log_dir, name + ".log"), "w") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        print(
            f"[paralelo] {name}: {'ok' if code == 0 else f'FALHOU (código {code}); veja {log_dir}'}"
        )
        return code

    print(
        f"[paralelo] {len(cmds)} execuções, {n_workers} simultâneas; logs em {log_dir}/"
    )
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        codes = list(ex.map(go, cmds))
    if any(codes):
        print(
            f"[paralelo] {sum(1 for c in codes if c)} falharam; a agregação abaixo as refaz em série"
        )


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
    p.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="nº de execuções simultâneas na mesma GPU (cada uma é uma seed de uma configuração)",
    )
    p.add_argument("--log_dir", default="logs")
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
    if a.parallel > 1:
        _run_parallel(a.dataset, jobs, extra, a.parallel, a.log_dir)
    # em série (ou, após o paralelo, só para agregar: as seeds prontas são puladas e as que
    # falharam são refeitas aqui)
    for job in jobs:
        run_experiment.main(["--dataset", a.dataset] + job + extra)


if __name__ == "__main__":
    main()
