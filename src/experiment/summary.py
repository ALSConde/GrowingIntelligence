"""Agregação dos resultados de várias seeds e comparação pareada entre métodos."""

import csv
import glob
import json
import os

import numpy as np


def load_runs(result_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(result_dir, "seed*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def _ms(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def aggregate(result_dir):
    runs = load_runs(result_dir)
    if not runs:
        return None
    keys = [
        "final_acc_seen",
        "avg_incremental_acc",
        "forgetting",
        "bwt",
        "params_final",
    ]
    agg = {
        "dir": result_dir,
        "n_runs": len(runs),
        "seeds": [r["config"]["seed"] for r in runs],
    }
    for k in keys:
        agg[k] = _ms([r["summary"].get(k) for r in runs])
    agg["params_initial"] = _ms([r["params_initial"] for r in runs])
    agg["seconds"] = _ms([sum(s["seconds"] for s in r["steps"]) for r in runs])
    pc = np.array([r["summary"]["per_class_final"] for r in runs], dtype=float)
    agg["per_class_mean"] = pc.mean(0).tolist()
    agg["per_class_std"] = (
        pc.std(0, ddof=1) if len(runs) > 1 else np.zeros(pc.shape[1])
    ).tolist()
    steps = np.array([r["summary"]["acc_seen_per_step"] for r in runs], dtype=float)
    agg["acc_seen_per_step_mean"] = steps.mean(0).tolist()
    agg["acc_seen_per_step_std"] = (
        steps.std(0, ddof=1) if len(runs) > 1 else np.zeros(steps.shape[1])
    ).tolist()
    return agg


def write_aggregate(result_dir):
    agg = aggregate(result_dir)
    if agg is None:
        return None
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    with open(os.path.join(result_dir, "per_class_final.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "acc_mean", "acc_std", "n_runs"])
        for c, (m, s) in enumerate(zip(agg["per_class_mean"], agg["per_class_std"])):
            w.writerow([c, f"{m:.4f}", f"{s:.4f}", agg["n_runs"]])
    with open(os.path.join(result_dir, "acc_per_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "acc_seen_mean", "acc_seen_std"])
        for t, (m, s) in enumerate(
            zip(agg["acc_seen_per_step_mean"], agg["acc_seen_per_step_std"])
        ):
            w.writerow([t, f"{m:.4f}", f"{s:.4f}"])
    return agg


def fmt(ms, scale=100.0, nd=2):
    m, s = ms
    return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}"


def paired_diff(a_dir, b_dir, key="final_acc_seen"):
    """Diferença pareada por seed (a - b), em pontos percentuais."""
    ra = {r["config"]["seed"]: r["summary"][key] for r in load_runs(a_dir)}
    rb = {r["config"]["seed"]: r["summary"][key] for r in load_runs(b_dir)}
    common = sorted(set(ra) & set(rb))
    d = np.array([ra[s] - rb[s] for s in common]) * 100.0
    if len(d) == 0:
        return None
    return {
        "n": len(d),
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1) if len(d) > 1 else 0.0),
        "wins": int((d > 0).sum()),
    }
