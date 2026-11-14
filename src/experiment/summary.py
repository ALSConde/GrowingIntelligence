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


def _nan_stats(rows):
    a = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
    cnt = (~np.isnan(a)).sum(0)
    mean = np.where(cnt > 0, np.nansum(a, 0) / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1, np.nansum((a - mean) ** 2, 0) / np.maximum(cnt - 1, 1), 0.0)
    return mean.tolist(), np.sqrt(var).tolist(), cnt.tolist()


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
    if all("per_class_forgetting" in r["summary"] for r in runs):
        for key in (
            "per_class_learned",
            "per_class_forgetting",
            "per_class_drop",
            "class_learned_step",
        ):
            m, s, c = _nan_stats([r["summary"][key] for r in runs])
            agg[key + "_mean"], agg[key + "_std"] = m, s
            if key == "per_class_forgetting":
                agg["per_class_forgetting_n"] = c

        T = len(runs[0]["summary"]["acc_seen_per_step"])
        by_step = []
        for r in runs:
            f, ls = (
                r["summary"]["per_class_forgetting"],
                r["summary"]["class_learned_step"],
            )
            by_step.append(
                [
                    (
                        float(
                            np.mean(
                                [
                                    f[c]
                                    for c in range(len(f))
                                    if ls[c] == j and f[c] is not None
                                ]
                            )
                        )
                        if any(ls[c] == j and f[c] is not None for c in range(len(f)))
                        else None
                    )
                    for j in range(T)
                ]
            )
        m, s, _ = _nan_stats(by_step)
        (
            agg["forgetting_by_learned_step_mean"],
            agg["forgetting_by_learned_step_std"],
        ) = (m, s)
    steps = np.array([r["summary"]["acc_seen_per_step"] for r in runs], dtype=float)
    params = np.array(
        [[s["params_after"] for s in r["steps"]] for r in runs], dtype=float
    )
    agg["params_per_step_mean"] = params.mean(0).tolist()
    agg["params_per_step_std"] = (
        params.std(0, ddof=1) if len(runs) > 1 else np.zeros(params.shape[1])
    ).tolist()
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

    def f4(v):
        return "" if v is None or v != v else f"{v:.4f}"

    has_forg = "per_class_forgetting_mean" in agg
    with open(os.path.join(result_dir, "per_class_final.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["class", "acc_mean", "acc_std", "n_runs"]
        if has_forg:
            head += [
                "acc_learned_mean",
                "acc_learned_std",
                "forgetting_mean",
                "forgetting_std",
                "n_forgetting",
                "drop_mean",
                "drop_std",
                "learned_step_mean",
            ]
        w.writerow(head)
        for c in range(len(agg["per_class_mean"])):
            row = [
                c,
                f4(agg["per_class_mean"][c]),
                f4(agg["per_class_std"][c]),
                agg["n_runs"],
            ]
            if has_forg:
                row += [
                    f4(agg["per_class_learned_mean"][c]),
                    f4(agg["per_class_learned_std"][c]),
                    f4(agg["per_class_forgetting_mean"][c]),
                    f4(agg["per_class_forgetting_std"][c]),
                    agg["per_class_forgetting_n"][c],
                    f4(agg["per_class_drop_mean"][c]),
                    f4(agg["per_class_drop_std"][c]),
                    f4(agg["class_learned_step_mean"][c]),
                ]
            w.writerow(row)
    with open(os.path.join(result_dir, "acc_per_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["step", "acc_seen_mean", "acc_seen_std", "params_mean", "params_std"]
        if has_forg:
            head += [
                "forgetting_of_classes_learned_here_mean",
                "forgetting_of_classes_learned_here_std",
            ]
        w.writerow(head)
        for t in range(len(agg["acc_seen_per_step_mean"])):
            row = [
                t,
                f4(agg["acc_seen_per_step_mean"][t]),
                f4(agg["acc_seen_per_step_std"][t]),
                f"{agg['params_per_step_mean'][t]:.0f}",
                f"{agg['params_per_step_std'][t]:.0f}",
            ]
            if has_forg:
                row += [
                    f4(agg["forgetting_by_learned_step_mean"][t]),
                    f4(agg["forgetting_by_learned_step_std"][t]),
                ]
            w.writerow(row)
    return agg


def fmt(ms, scale=100.0, nd=2):
    m, s = ms
    return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}"


def paired_diff(a_dir, b_dir, key="final_acc_seen"):
    """Diff paired between two methods, using the same seeds. Returns None if no common seeds."""
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
