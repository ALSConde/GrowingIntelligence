"""Resume a varredura do joint: acurácia final (cabeça linear e protótipos), treino x teste."""
import glob, json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_joint"
print(f"{'variante':22s} {'n':>2s} {'linear':>13s} {'protótipos':>13s} {'treino':>7s} {'params(M)':>9s}")
for d in sorted(glob.glob(os.path.join(root, "*"))):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "seed*.json")))]
    if not runs:
        continue
    lin = np.array([r["summary"].get("final_acc_seen_linear", r["summary"]["final_acc_seen"]) for r in runs]) * 100
    ncm = [r["summary"].get("final_acc_seen_ncm") for r in runs]
    ncm_s = "-" if ncm[0] is None else f"{100*np.mean(ncm):5.1f}±{100*np.std(ncm, ddof=1) if len(ncm)>1 else 0:4.1f}"
    tr = np.mean([r["epoch_log"][-1]["train_acc"] for r in runs]) * 100
    par = np.mean([r["summary"]["params_final"] for r in runs]) / 1e6
    sd = lin.std(ddof=1) if len(lin) > 1 else 0.0
    print(f"{os.path.basename(d):22s} {len(runs):2d} {lin.mean():6.1f}±{sd:4.1f}   {ncm_s:>13s} {tr:7.1f} {par:9.2f}")
