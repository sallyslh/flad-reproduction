# run_csv_experiments.py
# Runs the paper's main comparison (Tables 5 and 6) on the CSV-built clients:
# FLAD first, then FedAvg (E=1, E=5) and FLDDoS for the same number of rounds,
# exactly as described in Section 8.1.1 of the paper, repeated over several seeds.
#
# Usage:
#   python run_csv_experiments.py --clients Dataset/CSV-DDoS2019 --seeds 0 1 2 --out results/main

import argparse
import glob
import os
import re
import subprocess
import sys

import pandas as pd

METHODS = ["flad", "fedavg_1e", "fedavg_5e", "flddos"]


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def rounds_of(run_dir):
    """FLAD writes '...-rounds-<N>.csv' when it stops; read N from the name."""
    files = glob.glob(os.path.join(run_dir, "*-rounds-*.csv"))
    return int(re.search(r"-rounds-(\d+)\.csv$", files[0]).group(1))


def summarise(run_dir, method, seed):
    t = pd.read_csv(os.path.join(run_dir, "federated-tuning.csv"))
    last = t.iloc[-1]
    row = {"method": method, "seed": seed, "rounds": int(last["Round"]),
           "round_time_s": t["Time(sec)"].mean(), "total_time_s": t["Time(sec)"].sum(),
           "F1_val": last["F1"], "F1_std_val": last["F1_std"]}
    for c in t.columns:
        if c.endswith("(f1)"):
            row["F1_val " + c[:-4]] = last[c]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary, tests = [], []
    for seed in args.seeds:
        n_rounds = None
        for method in args.methods:
            run_dir = os.path.join(args.out, f"{method}_seed{seed}") + "/"
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, "flad_main.py", "-c", args.clients, "-t", method,
                   "-o", run_dir, "--rn_seed", str(seed)]
            if method != "flad":
                if n_rounds is None:
                    raise SystemExit("Run 'flad' first: baselines use FLAD's number of rounds")
                cmd += ["-r", str(n_rounds)]
            run(cmd)
            if method == "flad":
                n_rounds = rounds_of(run_dir)
            summary.append(summarise(run_dir, method, seed))

            model = glob.glob(os.path.join(run_dir, "*-global-model.h5"))[0]
            test_csv = os.path.join(run_dir, "test_metrics.csv")
            run([sys.executable, "evaluate_test.py", "--clients", args.clients,
                 "--model", model, "--out", test_csv])
            t = pd.read_csv(test_csv)
            t["method"], t["seed"] = method, seed
            tests.append(t)

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.out, "table5_all_seeds.csv"), index=False)
    agg = s.drop(columns=["seed"]).groupby("method").agg(["mean", "std"])
    agg.to_csv(os.path.join(args.out, "table5_mean_std.csv"))

    t = pd.concat(tests, ignore_index=True)
    t.to_csv(os.path.join(args.out, "table6_all_seeds.csv"), index=False)
    t6 = t.pivot_table(index="client", columns="method", values="TPR", aggfunc="mean")
    t6.to_csv(os.path.join(args.out, "table6_TPR_mean.csv"))

    print("\n===== Table 5 (mean over seeds) =====")
    print(agg.xs("mean", axis=1, level=1).T.to_string())
    print("\n===== Table 6: test TPR (mean over seeds) =====")
    print(t6.to_string())


if __name__ == "__main__":
    main()
