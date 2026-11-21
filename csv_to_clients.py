# csv_to_clients.py
# Builds the 13 FLAD clients (one DDoS attack each + benign traffic) from the
# CIC-DDoS2019 CSV / Parquet files (e.g. the Kaggle copy "dhoogla/cicddos2019"),
# and saves them in the SAME hdf5 format that the authors' code reads
# (set_x / set_y, files named <tw>t-<n>n-<name>-dataset-<train|val|test>.hdf5).
#
# The authors' FL code (flad_main.py, flad_training.py, ...) is NOT modified:
# the MLP starts with a Flatten layer, so it accepts any input shape.
#
# Usage:
#   python csv_to_clients.py --input /kaggle/input/cicddos2019 --output Dataset/CSV-DDoS2019
#   python csv_to_clients.py --input ... --output ... --max_samples 40000 --seed 0
#   python csv_to_clients.py --input ... --output ... --keep_identifiers   (pitfall P4 test)

import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd

# Paper, Table 3: total samples per client (attack + benign, about 50/50),
# doubling from one client to the next. Order = client order in the paper.
TABLE3_SAMPLES = [
    ("WebDDoS", 402), ("LDAP", 854), ("Portmap", 1605), ("DNS", 3207),
    ("UDPLag", 6400), ("NTP", 12807), ("SNMP", 25649), ("SSDP", 51207),
    ("Syn", 102400), ("TFTP", 204800), ("UDP", 409601), ("NetBIOS", 819200),
    ("MSSQL", 1638404),
]

# Columns that identify a specific capture rather than attack behaviour
# (removed by default to avoid shortcut learning, Arp et al. pitfall P4).
IDENTIFIER_COLUMNS = {
    "unnamed0", "flowid", "sourceip", "srcip", "destinationip", "dstip",
    "sourceport", "srcport", "destinationport", "dstport", "timestamp",
    "simillarhttp", "similarhttp", "inbound",
}


def norm(name):
    """Lower-case and keep only letters/digits: 'Flow ID' -> 'flowid'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def canonical_label(raw):
    """Map the many CIC label spellings to one name per attack."""
    s = norm(raw)
    s = s.replace("drdos", "")
    mapping = {
        "benign": "BENIGN", "webddos": "WebDDoS", "ldap": "LDAP",
        "portmap": "Portmap", "dns": "DNS", "udplag": "UDPLag",
        "ntp": "NTP", "snmp": "SNMP", "ssdp": "SSDP", "syn": "Syn",
        "tftp": "TFTP", "udp": "UDP", "netbios": "NetBIOS", "mssql": "MSSQL",
    }
    return mapping.get(s, None)


def read_all(input_folder):
    files = sorted(glob.glob(os.path.join(input_folder, "**", "*.parquet"), recursive=True))
    files += sorted(glob.glob(os.path.join(input_folder, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No .parquet or .csv files found under {input_folder}")
    frames = []
    for f in files:
        print("Reading", f)
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["__source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_label_column(df):
    for c in df.columns:
        if norm(c) == "label":
            return c
    raise SystemExit("No 'Label' column found. Columns are: " + ", ".join(df.columns))


def split_indices(n, rng, test_frac=0.10, val_frac=0.10):
    """Paper: 90% train / 10% test, then 10% of train for validation."""
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test, rest = idx[:n_test], idx[n_test:]
    n_val = int(round(len(rest) * val_frac))
    return rest[n_val:], rest[:n_val], test  # train, val, test


def save_h5(path, X, y):
    with h5py.File(path, "w") as h:
        h.create_dataset("set_x", data=X.astype(np.float32))
        h.create_dataset("set_y", data=y.astype(np.int64))


def main():
    ap = argparse.ArgumentParser(description="Build FLAD clients from CIC-DDoS2019 CSV/Parquet files")
    ap.add_argument("--input", required=True, help="Folder with the CSV/Parquet files")
    ap.add_argument("--output", required=True, help="Output folder (one subfolder per client)")
    ap.add_argument("--max_samples", type=int, default=40000,
                    help="Cap on total samples per client (Table 3 sizes are used up to this cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_identifiers", action="store_true",
                    help="Keep IP/port/timestamp/flow-id columns (only for the P4 pitfall test)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = read_all(args.input)
    label_col = find_label_column(df)
    df["__label"] = df[label_col].map(canonical_label)
    unknown = df.loc[df["__label"].isna(), label_col].unique()
    if len(unknown):
        print("Ignoring rows with labels not used in the paper:", list(unknown))
    df = df[df["__label"].notna()].reset_index(drop=True)

    # ---- feature columns ----
    drop = {label_col, "__label", "__source_file"}
    for c in df.columns:
        if norm(c) in {"class", "label"}:
            drop.add(c)
        if not args.keep_identifiers and norm(c) in IDENTIFIER_COLUMNS:
            drop.add(c)
    feat_cols = [c for c in df.columns if c not in drop]
    X_all = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    non_numeric = [c for c in feat_cols if X_all[c].isna().all()]
    X_all = X_all.drop(columns=non_numeric)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    feat_cols = [c for c in feat_cols if c not in non_numeric]
    labels = df["__label"].to_numpy()

    counts = pd.Series(labels).value_counts()
    print("\nRows available per label:\n" + counts.to_string())

    # ---- client sizes: Table 3, capped; limited by available attack rows ----
    benign_idx = rng.permutation(np.where(labels == "BENIGN")[0])
    def make_plan(cap):
        plan = []
        for name, total in TABLE3_SAMPLES:
            n = min(min(total, cap) // 2, int(counts.get(name, 0)))
            if n == 0:
                raise SystemExit(f"No rows for attack {name}: check the input files")
            plan.append((name, n, n))  # name, attack rows, benign rows (50/50)
        return plan

    # Each client gets its OWN benign rows (no benign row is shared between clients).
    # If there is not enough benign traffic, lower the cap (keeps 50/50 and the doubling
    # of the small clients) instead of unbalancing the clients.
    cap = args.max_samples
    plan = make_plan(cap)
    while sum(p[2] for p in plan) > len(benign_idx) and cap > 2:
        cap = int(cap * 0.95)
        plan = make_plan(cap)
    if cap != args.max_samples:
        print(f"\nNOTE: only {len(benign_idx)} benign rows available; "
              f"cap lowered from {args.max_samples} to {cap} samples per client.")
    if sum(p[2] for p in plan) > len(benign_idx):
        raise SystemExit("Not enough benign rows even with the smallest cap")

    # ---- build clients ----
    clients = {}
    start = 0
    for i, (name, n_attack, n_benign) in enumerate(plan):
        a_idx = rng.choice(np.where(labels == name)[0], size=n_attack, replace=False)
        b_idx = benign_idx[start:start + n_benign]
        start += n_benign
        idx = np.concatenate([a_idx, b_idx])
        y = np.concatenate([np.ones(n_attack), np.zeros(n_benign)])
        # split attack and benign separately so each split keeps the same ratio
        parts = {"train": [], "val": [], "test": []}
        for cls_positions in (np.arange(n_attack), np.arange(n_attack, n_attack + n_benign)):
            tr, va, te = split_indices(len(cls_positions), rng)
            parts["train"].append(cls_positions[tr])
            parts["val"].append(cls_positions[va])
            parts["test"].append(cls_positions[te])
        clients[f"{i:02d}-{name}"] = {
            s: (idx[np.concatenate(p)], y[np.concatenate(p)]) for s, p in parts.items()
        }

    # ---- scaling: min-max fitted on TRAINING rows only, then clipped to [0, 1] ----
    train_rows = np.concatenate([c["train"][0] for c in clients.values()])
    mins = X_all[train_rows].min(axis=0)
    maxs = X_all[train_rows].max(axis=0)
    keep = maxs > mins  # drop columns that are constant in the training data
    rng_ = np.where(keep, maxs - mins, 1.0)
    final_features = [f for f, k in zip(feat_cols, keep) if k]

    os.makedirs(args.output, exist_ok=True)
    summary = []
    for cname, sets in clients.items():
        folder = os.path.join(args.output, cname)
        os.makedirs(folder, exist_ok=True)
        row = {"client": cname}
        for s, (rows, y) in sets.items():
            X = np.clip((X_all[rows] - mins) / rng_, 0.0, 1.0)[:, keep]
            X = X.reshape(len(rows), 1, X.shape[1])  # (samples, 1 "packet", features)
            perm = rng.permutation(len(y))
            save_h5(os.path.join(folder, f"1t-1n-CSV2019-dataset-{s}.hdf5"), X[perm], y[perm])
            row[s] = len(y)
            row[s + "_attack_share"] = round(float(y.mean()), 3)
        summary.append(row)

    pd.DataFrame(summary).to_csv(os.path.join(args.output, "clients_summary.csv"), index=False)
    with open(os.path.join(args.output, "features.json"), "w") as f:
        json.dump({"features": final_features, "keep_identifiers": args.keep_identifiers,
                   "max_samples_requested": args.max_samples, "max_samples_used": cap,
                   "seed": args.seed}, f, indent=2)
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\n{len(final_features)} features used. Clients written to {args.output}")


if __name__ == "__main__":
    main()
