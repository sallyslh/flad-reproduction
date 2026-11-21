# csv_to_clients.py
# Builds the 13 FLAD clients (one DDoS attack each + benign traffic) from the
# official CIC-DDoS2019 CSV files (CSV-01-12 and CSV-03-11 from the UNB website),
# and saves them in the SAME hdf5 format that the authors' code reads
# (set_x / set_y, files named <tw>t-<n>n-<name>-dataset-<train|val|test>.hdf5).
#
# The CSV files are several GB, so they are read in pieces (chunks), in two passes:
#   pass 1: read only the Label column, to count the rows of each attack;
#   pass 2: read the files again and keep only the randomly selected rows.
# Parquet files (e.g. a Kaggle copy) are also accepted.
#
# The authors' FL code is not changed for this: the MLP starts with a Flatten layer,
# so it accepts any input shape.
#
# Usage:
#   python csv_to_clients.py --input "D:\CIC-DDoS2019" --output Dataset/CSV-DDoS2019
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

CHUNK_ROWS = 200_000

# Paper, Table 3: total samples per client (attack + benign, about 50/50),
# doubling from one client to the next. Order = client order in the paper.
TABLE3_SAMPLES = [
    ("WebDDoS", 402), ("LDAP", 854), ("Portmap", 1605), ("DNS", 3207),
    ("UDPLag", 6400), ("NTP", 12807), ("SNMP", 25649), ("SSDP", 51207),
    ("Syn", 102400), ("TFTP", 204800), ("UDP", 409601), ("NetBIOS", 819200),
    ("MSSQL", 1638404),
]
LABELS = ["BENIGN"] + [name for name, _ in TABLE3_SAMPLES]
LABEL_CODE = {name: i for i, name in enumerate(LABELS)}  # -1 = label not used

# Columns that identify a specific capture rather than attack behaviour
# (removed by default to avoid shortcut learning, Arp et al. pitfall P4).
IDENTIFIER_COLUMNS = {
    "unnamed0", "flowid", "sourceip", "srcip", "destinationip", "dstip",
    "sourceport", "srcport", "destinationport", "dstport", "timestamp",
    "simillarhttp", "similarhttp", "inbound",
}


def norm(name):
    """Lower-case and keep only letters/digits: ' Flow ID' -> 'flowid'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def canonical_label(raw):
    """Map the many CIC label spellings (DrDoS_DNS, UDP-lag, ...) to one name per attack."""
    s = norm(raw).replace("drdos", "")
    mapping = {
        "benign": "BENIGN", "webddos": "WebDDoS", "ldap": "LDAP",
        "portmap": "Portmap", "dns": "DNS", "udplag": "UDPLag",
        "ntp": "NTP", "snmp": "SNMP", "ssdp": "SSDP", "syn": "Syn",
        "tftp": "TFTP", "udp": "UDP", "netbios": "NetBIOS", "mssql": "MSSQL",
    }
    return mapping.get(s)


def label_codes(series):
    # map each distinct label once (fast even for millions of rows)
    codes = {v: LABEL_CODE.get(canonical_label(v), -1) for v in series.unique()}
    return series.map(codes).to_numpy(np.int8)


def iter_chunks(path, usecols=None):
    """Yield DataFrames of at most CHUNK_ROWS rows, with stripped column names."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        df.columns = [c.strip() for c in df.columns]
        if usecols is not None:
            df = df[[c for c in df.columns if c in usecols]]
        for start in range(0, len(df), CHUNK_ROWS):
            yield df.iloc[start:start + CHUNK_ROWS]
    else:
        reader = pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False,
                             usecols=(lambda c: c.strip() in usecols) if usecols is not None else None)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            yield chunk


def find_label_column(path):
    header = pd.read_parquet(path).columns if path.endswith(".parquet") else pd.read_csv(path, nrows=0).columns
    for c in header:
        if norm(c) == "label":
            return c.strip(), [h.strip() for h in header]
    raise SystemExit(f"No 'Label' column in {path}")


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
    ap = argparse.ArgumentParser(description="Build FLAD clients from the CIC-DDoS2019 CSV files")
    ap.add_argument("--input", required=True, help="Folder with the CSV (or Parquet) files; subfolders are searched too")
    ap.add_argument("--output", required=True, help="Output folder (one subfolder per client)")
    ap.add_argument("--max_samples", type=int, default=40000,
                    help="Cap on total samples per client (Table 3 sizes are used up to this cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_identifiers", action="store_true",
                    help="Keep IP/port/timestamp/flow-id columns (only for the P4 pitfall test)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.csv"), recursive=True))
    files += sorted(glob.glob(os.path.join(args.input, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No .csv or .parquet files found under {args.input}")

    # ---------- pass 1: labels only ----------
    per_file_codes = []
    label_col, header = find_label_column(files[0])
    for f in files:
        lc, _ = find_label_column(f)
        print("Counting labels in", f, flush=True)
        codes = [label_codes(ch[lc]) for ch in iter_chunks(f, usecols={lc})]
        per_file_codes.append(np.concatenate(codes) if codes else np.zeros(0, np.int8))
    all_codes = np.concatenate(per_file_codes)
    file_of_row = np.concatenate([np.full(len(c), i, np.int32) for i, c in enumerate(per_file_codes)])
    row_in_file = np.concatenate([np.arange(len(c), dtype=np.int64) for c in per_file_codes])
    file_start = np.cumsum([0] + [len(c) for c in per_file_codes])  # global row number of each file's first row
    counts = {name: int((all_codes == LABEL_CODE[name]).sum()) for name in LABELS}
    print("\nRows available per label:")
    for name in LABELS:
        print(f"  {name:8s} {counts[name]:>12,}")
    print(f"  (rows with other labels, ignored: {int((all_codes == -1).sum()):,})")

    # ---------- client sizes: Table 3, capped, 50/50 ----------
    def make_plan(cap):
        plan = []
        for name, total in TABLE3_SAMPLES:
            n = min(min(total, cap) // 2, counts[name])
            if n == 0:
                raise SystemExit(f"No rows for attack {name}: check the input files")
            plan.append((name, n))
        return plan

    # Each client gets its OWN benign rows (no benign row is shared between clients).
    # If there is not enough benign traffic, lower the cap (keeps 50/50 and the doubling
    # of the small clients) instead of unbalancing the clients.
    cap = args.max_samples
    plan = make_plan(cap)
    while sum(n for _, n in plan) > counts["BENIGN"] and cap > 2:
        cap = int(cap * 0.95)
        plan = make_plan(cap)
    if sum(n for _, n in plan) > counts["BENIGN"]:
        raise SystemExit("Not enough benign rows even with the smallest cap")
    if cap != args.max_samples:
        print(f"\nNOTE: only {counts['BENIGN']:,} benign rows; cap lowered from "
              f"{args.max_samples} to {cap} samples per client.")

    # ---------- choose the rows (global row numbers) ----------
    benign_rows = rng.permutation(np.where(all_codes == LABEL_CODE["BENIGN"])[0])
    selection = {}  # client name -> (attack rows, benign rows)
    start = 0
    for name, n in plan:
        attack_rows = rng.choice(np.where(all_codes == LABEL_CODE[name])[0], size=n, replace=False)
        selection[name] = (attack_rows, benign_rows[start:start + n])
        start += n
    wanted = np.sort(np.concatenate([np.concatenate(v) for v in selection.values()]))

    # ---------- pass 2: read only the selected rows ----------
    drop = {"label", "class"} | (set() if args.keep_identifiers else IDENTIFIER_COLUMNS)
    feat_cols = [c for c in header if norm(c) not in drop]
    pieces, piece_rows = [], []
    for i, f in enumerate(files):
        rows_here = row_in_file[wanted[file_of_row[wanted] == i]]
        if len(rows_here) == 0:
            continue
        print("Reading selected rows from", f, flush=True)
        offset = 0
        for ch in iter_chunks(f):
            in_chunk = rows_here[(rows_here >= offset) & (rows_here < offset + len(ch))]
            if len(in_chunk):
                part = ch.iloc[in_chunk - offset].reindex(columns=feat_cols)
                pieces.append(part.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64))
                piece_rows.append(file_start[i] + in_chunk)
            offset += len(ch)
    X_sel = np.concatenate(pieces)
    X_sel[~np.isfinite(X_sel)] = 0  # inf / NaN -> 0
    pos = {int(r): k for k, r in enumerate(np.concatenate(piece_rows))}

    # ---------- build clients, split, scale ----------
    clients = {}
    for i, (name, n) in enumerate(plan):
        attack_rows, b_rows = selection[name]
        rows = np.concatenate([attack_rows, b_rows])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        parts = {"train": [], "val": [], "test": []}
        # split attack and benign separately so every split stays 50/50
        for cls_positions in (np.arange(n), np.arange(n, 2 * n)):
            tr, va, te = split_indices(len(cls_positions), rng)
            parts["train"].append(cls_positions[tr])
            parts["val"].append(cls_positions[va])
            parts["test"].append(cls_positions[te])
        clients[f"{i:02d}-{name}"] = {
            s: (np.array([pos[int(r)] for r in rows[np.concatenate(p)]]), y[np.concatenate(p)])
            for s, p in parts.items()
        }

    # min-max fitted on TRAINING rows only; val/test are clipped to [0, 1]
    train_pos = np.concatenate([c["train"][0] for c in clients.values()])
    mins, maxs = X_sel[train_pos].min(axis=0), X_sel[train_pos].max(axis=0)
    keep = maxs > mins  # drop columns that are constant (or empty) in the training data
    span = np.where(keep, maxs - mins, 1.0)
    final_features = [f for f, k in zip(feat_cols, keep) if k]

    os.makedirs(args.output, exist_ok=True)
    summary = []
    for cname, sets in clients.items():
        folder = os.path.join(args.output, cname)
        os.makedirs(folder, exist_ok=True)
        row = {"client": cname}
        for s, (p, y) in sets.items():
            X = np.clip((X_sel[p] - mins) / span, 0.0, 1.0)[:, keep]
            X = X.reshape(len(p), 1, X.shape[1])  # (samples, 1 "packet", features)
            perm = rng.permutation(len(y))
            save_h5(os.path.join(folder, f"1t-1n-CSV2019-dataset-{s}.hdf5"), X[perm], y[perm])
            row[s] = len(y)
            row[s + "_attack_share"] = round(float(y.mean()), 3)
        summary.append(row)

    pd.DataFrame(summary).to_csv(os.path.join(args.output, "clients_summary.csv"), index=False)
    with open(os.path.join(args.output, "features.json"), "w") as f:
        json.dump({"features": final_features, "keep_identifiers": args.keep_identifiers,
                   "max_samples_requested": args.max_samples, "max_samples_used": cap,
                   "seed": args.seed, "input_files": [os.path.basename(x) for x in files]}, f, indent=2)
    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\n{len(final_features)} features used. Clients written to {args.output}")


if __name__ == "__main__":
    main()
