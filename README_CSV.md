# FLAD reproduction on the CIC-DDoS2019 CSV files (ISDI 509)

This repository starts from the authors' `flad-paper-evaluation` branch
(https://github.com/doriguzzi/flad-federated-learning-ddos), which contains FLAD and the
FedAvg and FLDDoS baselines used in the paper.

## What we added or changed

| File | What it does |
|---|---|
| `csv_to_clients.py` | Builds the 13 clients (one attack each + benign) from the official CIC-DDoS2019 CSV files and saves them in the authors' hdf5 format. Reads the large CSV files in pieces. |
| `evaluate_test.py` | Tests a trained model on each client's test set: TPR (paper, Table 6) plus FPR, precision, F1 and MCC |
| `run_csv_experiments.py` | Runs FLAD, then FedAvg E=1, FedAvg E=5 and FLDDoS for the same number of rounds, for several seeds, and writes Tables 5 and 6 |
| `requirements.txt` | Python 3.9 + TensorFlow 2.7.1, the versions used by the authors |
| `sample_clients/` | The authors' 4 sample clients (packet format), copied from their `main` branch, for the smoke test |
| `util_functions.py`, `ann_models.py` | **One line changed in each**, so that file and folder names are read correctly on Windows (`os.path.basename` instead of splitting on `/`). Nothing else in the authors' code is changed. |

## Differences from the paper (to report in Task 2 / Task 3)

1. **Features:** about 80 CICFlowMeter flow statistics (one row per flow, from the official
   CSV files) instead of the 10 packets x 11 features built by the LUCID parser from the raw pcaps.
2. **Client sizes:** the Table 3 sizes (doubling from client to client), capped with
   `--max_samples` (default 40,000 per client). The CSV files contain little benign traffic;
   if there is not enough, the script lowers the cap automatically and prints it.
3. **Both days pooled:** for attacks recorded on both days (LDAP, MSSQL, NetBIOS, UDP,
   UDPLag, Syn) rows are drawn from both CSV folders; the paper does not say which day it used.
4. **Identifier columns removed** (flow ID, IPs, ports, timestamp, Inbound, SimillarHTTP),
   so the model cannot learn the capture instead of the attack. `--keep_identifiers` puts
   them back for the P4 pitfall test.
5. **Scaling:** min-max fitted on the training rows only, then clipped to [0, 1]
   (the paper used fixed value ranges for its packet features). Infinite/missing values set to 0.
6. **Benign rows are not shared** between clients; each client is split 81/9/10
   (train/val/test) with the same 50/50 ratio in every split.
7. **Seeds:** 3 seeds by default (the paper used 10).

Unchanged: the FL algorithms, the MLP (2 hidden layers of 32 units), PATIENCE = 25,
the epochs (1-5) and steps (10-1000) ranges, FedAvg (B = 50, F = 0.8) and FLDDoS
(E = 10, B = 100, gamma = 0.9 for WebDDoS and Syn).

## How to run (Windows, PowerShell, from this folder)

### 1. Environment (once)

```
py install 3.9
py -V:3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Data

Download `CSV-01-12.zip` and `CSV-03-11.zip` from https://www.unb.ca/cic/datasets/ddos-2019.html
and unzip both into one folder, for example `D:\CIC-DDoS2019\`
(the script also searches subfolders).

### 3. Smoke test with the authors' own sample data (packet format)

```
.\.venv\Scripts\python.exe flad_main.py -c ./sample_clients -t flad -o log/sample_flad/
```

### 4. Build the 13 clients from the CSV files

```
.\.venv\Scripts\python.exe csv_to_clients.py --input "D:\CIC-DDoS2019" --output Dataset/CSV-DDoS2019 --max_samples 40000 --seed 0
```

Check `Dataset/CSV-DDoS2019/clients_summary.csv` (sizes) and `features.json` (features used).

### 5. Main comparison (Tables 5 and 6), 3 seeds

```
.\.venv\Scripts\python.exe run_csv_experiments.py --clients Dataset/CSV-DDoS2019 --seeds 0 1 2 --out results/main
```

Results in `results/main/`: `table5_mean_std.csv`, `table6_TPR_mean.csv`, and one folder
per run with the authors' logs, the trained model and `test_metrics.csv`.

## Dataset citation

I. Sharafaldin, A. H. Lashkari, S. Hakak and A. A. Ghorbani, "Developing Realistic Distributed
Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE ICCST, 2019.
