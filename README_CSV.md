# FLAD reproduction on the CIC-DDoS2019 CSV files (ISDI 509)

This branch (`csv-reproduction`) starts from the authors' `flad-paper-evaluation` branch,
which contains FLAD and the FedAvg and FLDDoS baselines used in the paper.

**The authors' files are not modified** (`flad_main.py`, `flad_training.py`,
`ann_models.py`, `util_functions.py`). We only added:

| File | What it does |
|---|---|
| `csv_to_clients.py` | Builds the 13 clients (one attack each + benign) from the CSV/Parquet files and saves them in the authors' hdf5 format |
| `evaluate_test.py` | Tests a trained model on each client's test set: TPR (paper, Table 6) plus FPR, precision, F1 and MCC |
| `run_csv_experiments.py` | Runs FLAD, then FedAvg E=1, FedAvg E=5 and FLDDoS for the same number of rounds, for several seeds, and writes Tables 5 and 6 |
| `sample_clients/` | The authors' 4 sample clients (packet format), copied from their `main` branch, for the smoke test |

## Differences from the paper (to report in Task 2 / Task 3)

1. **Features:** about 80 CICFlowMeter flow statistics (one row per flow) instead of
   the 10 packets x 11 features built by the LUCID parser from the raw pcaps.
2. **Client sizes:** the Table 3 sizes (doubling from client to client), capped with
   `--max_samples` (default 40,000 per client), because the CSV files contain little
   benign traffic and Kaggle has limited memory and time.
3. **Identifier columns removed** (flow ID, IPs, ports, timestamp), so the model cannot
   learn the capture instead of the attack. `--keep_identifiers` puts them back for the P4 test.
4. **Scaling:** min-max fitted on the training rows only, then clipped to [0, 1]
   (the paper used fixed value ranges for its packet features).
5. **Benign rows are not shared** between clients, and train/val/test are split before scaling.
6. **Seeds:** 3 seeds by default (the paper used 10).

The FL algorithms, the MLP (2 hidden layers of 32 units), PATIENCE = 25, the epochs and
steps ranges, FedAvg (B = 50, F = 0.8) and FLDDoS (E = 10, B = 100, gamma = 0.9 for
WebDDoS and Syn) are unchanged.

## How to run on Kaggle

1. Create a new Kaggle Notebook. *Add Input* -> search `cicddos2019` (user `dhoogla`) -> Add.
2. Settings: Internet **on** (needed for `git clone` and `pip`).
3. Run these cells:

```
!git clone -b csv-reproduction <YOUR-GITHUB-REPO-URL> flad
%cd flad
!pip install -q h5py scikit-learn pyarrow
!python -c "import tensorflow as tf; print(tf.__version__)"
```

Kaggle's own TensorFlow is used. The authors tested TF 2.7.1, so if a newer version raises
an error in the authors' code, note it and fix only that line (record it as a deviation).

```
# 1) Smoke test with the authors' own sample data (packet format)
!python flad_main.py -c ./sample_clients -t flad -o log/sample_flad/
```

```
# 2) Build the 13 clients from the CSV files
!python csv_to_clients.py --input /kaggle/input/cicddos2019 --output Dataset/CSV-DDoS2019 --max_samples 40000 --seed 0
```

```
# 3) Main comparison (Tables 5 and 6), 3 seeds
!python run_csv_experiments.py --clients Dataset/CSV-DDoS2019 --seeds 0 1 2 --out results/main
```

Results are written to `results/main/`:
`table5_mean_std.csv`, `table6_TPR_mean.csv`, and per-run folders with the authors' logs.

If the Kaggle input path is different, check it with `!ls /kaggle/input/`.
