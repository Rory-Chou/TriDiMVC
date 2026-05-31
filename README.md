# TriD-MVC Paper Reproduction Code

Minimal runnable code for TriD-MVC (v5): training and evaluation only.

## Layout

```
paper_repo/
├── train_v5.py           # Main training entry
├── data.py               # Dataset loading
├── models_v5.py          # Model
├── losses_v5.py          # Loss functions
├── clustering.py         # K-means evaluation
├── uot_semantic_fusion.py  # UOT semantic fusion
├── alignment.py          # Hungarian / max-similarity alignment
├── utils.py              # Utilities
├── requirements.txt
└── datasets/             # Place *.mat datasets here
```

## Environment

```bash
cd paper_repo
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch, NumPy, SciPy, scikit-learn.

## Data

Put `<dataset_name>.mat` under `datasets/`; see `datasets/README.md`.

## Run

```bash
# Aligned multi-view data
python train_v5.py -data BBCsports -eps 200 --seed 42

# Partial misalignment / missing views
python train_v5.py -data BBCsports -mar 0.5 -msr 0.5 -eps 500 --seed 42
```

Common flags: `-data`, `-eps`, `-mar` (misalignment rate), `-msr` (missing rate), `-cui` (cluster update interval), `-wco` / `-wcl` (loss weights), `--logs_root`, `--save_best_features`, `--seed`.

Logs are written to `logs/<dataset>/` by default.
