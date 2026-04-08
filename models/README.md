# Models

Model training scripts and output artifacts.

## Files

- `build_models.py` — Full pipeline as a Python script (Ridge, KNN, SVR). Reads from `../data/merged_output.csv`.
- `model_comparison_results.json` — All metrics dumped to JSON so you can load them without rerunning.
- `deal_scored_test_set.csv` — Test set with deal scores attached, sorted by score descending.

## Quick Results

| Model | MAE | R² |
|-------|-----|-----|
| KNN (k=10, distance weighted) | $1,309 | 0.971 |
| Ridge (alpha=0.02) | $3,876 | 0.822 |
| LinearSVR (30k sample) | $4,271 | 0.784 |
| SVR RBF (30k sample) | $5,206 | 0.650 |

## How to Run

```bash
pip install pandas scikit-learn matplotlib

python build_models.py
```

See `../notebooks/` for interactive notebook versions.
