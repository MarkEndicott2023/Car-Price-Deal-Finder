# TabNet Neural Network Results — Interpretation

## Setup

A TabNet regressor was trained to predict used car prices using only 3 numeric features (`year`, `engine_size`, `mileage`) on a 1,000-row subsample of the 273K-row dataset. Categorical features (`make`, `model`, `fuel_type`, `transmission`) were dropped. Data was split 70/15/15 (train/validation/test).

## Results

- **Test MAE: ~$25,000** — predictions are off by roughly the median car price, indicating the model has no real predictive power.
- The model early-stopped at epoch 10, reverting to epoch 0 weights. Training loss stayed flat and validation MAE fluctuated wildly (27K–101K), showing the model never converged.

## Why It Failed

1. **Too little data**: Only 1,000 of 273K rows used — TabNet needs far more data to learn effectively.
2. **Key features excluded**: Dropping `make`, `model`, `fuel_type`, and `transmission` removes the strongest price predictors.
3. **No feature normalization**: Features on vastly different scales (year ~2000, mileage ~100K, engine_size ~1–5) likely caused training instability.
4. **Overparameterized**: `n_d=32`, `n_a=32`, `n_steps=5` is too complex for just 3 features and 700 training samples.

## Recommendations

- Use the full dataset (273K rows).
- Include categorical features via TabNet's native `cat_idxs`/`cat_dims` support.
- Normalize numeric features before training.
- Reduce model complexity or tune the learning rate.

