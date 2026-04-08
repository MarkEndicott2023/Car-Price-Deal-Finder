# Model Comparison & Deal Scoring (helodani)

## what's in here

i ran ridge, knn, linesvr, and rbf svr on the full merged dataset (276k rows after cleaning) with proper feature engineering. the big finding is that target encoding make/model is basically the whole game -- went from tabnet's initial $23k MAE down to $1,309 with knn just by encoding categoricals properly.

## quick results

| model | MAE | R² |
|-------|-----|-----|
| KNN (k=10, distance weighted) | $1,309 | 0.971 |
| Ridge (alpha=0.02) | $3,876 | 0.822 |
| LinearSVR (30k sample) | $4,271 | 0.784 |
| SVR RBF (30k sample) | $5,206 | 0.650 |

knn crushes everything else. makes sense -- car pricing is super clustered (a 2015 civic with 60k miles has a pretty narrow price range and knn picks that up naturally).

deal scoring at 20% threshold flags 8.4% of listings as deals with avg savings of $2,309.

## files

- `helodani_model_comparison.ipynb` -- full notebook, runs end to end. has all the preprocessing, model training, evaluation, and deal scoring
- `build_models.py` -- same pipeline as the notebook but as a python script (easier to run headless)
- `model_comparison_results.json` -- all metrics dumped to json so you can load them without rerunning
- `deal_scored_test_set.csv` -- test set with deal scores attached, sorted by score descending

## how to run

```bash
# need these
pip install pandas scikit-learn matplotlib jupyter

# grab the dataset (google drive)
# file id: 1PugpvRXHmyp9BObilhligI9WTBLqi9Ae
# or just use the merged_output.csv if you already have it

# run the notebook
jupyter notebook helodani_model_comparison.ipynb

# or run the script directly
python build_models.py
```

takes about 2-3 min on a decent laptop. knn training is the slow part since its fitting on 190k rows.

## preprocessing pipeline

1. drop UCI rows (only 201, missing year/mileage/model/transmission)
2. clip prices at 1st/99th percentile ($699 - $89,990) to kill outliers
3. target encode make (86 unique) and model (887 unique) -- this is the key step, maps each category to its smoothed mean price
4. one-hot encode fuel_type (13 unique) and transmission (3 unique)
5. z-score standardize year, engine_size, mileage
6. 70/15/15 train/val/test split, encoding fit on train only to prevent leakage

## classification metrics (for the paper)

prof wants precision/recall/accuracy/f1 so we binned prices into Low (<$5k), Mid ($5k-$15k), High (>$15k) and computed weighted metrics on those bins. knn gets 0.929 across the board.

## references

- Vaneesha et al. (2024) "Comparative Analysis of ML Algorithms for Used Car Price Prediction" IJCSRR 7(9)
- Li & Lin (2018) "Predicting Used Car Prices with Deep Learning" Stanford CS230
- Micci-Barreca (2001) target encoding paper, ACM SIGKDD
