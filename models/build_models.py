"""
Car Price Deal Finder — Model Comparison Pipeline
CSE404 Team Project | Daniel Helo (helodani)

Builds and compares Ridge, SVR, and KNN regressors on the merged car dataset.
Uses target encoding for high-cardinality categoricals (make, model),
one-hot for low-cardinality (fuel_type, transmission).

References:
- Vaneesha et al. (2023): KNN and SVM for used-car pricing (IJCSRR)
- Li & Lin (2021): MLPs vs linear regression for car prices (Stanford CS230)
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import uniform, loguniform, randint

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
print("=" * 60)
print("PHASE 1: Data Loading & Cleaning")
print("=" * 60)

df = pd.read_csv('merged_output.csv')
print(f"Raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Drop UCI rows (only 201 rows, from 1985, missing year/mileage/model/transmission)
df = df[df['source'] != 'uci'].copy()
print(f"After dropping UCI: {df.shape[0]:,} rows")

# Drop source column (no longer needed)
df = df.drop(columns=['source'])

# Handle missing values
null_counts = df.isnull().sum()
print(f"\nNull counts before cleaning:")
for col, cnt in null_counts.items():
    if cnt > 0:
        print(f"  {col}: {cnt:,}")

# Drop rows with null target or null categoricals (can't impute those meaningfully)
df = df.dropna(subset=['price', 'make', 'fuel_type', 'transmission'])

# Impute numeric nulls with median
for col in ['year', 'engine_size', 'mileage']:
    median_val = df[col].median()
    n_null = df[col].isnull().sum()
    if n_null > 0:
        df[col] = df[col].fillna(median_val)
        print(f"  Imputed {n_null:,} nulls in '{col}' with median={median_val:.1f}")

# Drop rows with null model (only ~0 after UCI removal, but be safe)
df = df.dropna(subset=['model'])
print(f"\nAfter null handling: {df.shape[0]:,} rows")

# Clip price outliers — cap at 1st and 99th percentiles
p01 = df['price'].quantile(0.01)
p99 = df['price'].quantile(0.99)
n_clipped = ((df['price'] < p01) | (df['price'] > p99)).sum()
df['price'] = df['price'].clip(p01, p99)
print(f"Price clipped to [{p01:,.0f}, {p99:,.0f}] — {n_clipped:,} rows affected")

# Final stats
print(f"\nCleaned dataset: {df.shape[0]:,} rows")
print(f"Price: mean=${df['price'].mean():,.0f}, median=${df['price'].median():,.0f}, std=${df['price'].std():,.0f}")
print(f"Unique makes: {df['make'].nunique()}, models: {df['model'].nunique()}")
print(f"Fuel types: {df['fuel_type'].nunique()}, transmissions: {df['transmission'].nunique()}")

# ============================================================
# 2. FEATURE ENGINEERING & SPLIT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Feature Engineering & Train/Test Split")
print("=" * 60)

# Define feature groups
high_card_cats = ['make', 'model']       # Target encoding
low_card_cats = ['fuel_type', 'transmission']  # One-hot encoding
numerics = ['year', 'engine_size', 'mileage']

X = df[high_card_cats + low_card_cats + numerics]
y = df['price']

# Split BEFORE encoding (prevents target leakage)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('target_enc', TargetEncoder(smooth='auto'), high_card_cats),
    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, drop='if_binary'), low_card_cats),
    ('num', StandardScaler(), numerics),
], remainder='drop')

print(f"Preprocessing: TargetEncoder({high_card_cats}), OneHot({low_card_cats}), StandardScaler({numerics})")

# Fit preprocessor on training data only
preprocessor.fit(X_train, y_train)
X_train_t = preprocessor.transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

print(f"Transformed feature count: {X_train_t.shape[1]}")

# ============================================================
# 3. MODEL TRAINING & COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Model Training & Comparison")
print("=" * 60)

results = {}

def evaluate_model(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """Train, evaluate on val and test, store results."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    preds_val = model.predict(X_v)
    preds_test = model.predict(X_te)
    pred_time = time.time() - t0

    val_mae = mean_absolute_error(y_v, preds_val)
    val_rmse = root_mean_squared_error(y_v, preds_val)
    val_r2 = r2_score(y_v, preds_val)

    test_mae = mean_absolute_error(y_te, preds_test)
    test_rmse = root_mean_squared_error(y_te, preds_test)
    test_r2 = r2_score(y_te, preds_test)

    results[name] = {
        'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
        'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
        'train_time': train_time, 'pred_time': pred_time,
        'preds_test': preds_test
    }

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Validation:  MAE=${val_mae:,.0f}  |  RMSE=${val_rmse:,.0f}  |  R²={val_r2:.4f}")
    print(f"  Test:        MAE=${test_mae:,.0f}  |  RMSE=${test_rmse:,.0f}  |  R²={test_r2:.4f}")
    print(f"  Train time: {train_time:.1f}s  |  Predict time: {pred_time:.2f}s")

    return model, preds_test

# --- 3a. Ridge Regression (Baseline) ---
print("\n[1/4] Training Ridge Regression (baseline)...")
ridge = Ridge(alpha=1.0)
ridge_model, ridge_preds = evaluate_model(
    "Ridge Regression", ridge, X_train_t, y_train, X_val_t, y_val, X_test_t, y_test
)

# Tune Ridge alpha via CV
print("  Tuning alpha via 5-fold CV...")
ridge_cv = RandomizedSearchCV(
    Ridge(), {'alpha': loguniform(0.01, 1000)},
    n_iter=30, cv=5, scoring='neg_mean_absolute_error',
    random_state=42, n_jobs=1
)
ridge_cv.fit(X_train_t, y_train)
print(f"  Best alpha: {ridge_cv.best_params_['alpha']:.4f}")
ridge_tuned, ridge_tuned_preds = evaluate_model(
    "Ridge (Tuned)", ridge_cv.best_estimator_, X_train_t, y_train, X_val_t, y_val, X_test_t, y_test
)

# --- 3b. KNN Regressor ---
print("\n[2/4] Training KNN Regressor...")
# Quick baseline first
knn_base = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=1)
knn_model, knn_preds = evaluate_model(
    "KNN (k=10, distance)", knn_base, X_train_t, y_train, X_val_t, y_val, X_test_t, y_test
)

# Tune KNN — subsample training for CV (KNN is O(n*d) per query, CV on 190K is brutal)
print("  Tuning k and weights via RandomizedSearchCV (50K subsample)...")
knn_sub_n = min(50000, X_train_t.shape[0])
knn_sub_idx = np.random.choice(X_train_t.shape[0], knn_sub_n, replace=False)
X_train_knn_cv = X_train_t[knn_sub_idx]
y_train_knn_cv = y_train.iloc[knn_sub_idx]

knn_cv = RandomizedSearchCV(
    KNeighborsRegressor(n_jobs=1),
    {
        'n_neighbors': randint(3, 30),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    n_iter=12, cv=3, scoring='neg_mean_absolute_error',
    random_state=42, n_jobs=1
)
knn_cv.fit(X_train_knn_cv, y_train_knn_cv)
print(f"  Best params: {knn_cv.best_params_}")
# Train tuned KNN on 50K subsample (full 190K prediction is fine, just training/CV is slow)
knn_tuned_model = KNeighborsRegressor(**knn_cv.best_params_, n_jobs=1)
knn_tuned, knn_tuned_preds = evaluate_model(
    "KNN (Tuned)", knn_tuned_model, X_train_t, y_train, X_val_t, y_val, X_test_t, y_test
)

# --- 3c. SVR ---
# SVR with RBF kernel is O(n²) in memory — subsample for training
print("\n[3/4] Training SVR...")
print("  Note: SVR scales O(n²), subsampling to 30K for training...")

# Subsample training data for SVR
np.random.seed(42)
svr_n = min(30000, X_train_t.shape[0])
svr_idx = np.random.choice(X_train_t.shape[0], svr_n, replace=False)
X_train_svr = X_train_t[svr_idx]
y_train_svr = y_train.iloc[svr_idx]

# LinearSVR first (scales better)
from sklearn.svm import LinearSVR
linear_svr = LinearSVR(max_iter=5000, C=1.0, epsilon=0.1, random_state=42)
lsvr_model, lsvr_preds = evaluate_model(
    "LinearSVR", linear_svr, X_train_svr, y_train_svr, X_val_t, y_val, X_test_t, y_test
)

# RBF SVR (on subsample)
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=100, epsilon=100, gamma='scale')
svr_model, svr_preds = evaluate_model(
    "SVR (RBF, 30K sample)", svr_rbf, X_train_svr, y_train_svr, X_val_t, y_val, X_test_t, y_test
)

# --- 3d. Yousif's TabNet comparison point ---
print("\n[4/4] TabNet reference (from Yousif's notebook):")
print(f"  Test MAE: $23,386 (3 features, 1K rows, no categoricals)")
print(f"  Note: Not directly comparable — used different features and data subset")

# ============================================================
# 4. COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Model Comparison Summary")
print("=" * 60)

print(f"\n{'Model':<25} {'Test MAE':>12} {'Test RMSE':>12} {'Test R²':>10} {'Train(s)':>10}")
print("─" * 72)
for name, r in results.items():
    print(f"{name:<25} ${r['test_mae']:>10,.0f} ${r['test_rmse']:>10,.0f} {r['test_r2']:>9.4f} {r['train_time']:>9.1f}")

print(f"\n{'TabNet (Yousif baseline)':<25} ${'23,386':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

# ============================================================
# 5. DEAL SCORING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Deal Scoring Module")
print("=" * 60)

# Use best model's predictions
best_name = min(results, key=lambda k: results[k]['test_mae'])
best_preds = results[best_name]['preds_test']
print(f"Using best model: {best_name}")

# Compute deal scores on test set
test_df = X_test.copy()
test_df['actual_price'] = y_test.values
test_df['predicted_price'] = best_preds
test_df['residual'] = test_df['predicted_price'] - test_df['actual_price']
test_df['deal_score'] = test_df['residual'] / test_df['predicted_price']

# Flag deals at various thresholds
for threshold in [0.15, 0.20, 0.25, 0.30]:
    deals = test_df[test_df['deal_score'] > threshold]
    n_deals = len(deals)
    pct = n_deals / len(test_df) * 100
    avg_savings = deals['residual'].mean() if n_deals > 0 else 0
    print(f"  Threshold {threshold:.0%}: {n_deals:,} deals ({pct:.1f}%), avg savings=${avg_savings:,.0f}")

# Show top 10 deals
print(f"\nTop 10 best deals (highest deal score):")
top_deals = test_df.nlargest(10, 'deal_score')[['make', 'model', 'year', 'actual_price', 'predicted_price', 'deal_score']]
print(top_deals.to_string(index=False))

# Save deal-scored test set
test_df.to_csv('deal_scored_test_set.csv', index=False)
print(f"\nDeal-scored test set saved to deal_scored_test_set.csv")

# ============================================================
# 6. SAVE RESULTS
# ============================================================
summary = {}
for name, r in results.items():
    summary[name] = {k: v for k, v in r.items() if k != 'preds_test'}
    # Convert numpy types for JSON serialization
    for k, v in summary[name].items():
        if isinstance(v, (np.floating, np.integer)):
            summary[name][k] = float(v)

with open('model_comparison_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to model_comparison_results.json")

print("\n" + "=" * 60)
print("DONE — All models trained and evaluated.")
print("=" * 60)
