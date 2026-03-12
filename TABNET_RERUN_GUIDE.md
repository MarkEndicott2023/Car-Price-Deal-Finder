# TabNet Re-Run Guide (for Mark)

hey mark -- the initial tabnet run only used 3 numeric features on 1k rows which is why it got $23k MAE. we need to rerun it with the full feature set so its a fair comparison against the other models (knn is at $1,309 MAE with all features).

## what to change

the main thing is using the same preprocessing pipeline as the other models. here's exactly what to do:

### 1. use the full dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('merged_output.csv')

# drop UCI rows (missing too much)
df = df[df['source'] != 'uci'].drop(columns=['source'])

# handle nulls
df = df.dropna(subset=['price', 'make', 'fuel_type', 'transmission', 'model'])
for col in ['year', 'engine_size', 'mileage']:
    df[col] = df[col].fillna(df[col].median())

# clip price outliers
p01, p99 = df['price'].quantile(0.01), df['price'].quantile(0.99)
df['price'] = df['price'].clip(p01, p99)
```

### 2. feature engineering (this is the big one)

```python
X = df[['make', 'model', 'fuel_type', 'transmission', 'year', 'engine_size', 'mileage']]
y = df['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

preprocessor = ColumnTransformer([
    ('target_enc', TargetEncoder(smooth='auto'), ['make', 'model']),
    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, drop='if_binary'), ['fuel_type', 'transmission']),
    ('scaler', StandardScaler(), ['year', 'engine_size', 'mileage']),
])

preprocessor.fit(X_train, y_train)
X_train_enc = preprocessor.transform(X_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)
```

### 3. train tabnet on the encoded features

```python
from pytorch_tabnet.tab_model import TabNetRegressor

model = TabNetRegressor(
    n_d=32, n_a=32,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
)

model.fit(
    X_train_enc, y_train.values.reshape(-1, 1),
    eval_set=[(X_val_enc, y_val.values.reshape(-1, 1))],
    eval_metric=['mae'],
    max_epochs=200,
    patience=20,
    batch_size=1024,
)
```

### 4. report these metrics

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

preds = model.predict(X_test_enc).flatten()
print(f"MAE:  ${mean_absolute_error(y_test, preds):,.0f}")
print(f"RMSE: ${root_mean_squared_error(y_test, preds):,.0f}")
print(f"R²:   {r2_score(y_test, preds):.4f}")

# classification metrics on price bins (for the paper)
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

bins = [0, 5000, 15000, float('inf')]
labels = ['Low', 'Mid', 'High']
y_true_bins = pd.cut(y_test, bins=bins, labels=labels)
y_pred_bins = pd.cut(pd.Series(preds).clip(0), bins=bins, labels=labels)

print(f"Accuracy:  {accuracy_score(y_true_bins, y_pred_bins):.4f}")
print(f"Precision: {precision_score(y_true_bins, y_pred_bins, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_true_bins, y_pred_bins, average='weighted'):.4f}")
print(f"F1:        {f1_score(y_true_bins, y_pred_bins, average='weighted'):.4f}")
```

## what to write for the paper (section 5.5)

for the methodology section you need to explain:
- tabnet uses sequential attention (cite Arik & Pfister 2021 AAAI) not a simple feedforward net
- the sparse attention masks select relevant features at each decision step
- this is what makes it qualify as a "complex neural network" for the project requirements
- architecture params: n_d, n_a, n_steps, gamma, batch size, lr, patience
- include the training loss / val MAE curves (save them with matplotlib)

## important

- use `random_state=42` for the train/test split so results are reproducible and comparable
- the `merged_output.csv` file -- grab it from google drive (id: 1PugpvRXHmyp9BObilhligI9WTBLqi9Ae) or ask zack
- if you want to compare directly with my results, the json is at `models/helodani/model_comparison_results.json`

lmk if anything is unclear!
