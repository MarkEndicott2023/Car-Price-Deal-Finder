# Car Price Deal Finder: A Two-Stage Machine Learning Pipeline for Detecting Undervalued Used-Car Listings

**Authors:** Mark Endicott, Zachary Elakel, Yousif Elya, Daniel Helo, Nishit Nagpal

---

## Abstract

We develop a two-stage machine learning pipeline to detect undervalued used-car listings in a noisy, nonstationary market. The system unifies vehicle data from multiple repositories into a standardized schema to improve price transparency. In the regression stage, we compare Support Vector Machines, K-Nearest Neighbors, Ridge Regression, and TabNet — an attentive transformer-based architecture — for fair market value estimation. KNN (k=10, distance-weighted) achieves the strongest performance with an MAE of $1,309 and R² = 0.971 on the held-out test set, substantially outperforming Ridge (MAE = $3,876, R² = 0.822) and both SVR variants. TabNet shows promising MAE ($2,168) but suffers from training instability on high-value outliers, requiring further hyperparameter refinement. In the second stage, regression residuals are converted into a normalized deal score for binary classification: at a 20% threshold, the KNN-based system flags 8.4% of listings as underpriced with average predicted savings of $2,309.

> **TODO (long paper):** Update abstract with final results after retraining on full dataset.

---

## 1. Introduction and Motivation

The used-car market is opaque: limited pricing transparency creates significant information gaps for consumers, and undervalued listings disappear quickly. Traditional research treats car pricing as a static regression task, yet real-world markets are noisy and nonstationary. We address this gap with a two-stage pipeline that first estimates fair market value via regression, then converts residuals into a deal score to flag underpriced listings in real-time. By unifying disparate data sources into a standardized schema, the system bridges information gaps for both buyers and sellers.

> **TODO (long paper):** Add paragraph summarizing final contributions and results.

---

## 2. Related Work

Vaneesha et al. (2024) compared KNN and SVM regressors for car price prediction, finding KNN slightly superior. Li and Lin (2021) evaluated multilayer perceptrons and found that deeper architectures (more layers, fewer neurons per layer) outperform linear regression and shallow MLPs, providing a neural-network benchmark against which we compare TabNet. Our work extends both by incorporating an attention-based neural architecture (TabNet) and adding a second-stage deal-classification mechanism absent from prior car-pricing studies.

> **TODO (long paper):** Add 2–3 more citations (+0.5 page).

---

## 3. Dataset Description

We built our training corpus by merging three publicly available vehicle-pricing datasets, chosen to vary in size, geographic origin, and feature coverage.

**Hugging Face Car Price Dataset (Gupta, 2024):** 10,000 listings across five manufacturers (Audi, BMW, Ford, Honda, Toyota) with eight attributes including make, model, year, mileage, and price. Model names are anonymized (Model A–E).

**UCI Automobile Dataset (Schlimmer, 1985):** 205 entries across 22 manufacturers and 26 attributes from the 1985 Ward's Automotive Yearbook. It lacks year, mileage, and model fields, so these are null in the merged set, but it adds manufacturer diversity (e.g., Jaguar, Porsche, Saab).

**DVM-CAR Advertisement Table (Huang et al., 2022):** 268,255 UK used-car advertisements spanning 88 manufacturers and 896 models (1900–2019), recording make, model, year, mileage, engine size, gearbox type, fuel type, and price.

### 3.1 Ingestion Pipeline

A Python script normalizes column names, standardizes string casing, coerces types, and concatenates the three sources into a nine-column unified schema (make, model, year, engine_size, mileage, fuel_type, transmission, price, source). Rows with missing or non-positive prices were removed, yielding 277,311 listings before outlier clipping. DVM-CAR dominates at 267,110 rows (96.3%); Hugging Face contributes 10,000; UCI adds 201.

> **TODO (long paper):** Add EDA plots / feature distributions (+0.5–1 page).

---

## 4. Methodology

### 4.1 Preprocessing

After merging, the 201 UCI rows were removed due to missing year, mileage, and model fields, yielding 276,627 listings. Prices were clipped at the 1st and 99th percentiles ($699–$89,990). High-cardinality categoricals (make: 86 unique; model: 887 unique) were target-encoded (Micci-Barreca, 2001); low-cardinality categoricals (fuel_type, transmission) were one-hot encoded. Numerical features (year, engine_size, mileage) were z-score standardized. All encoding was fit on training data only. The dataset was split 70/15/15 into train (193,638), validation (41,494), and test (41,495).

### 4.2 Models

**Ridge Regression** (Hoerl and Kennard, 1970) serves as our linear baseline, with α tuned via 5-fold cross-validation (optimal α = 0.02).

**K-Nearest Neighbors** predicts price as the distance-weighted mean of k nearest neighbors. We tuned k over [3, 30] with 3-fold CV on a 50K subsample (Figure 2).

**Support Vector Regression** was evaluated in linear (LinearSVR) and RBF-kernel variants, both trained on a 30K subsample due to O(n²) memory scaling.

**TabNet** (Arik and Pfister, 2021) is an attention-based neural network that uses sequential sparse attention masks to select features at each decision step. Hyperparameters: max_epochs = 40, patience = 15, batch_size = 128.

> **TODO (long paper):** Expand architecture, hyperparameter, and training details (+1 page).

### 4.3 Deal Scoring

We convert regression output to a deal score:

> deal_score = (ŷ − y) / ŷ     (1)

where ŷ is the predicted price and y is the listing price. Listings with deal_score > τ are classified as deals.

---

## 5. Results

### 5.1 Regression Performance

**Table 1: Regression performance on the held-out test set (n = 41,495). SVR models trained on a 30K subsample.**

| Model        | MAE ($) | RMSE ($) | R²    |
|--------------|---------|----------|-------|
| KNN (k=10)   | 1,309   | 2,512    | 0.971 |
| Ridge        | 3,876   | 6,180    | 0.822 |
| LinearSVR    | 4,271   | 6,821    | 0.784 |
| SVR (RBF)    | 5,206   | 8,679    | 0.650 |
| TabNet       | 2,168   | 49,200   | 0.176 |

[Figure 1: Model comparison on MAE (left) and R² (right) across all architectures.]

Table 1 summarizes test-set regression metrics. KNN achieves the lowest MAE of $1,309 (13.5% of median price $9,699), explaining 97.1% of variance. Ridge captures only 82.2%, confirming strong nonlinear structure in car pricing. SVR underperforms due to its 30K training subsample. TabNet shows a competitive MAE of $2,168 but exhibits training instability (Figure 6), with an R² of only 0.176 due to extreme errors on high-value outliers.

### 5.2 Classification Metrics

To satisfy classification evaluation requirements, we discretized prices into Low (<$5K), Mid ($5K–$15K), and High (>$15K) bins. Table 2 reports weighted-average classification metrics.

**Table 2: Classification metrics (price-bin, weighted avg.).**

| Model        | Acc.  | Prec. | Rec.  | F1    |
|--------------|-------|-------|-------|-------|
| KNN (k=10)   | 0.929 | 0.929 | 0.929 | 0.929 |
| Ridge        | 0.821 | 0.825 | 0.821 | 0.820 |
| LinearSVR    | 0.746 | 0.773 | 0.746 | 0.739 |
| SVR (RBF)    | 0.692 | 0.706 | 0.692 | 0.687 |

> **TODO (long paper):** Add TabNet row once retrained.

### 5.3 Error Analysis

Figure 3 shows KNN predictions closely track the identity line. Figure 4 reveals residuals are centered near zero, with error magnitude increasing for higher-priced vehicles — expected given greater price variation in the luxury segment.

[Figure 2: KNN validation MAE as a function of k. Optimal k=10 balances bias and variance.]
[Figure 3: KNN actual vs. predicted price (5K sample). Points cluster tightly along the identity line.]
[Figure 4: KNN residual distribution (left) and residuals vs. predicted price (right).]

> **TODO (long paper):** Add 3–5 concrete failure cases with speculation on causes (+0.5 page).

### 5.4 Deal Scoring

At τ = 0.20, KNN flags 8.4% of test listings (3,493 / 41,495) as deals with average savings of $2,309 (Figure 5). The inverse relationship between threshold and flagged volume suggests well-calibrated residuals.

[Figure 5: Deal-score distribution (KNN, k=10) with 20% and 30% threshold markers.]

### 5.5 TabNet Training Dynamics

Figure 6 contrasts TabNet's initial 3-feature run with the full-data run (273K rows, label-encoded categoricals): training loss decreases steadily from 2.9 × 10⁸ to 7.5 × 10⁷ over 28 epochs before early stopping (best epoch 12, val MAE = $1,998). Despite convergence in loss, the high RMSE ($49,200) and low R² (0.176) indicate that a small number of extreme residuals on high-value vehicles dominate squared-error metrics, even as the median prediction quality is reasonable (MAE = $2,168).

[Figure 6: TabNet MAE loss per epoch on the full dataset (273K rows). Early stopping at epoch 27, best epoch 12.]

---

## 6. Discussion

Feature engineering dominated model choice: adding target-encoded make/model reduced MAE 18× compared to TabNet's initial 3-feature run. KNN's strength is domain-specific, as car pricing is naturally clustered by make, model, year, and mileage.

> **TODO (long paper):** Expand discussion of trade-offs (inference cost, generalization, calibration).

---

## 7. Conclusion and Future Work

We have unified over 276,000 listings into a standardized schema and demonstrated a two-stage pipeline where KNN achieves an R² of 0.971 and flags underpriced listings with average savings of $2,309. Car pricing exhibits strong nonlinear structure that simple linear models struggle to capture.

For the final report, we plan to:

- **Optimize TabNet** on the full target-encoded feature set to determine if its attention mechanism can match or surpass KNN.
- **Calibrate deal-score thresholds** via precision–recall analysis to ensure flagged deals are genuinely underpriced rather than noisy labels.
- **Analyze high-value outliers** to determine if separate models for luxury vs. budget segments reduce heteroscedastic error.
- **Retrain all models** on the full 276,627-row dataset for a final head-to-head comparison.
- **Build an interface prototype** for real-time fair market value and deal score queries.
- **Test regional features** (geographic and condition data) to reduce sensitivity to extreme errors on high-value vehicles.

> **TODO (long paper):** Expand each future work bullet to 1–2 sentences with "why it matters."

---

## References

- Sercan Ö. Arik and Tomas Pfister. *TabNet: Attentive interpretable tabular learning.* Proceedings of the AAAI Conference on Artificial Intelligence, 35(8):6679–6687, 2021.
- Varun Kumar Gupta. *Car price dataset.* Hugging Face Datasets, 2024.
- Arthur E. Hoerl and Robert W. Kennard. *Ridge regression: Biased estimation for nonorthogonal problems.* Technometrics, 12(1):55–67, 1970.
- Jianhan Huang, Bowen Chen, Linxiao Luo, Shiqiang Yue, and Iadh Ounis. *DVM-CAR: A large-scale automotive dataset for visual marketing research and applications.* In Proceedings of the 2022 IEEE International Conference on Big Data, pages 4140–4147, 2022.
- T. Li and J. Lin. *Used car price prediction using machine learning.* Technical report, Stanford University, CS230, 2021.
- Daniele Micci-Barreca. *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems.* ACM SIGKDD Explorations Newsletter, 3(1):27–32, 2001.
- Jeffrey C. Schlimmer. *Automobile dataset.* UCI Machine Learning Repository, 1985.
- B. Vaneesha, P. S. Reddy, A. Kumar, and V. Rao. *Predicting car prices using machine learning: A comparative study of KNN and SVM regressors.* International Journal of Current Science Research and Review, 7(9), 2024.
