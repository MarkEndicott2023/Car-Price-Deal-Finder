# Car Price Deal Finder: A Two-Stage Machine Learning Pipeline for Detecting Undervalued Used-Car Listings

**Authors:** Mark Endicott, Zachary Elakel, Yousif Elya, Daniel Helo, Nishit Nagpal

---

## Abstract

> TODO (long paper): Update abstract with final results after retraining on full dataset.

We develop a two-stage machine learning pipeline to detect undervalued used-car listings in a noisy, fragmented market. The system unifies vehicle data from multiple repositories into a standardized schema to improve price transparency. In the regression stage, we compare Support Vector Machines, K-Nearest Neighbors, Ridge Regression, and TabNet (an attentive transformer-based architecture) for fair market value estimation. Our evaluation shows that instance-based learning and optimized neural architectures significantly outperform linear baselines. KNN (k=10, distance-weighted) was the most accurate model, with a test Mean Absolute Error (MAE) of $1,309 and an R² of 0.971. Following optimization via log-target transformation and target encoding, TabNet reached an MAE of $1,921 and an R² of 0.943, capturing the nonlinearities of the used-car market. Both substantially outperformed the Ridge Regression baseline (MAE = $3,876, R² = 0.822) and both SVR variants, which struggled with scalability and high-cardinality features. In the second stage, regression residuals are normalized into a deal score and thresholded to flag underpriced listings: at a 20% cutoff, the KNN-based system flags 8.4% of listings as deals with average predicted savings of $2,309. This two-stage approach provides a framework for identifying high-value opportunities in opaque secondary markets.

---

## 1. Introduction and Motivation

The used-car market is opaque: limited pricing transparency creates significant information gaps for consumers, and undervalued listings disappear quickly. Traditional research treats car pricing as a static regression task, yet real-world markets are noisy and heterogeneous. We address this gap with a two-stage pipeline that first estimates fair market value via regression, then converts residuals into a deal score to flag underpriced listings as they enter the market. By unifying disparate data sources into a standardized schema, the system bridges information gaps for both buyers and sellers.

Our results show that instance-based learning and optimized neural architectures outperform linear baselines. By benchmarking multiple architectural families, we find that a vehicle's neighborhood in feature space is the strongest predictor of its fair market value. The K-Nearest Neighbors model was the most accurate regressor, capturing nonlinear depreciation curves that global linear approximations miss. We then convert its residuals into a calibrated deal score, yielding a framework for identifying underpriced listings in fragmented secondary markets.

---

## 2. Related Work

Vaneesha et al. (2024) compared KNN and SVM regressors for car price prediction, finding that KNN generally performs better on vehicle datasets. Similarly, Li and Lin (2021) evaluated multilayer perceptrons and found that deeper architectures (more layers, fewer neurons per layer) outperform linear regression and shallow MLPs, providing a neural-network benchmark against which we compare TabNet. Our work extends both by incorporating an attention-based neural architecture (TabNet) and adding a second-stage deal-classification mechanism absent from prior car-pricing studies.

As used-car data involves hundreds of unique models, feature representation is a critical bottleneck. Zhu (2023) emphasizes the importance of feature representation and encoding in improving prediction accuracy, which aligns closely with our use of target encoding and structured preprocessing. Micci-Barreca (2001) provided the foundation for target encoding high-cardinality features. We adopt these principles to handle our 887 unique car models, allowing the model to learn from specific manufacturers and models without the dimensionality explosion caused by traditional one-hot encoding.

A previous effort by Gegic et al. (2019) used ensembles to overcome the poor performance of single classifiers, made easier by binning prices into broad categories. In contrast, our pipeline maintains high-precision regression across the entire continuous market spectrum. One way we achieve this is by using TabNet, an attentive transformer architecture introduced by Arik and Pfister (2021). The model uses sequential attention masks to select the most relevant features at each decision step, combining the interpretability of tree-based models with the representational power of deep learning over heterogeneous tabular inputs.

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

**Ridge Regression:** (Hoerl and Kennard, 1970) minimizes the penalized objective L(w) = ||Xw − y||² + α||w||², trading variance for bias reduction via the L2 coefficient penalty. We tuned α via RandomizedSearchCV over a log-uniform prior on [0.01, 1000], 30 iterations, 5-fold cross-validation, finding optimal α = 0.0195. Ridge serves as the interpretable linear baseline: any nonlinear model that cannot outperform it substantially does not justify the added complexity.

**K-Nearest Neighbors:** predicts price as the distance-weighted mean of k neighbors (Dudani, 1976): ŷ = Σᵢ wᵢ yᵢ / Σᵢ wᵢ, wᵢ = 1 / d(x, xᵢ). Hyperparameters were tuned via RandomizedSearchCV (12 iterations, 3-fold CV on a 50K subsample) over k ∈ [3, 30], weights ∈ {uniform, distance}, and metric ∈ {Euclidean, Manhattan}, yielding k = 10 with distance weighting and Euclidean distance. KNN's inductive bias (similar cars should have similar prices) maps directly onto the used-car domain (Vaneesha et al., 2024).

**Support Vector Regression:** minimizes the ε-insensitive loss (Smola and Schölkopf, 2004): min ½||w||² + C Σ (ξᵢ + ξᵢ*) s.t. |yᵢ − ŷᵢ| ≤ ε + ξᵢ(*). Predictions within ε of the true price incur no penalty; C controls tolerance for larger deviations. We evaluated LinearSVR (C = 1, ε = 0.1), which scales linearly, and RBF-kernel SVR (C = 100, ε = 100, γ = scale), which captures nonlinear depreciation curves. Both were trained on a 30K subsample due to O(n²) memory scaling.

**TabNet:** (Arik and Pfister, 2021) neural network designed for tabular data that learns which features matter for each individual prediction. It processes inputs through a sequence of decision steps (N_steps = 5). At each step, a sparse attention mask selects a small subset of features to focus on, passes them through a fully connected layer, and adds the result to a running total. The sparsity coefficient γ = 1.5 encourages different steps to attend to different features rather than reusing the same ones. Decision and attention layer widths were both set to 32 (n_d = n_a = 32). Training used the Adam optimizer with max_epochs = 40, patience = 15, and batch_size = 128. We evaluated TabNet under two preprocessing configurations: an initial run with label-encoded categoricals, and an optimized run using the same target-encoding pipeline as the other models with a log1p-transformed target (see §5.5).

### 4.3 Deal Scoring

We convert regression output to a deal score:

> deal_score = (ŷ − y) / ŷ     (1)

where ŷ is the predicted price and y is the listing price. Listings with deal_score > τ are classified as deals.

---

## 5. Results

### 5.1 Regression Performance

**Table 1: Regression performance on the held-out test set (n = 41,495). SVR models trained on a 30K subsample. TabNet (optimized) applies target encoding and log-target transform.**

| Model              | MAE ($) | RMSE ($) | R²    |
|--------------------|---------|----------|-------|
| KNN (k=10)         | 1,309   | 2,512    | 0.971 |
| TabNet (optimized) | 1,921   | 3,501    | 0.943 |
| TabNet (initial)   | 2,168   | 49,200   | 0.176 |
| Ridge              | 3,876   | 6,180    | 0.822 |
| LinearSVR          | 4,271   | 6,821    | 0.784 |
| SVR (RBF)          | 5,206   | 8,679    | 0.650 |

[Figure 1: Model comparison on MAE (left) and R² (right) across all architectures.]

Table 1 summarizes test-set regression metrics. KNN achieves the lowest MAE of $1,309 (13.5% of median price $9,699), explaining 97.1% of variance. Ridge captures only 82.2%, confirming strong nonlinear structure in car pricing. SVR underperforms due to its 30K training subsample. TabNet's initial run, which used label-encoded categoricals on the full dataset, produced a competitive MAE of $2,168 but an R² of only 0.176, as a small number of extreme residuals on high-value vehicles dominated squared-error metrics. After applying the same target-encoding pipeline used by the other models and training on a log-transformed target to compress the luxury tail, TabNet improved across the board: MAE fell from $2,168 to $1,921, RMSE from $49,200 to $3,501, and R² rose from 0.176 to 0.943. This confirms that the initial gap was driven by preprocessing rather than architectural limitations, and places TabNet as the second-strongest model behind KNN.

### 5.2 Classification Metrics

To complement the regression metrics with a classification view, we discretize prices into Low (<$5K), Mid ($5K–$15K), and High (>$15K) tiers. Each model's predicted price is mapped to a bin and scored against the bin of the true price. This measures whether each regressor places listings in the correct price tier, independent of exact dollar error. Table 2 reports weighted-average classification metrics across the three bins.

**Table 2: Classification metrics (price-bin, weighted avg.).**

| Model        | Acc.  | Prec. | Rec.  | F1    |
|--------------|-------|-------|-------|-------|
| KNN (k=10)   | 0.929 | 0.929 | 0.929 | 0.929 |
| Ridge        | 0.821 | 0.825 | 0.821 | 0.820 |
| LinearSVR    | 0.746 | 0.773 | 0.746 | 0.739 |
| SVR (RBF)    | 0.692 | 0.706 | 0.692 | 0.687 |

TabNet is omitted from Table 2; its optimized-run test predictions were not mapped to price bins in the original experiment.

### 5.3 Error Analysis

Figure 3 shows KNN predictions closely tracking the identity line. Figure 4 reveals residuals centered near zero with a positive skew: error magnitude grows with vehicle price, consistent with greater dispersion in the luxury segment. Inspection of the highest-error test-set predictions reveals five systematic failure modes.

[Figure 2: KNN validation MAE as a function of k. Optimal k=10 balances bias and variance.]
[Figure 3: KNN actual vs. predicted price (5K sample). Points cluster tightly along the identity line.]
[Figure 4: KNN residual distribution (left) and residuals vs. predicted price (right).]

**Price-cap saturation:** Percentile clipping forces heterogeneous luxury vehicles to share the $89,990 ceiling. A 2011 VW Tiguan with 115,000 miles and a 2018 Cadillac Escalade with 1,000 miles carry identical labels despite very different underlying values. KNN predicts $6,933 and $20,134 respectively (absolute errors of $83,057 and $69,856). Neither prediction is unreasonable given the available neighbors; the failures are clipping artifacts, not model mis-specification.

**Out-of-distribution exotics:** A 2007 Bugatti Veyron (5,000 miles, $89,990) is predicted at $13,563 (85% error) because no comparable training examples exist. KNN's ten nearest neighbors are ordinary mid-range vehicles; instance-based learners degrade at the distribution boundary where true neighbors are absent.

**Niche low-production models:** The 2015 VW XL1 (~250 units ever produced) is predicted at $15,189 against an actual $87,950. Unlike the exotic case, this is not a luxury vehicle but a limited-production economy car with anomalous pricing. KNN anchors to structurally unrelated Volkswagen listings, producing a plausible but wrong prediction.

**Geographic price mismatch:** DVM-CAR (96.3% of training data) consists entirely of UK listings. Old European city cars (a 2003 Seat Cordoba at £850, a 2003 Mini at £740) appear in the test set at UK auction distress prices that KNN over-predicts at ~$10K, generating spuriously high deal scores that would not survive a real-world sanity check.

**Regression to the mean on heavy tails:** KNN's weighted average cannot extrapolate beyond the price range of its local neighborhood. Listings at the extreme high end are systematically underestimated as the neighbor distribution pulls the weighted mean down, visible in Figure 4's increasing residual variance at high predicted prices.

Ridge, LinearSVR, and RBF-SVR exhibit the same failure modes with higher average error throughout, making KNN's failures the most analytically interesting. The common thread: all models concentrate errors at the distribution tails, a consequence of sparse training data there rather than model mis-specification.

### 5.4 Deal Scoring

At τ = 0.20, KNN flags 8.4% of test listings (3,493 / 41,495) as deals with average savings of $2,309 (Figure 5). The inverse relationship between threshold and flagged volume suggests well-calibrated residuals.

**Table 3: Deal-flag sensitivity to threshold τ (KNN, k=10) on the held-out test set.**

| τ    | Flagged (%) | Flagged (n) | Avg. predicted savings ($) |
|------|-------------|-------------|----------------------------|
| 0.15 | 13.4        | 5,560       | 2,254                      |
| 0.20 | 8.4         | 3,493       | 2,309                      |
| 0.30 | 3.6         | 1,494       | 2,372                      |

As τ tightens, flagged volume falls while average predicted savings per flagged listing rises monotonically, consistent with well-calibrated residuals at the right tail.

[Figure 5: Deal-score distribution (KNN, k=10) with 20% and 30% threshold markers.]

### 5.5 TabNet Training Dynamics

**Initial run.** The first TabNet run used label-encoded categoricals on the full dataset (273K rows). Training loss decreased steadily from 2.9 × 10⁸ to 7.5 × 10⁷ over 28 epochs before early stopping (best epoch 12, val MAE = $1,998). Despite convergence in loss, the high RMSE ($49,200) and low R² (0.176) indicated that a small number of extreme residuals on high-value vehicles dominated squared-error metrics, even as the median prediction quality was reasonable (MAE = $2,168).

[Figure 6: TabNet MAE loss per epoch on the initial full-data run (273K rows). Early stopping at epoch 27, best epoch 12.]

**Optimized run.** We applied two preprocessing changes while holding all hyperparameters constant (max_epochs=40, patience=15, batch_size=128) to isolate the effect of data preparation. First, we replaced label encoding with the same target-encoding and one-hot pipeline used by the baseline models, giving TabNet access to the smoothed price signal embedded in make and model. Second, we applied a log1p transform (log(1 + y)) to the target variable. We use log1p rather than a raw log because it is numerically stable near zero (log(0) is undefined, whereas log1p(0) = 0), so the transform is safe even if prices approach zero after future filtering. In practice our clipped prices start at $699, so the +1 offset has negligible effect, but log1p is the standard defensive choice. This compressed the price range from [$699, $89,990] to [6.55, 11.41] in log-space, which matters because TabNet minimizes mean squared error during training: without the transform, a $10,000 error on an $80,000 listing contributes 100× more to the loss than a $1,000 error on an $8,000 listing, causing the model to over-attend to the luxury tail at the expense of the majority of listings. In log-space, errors are proportional rather than absolute, so a 10% misprediction contributes roughly the same loss regardless of price level. At inference time, predictions were inverted with expm1 (exp(ŷ) − 1), the exact algebraic inverse of log1p, before computing dollar-scale metrics.

The effect was substantial. Training loss operated in log-space (0.47 → 0.06 over 39 epochs, best epoch 23, val MAE = 0.161 in log-units), and the resulting dollar-scale metrics improved across every measure: MAE fell from $2,168 to $1,921, RMSE from $49,200 to $3,501, and R² rose from 0.176 to 0.943. The RMSE reduction of over 13× confirms that the original R² collapse was caused by outlier-driven squared error, not by a fundamental limitation of the TabNet architecture. Residual diagnostics on the optimized run show predictions tracking the identity line with no systematic fan-out at high prices, in contrast to the initial run.

[Figure 7: TabNet training loss and validation MAE (log-space) for the optimized run. Smooth convergence over 39 epochs, best epoch 23.]

---

## 6. Discussion

**Preprocessing dominates architecture within a model family:** For TabNet, switching from label-encoded categoricals with a raw-dollar target to target-encoded categoricals with a log1p target reduced MAE from $2,168 to $1,921 and R² from 0.176 to 0.943, with the same architecture, driven entirely by how the inputs and outputs were represented. Across model families, KNN's instance-based bias still beat optimized TabNet ($1,309 vs $1,921) on every metric. Grinsztajn et al. (2022) find that tree-based models outperform deep learning on tabular data due to inductive biases suited to irregular decision boundaries; our results suggest KNN's neighborhood-based bias similarly exploits the clustered structure of used-car pricing more effectively than TabNet's attention mechanism, at least given current hyperparameters.

**Generalization and geographic distribution shift:** 96.3% of training data originates from the UK DVM-CAR dataset, with prices reflecting UK market conditions. Deploying this system in the US market would require retraining on US listings: prices for vehicles common in the US but rare in the UK (full-size trucks, domestic brands) will be systematically underestimated, while European city cars prevalent in DVM-CAR will be overestimated. The failure cases in §5.3 provide concrete illustrations of this mismatch.

**Deal score calibration:** The deal-scoring module has no ground truth for actual deals, only a threshold τ applied to regression residuals. Table 3 shows that as τ increases from 0.15 to 0.30, flagged listings drop from 13.4% to 3.6% while average predicted savings rises monotonically from $2,254 to $2,372. The monotonic trend suggests reasonable calibration (larger predicted discounts correspond to more underpriced listings), but this is unverified against actual sale outcomes. Properly calibrating τ requires a labeled holdout of listings with known transaction prices, outside the current data scope. Until then, τ should be treated as a precision-recall trade-off parameter rather than a hard semantic boundary.

**Production considerations:** KNN stores the entire 193K training set at inference time and runs a full nearest-neighbor search per query. For a deployed real-time system, an approximate nearest-neighbor index such as FAISS (Johnson et al., 2019) or a gradient-boosted tree trained on the same feature set would be worth evaluating as a faster alternative with comparable accuracy.

---

## 7. Conclusion and Future Work

Our work shows that a two-stage machine learning pipeline can bridge the information gap in the used-car market. We unified over 276,000 listings and built a framework for fair market value estimation that works beyond simple linear assumptions. The market data is noisy, but an underlying structure is recoverable with models capable of capturing nonlinear depreciation.

The KNN (k=10, distance-weighted) model was the strongest performer, explaining 97.1% of market variance with a Mean Absolute Error of $1,309. This reflects the local nature of car pricing, where a vehicle's value is most accurately determined by its immediate neighborhood in the feature space. Our optimized TabNet narrowed the gap between deep learning and traditional methods, reaching a competitive R² of 0.943. Converting these residuals into a deal score identified the top 8.4% of listings as deals, with average predicted savings of $2,309 per transaction.

The current system provides high precision and we have identified several avenues to continue the work and make the system more efficient:
- ANN Integration: Our results revealed a significant computational trade-off, with KNN requiring over 75 seconds for inference. Future iterations can implement FAISS or other Approximate Nearest Neighbor libraries to enable real-time scaling for millions of listings.
- Ensemble Modeling: We plan to explore a weighted stacking approach that blends KNN's local precision with TabNet's global feature attention to potentially surpass the individual performance of either model.
- Feature Enrichment: Incorporating regional economic indicators and physical vehicle condition reports will likely reduce errors on luxury-segment outliers where maintenance history is as critical as mileage.
- Temporal Analysis: To account for market drift, future work will involve training models on a sliding window to adapt to rapid inflation or supply chain shifts.

---

## References

- Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679–6687.
- Dudani, S. A. (1976). The distance-weighted k-nearest-neighbor rule. *IEEE Transactions on Systems, Man, and Cybernetics*, 6(4), 325–327.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *Advances in Neural Information Processing Systems*, 35.
- Gupta, V. K. (2024). Car price dataset. *Hugging Face Datasets*.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Huang, J., Chen, B., Luo, L., Yue, S., & Ounis, I. (2022). DVM-CAR: A large-scale automotive dataset for visual marketing research and applications. In *Proceedings of the 2022 IEEE International Conference on Big Data* (pp. 4140–4147).
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
- Li, T., & Lin, J. (2021). Used car price prediction using machine learning. *Technical Report, Stanford University, CS230*.
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32.
- Schlimmer, J. C. (1985). Automobile dataset. *UCI Machine Learning Repository*.
- Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199–222.
- Vaneesha, B., Reddy, P. S., Kumar, A., & Rao, V. (2024). Predicting car prices using machine learning: A comparative study of KNN and SVM regressors. *International Journal of Current Science Research and Review*, 7(9).
- Zhu, A. (2023). Pre-owned car price prediction using machine learning techniques. In *Proceedings of the 1st International Conference on Data Analysis and Machine Learning (DAML)* (pp. 356–360).
- Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019). Car price prediction using machine learning techniques. *TEM Journal*, 8(1), 113–118.


