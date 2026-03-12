# Sections 5 & 6 -- Compact (for 4-page ACL paper)

## 5. Methodology

### 5.1 Preprocessing

After merging three data sources (Hugging Face, DVM-CAR, UCI), we removed the 201 UCI rows due to missing year, mileage, and model fields, yielding 276,627 listings with 9 features. Prices were clipped at the 1st and 99th percentiles ($699 to $89,990) to remove outliers. High-cardinality categoricals (make: 86 unique, model: 887 unique) were target-encoded (Micci-Barreca, 2001) to preserve price signal without dimensionality explosion. Low-cardinality categoricals (fuel_type, transmission) were one-hot encoded. Numerical features (year, engine_size, mileage) were z-score standardized. All encoding was fit on training data only. The dataset was split 70/15/15 into train (193,638), validation (41,494), and test (41,495) sets.

### 5.2 Models

**Ridge Regression** (Hoerl and Kennard, 1970) serves as our linear baseline, with alpha tuned via 5-fold cross-validation (optimal alpha = 0.02).

**K-Nearest Neighbors** predicts price as the distance-weighted mean of k nearest neighbors. Vaneesha et al. (2024) found KNN competitive with SVM for car pricing. We tuned k over [3, 30] with 3-fold CV on a 50K subsample.

**Support Vector Regression** was evaluated in linear (LinearSVR) and RBF-kernel variants. Both were trained on a 30K subsample due to O(n^2) memory scaling.

**TabNet** (Arik and Pfister, 2021) is our complex neural network. It uses sequential sparse attention masks to select features at each decision step, qualifying as an attention-based architecture. [Mark: insert architecture params and training details here.]

### 5.3 Deal Scoring

We convert regression output to a deal score: (predicted - actual) / predicted. Listings exceeding a threshold tau are classified as deals.

## 6. Initial Results

### 6.1 Regression

Table 1 reports test set performance. KNN achieves the lowest MAE of $1,309 (13.5% of median price $9,699), explaining 97.1% of variance. Ridge captures only 82.2%, confirming strong non-linear structure in car pricing. SVR underperforms due to its 30K training subsample.

| Model | MAE ($) | RMSE ($) | R^2 |
|-------|---------|----------|-----|
| KNN (k=10) | 1,309 | 2,512 | 0.971 |
| Ridge | 3,876 | 6,180 | 0.822 |
| LinearSVR | 4,271 | 6,821 | 0.784 |
| SVR (RBF) | 5,206 | 8,679 | 0.650 |
| TabNet* | 23,386 | -- | -- |

*Initial TabNet run used 3 features on 1K rows. Full-feature run pending.

[INSERT Figure 1 -- fig1_model_comparison.png]

### 6.2 Classification

We discretized prices into Low (<$5K), Mid ($5K-$15K), and High (>$15K) bins to compute required classification metrics (Table 2).

| Model | Acc | Prec | Rec | F1 |
|-------|-----|------|-----|-----|
| KNN | .929 | .929 | .929 | .929 |
| Ridge | .821 | .825 | .821 | .820 |
| LinearSVR | .746 | .773 | .746 | .739 |
| SVR (RBF) | .692 | .706 | .692 | .687 |

### 6.3 Deal Scoring

At tau = 0.20, KNN flags 8.4% of test listings (3,493 / 41,495) as deals with average savings of $2,309.

[INSERT Figure 5 -- fig5_deal_scores.png]

### 6.4 Discussion

Feature engineering dominated model choice: adding target-encoded make/model reduced MAE 18x compared to TabNet's initial 3-feature run. KNN's strength is domain-specific, as car pricing is naturally clustered by make, model, year, and mileage.

[INSERT Figure 3 -- fig3_actual_vs_predicted.png]
