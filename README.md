# 💳 Credit Card Fraud Detection with Machine Learning

A binary classification project that detects fraudulent credit card transactions from a highly imbalanced, anonymized real-world dataset. Four supervised learning algorithms are trained and compared, with a focus on evaluation metrics that are meaningful under severe class imbalance.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-DataAnalysis-150458)
![Status](https://img.shields.io/badge/Status-Baseline%20Complete-yellow)

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Modeling & Evaluation](#modeling--evaluation)
- [Results](#results)
- [Critical Analysis & Next Steps](#critical-analysis--next-steps)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## 💡 Problem Statement

Credit card fraud occurs when an account is used by someone other than its owner without authorization. The goal of this project is to build a model that flags fraudulent transactions from transaction-level data, so that card issuers can intervene before legitimate customers are charged for purchases they never made.

This is framed as a **supervised binary classification** problem (`Class = 1` → fraud, `Class = 0` → legitimate), with the explicit objective of **maximizing detected fraud while minimizing false fraud flags** — i.e., balancing recall and precision rather than optimizing for raw accuracy alone.

---

## 📊 Dataset

The project uses the well-known **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**, containing transactions made by European cardholders over two days in September 2013.

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | **0.17%** |
| Features | `V1`–`V28` (PCA components), `Time`, `Amount`, `Class` |

- `V1`–`V28` are anonymized principal components derived via PCA — the original features were not disclosed for confidentiality.
- `Time`: seconds elapsed since the first transaction in the dataset.
- `Amount`: transaction amount, usable for cost-sensitive learning.
- `Class`: target variable (1 = fraud, 0 = legitimate).

---

## 🔍 Exploratory Data Analysis

- **Severe class imbalance:** Of 284,807 transactions, only 492 (0.17%) are fraudulent. This single fact drives almost every modeling and evaluation decision in the project — accuracy alone is not a meaningful metric on this dataset (a model predicting "non-fraud" for everything would already score ~99.83% accuracy).
- **Transaction amount statistics:**

| Statistic | Non-Fraud | Fraud |
|-----------|-----------|-------|
| Count | 284,315 | 492 |
| Mean | 88.29 | 122.21 |
| Std | 250.11 | 256.68 |
| Min | 0.00 | 0.00 |
| 25% | 5.65 | 1.00 |
| 50% (median) | 22.00 | 9.25 |
| 75% | 77.05 | 105.89 |
| Max | 25,691.16 | 2,125.87 |

Fraudulent transactions tend to have a **higher mean amount but lower median** than legitimate ones, with a much narrower maximum — suggesting fraud is concentrated in small-to-mid value transactions rather than extreme outliers.

---

## 🛠️ Preprocessing

- **Dropped `Time`** — it represents elapsed seconds within a 2-day window and was not used as a model feature.
- **Feature scaling:** `Amount` varies on a much larger scale than the PCA-derived `V1`–`V28` features (which are already roughly standardized), so `StandardScaler` was identified as the appropriate normalization step before modeling.
- **Train/test split:** 80/20 split (`test_size=0.2`, `random_state=0`) using `train_test_split`.

```python
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

---

## 🤖 Modeling & Evaluation

Four classifiers were trained on the same train/test split:

| Model | Configuration |
|-------|----------------|
| Decision Tree | `max_depth=4`, `criterion='entropy'` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Logistic Regression | default parameters |
| Random Forest | `max_depth=4` |

Each model was evaluated with three metrics, deliberately chosen because of the class imbalance:

- **Accuracy score** — included for completeness, but interpreted with caution.
- **F1 score** — harmonic mean of precision and recall; the primary metric used to compare models, since it balances false positives and false negatives on the minority (fraud) class.
- **Confusion matrix** — used to directly inspect how many fraud cases were caught vs. missed by each model.

---

## 📈 Results

### Accuracy & F1 Score

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Decision Tree** | 0.99968 | **0.8105** |
| K-Nearest Neighbors | 0.99933 | 0.7865 |
| Random Forest | 0.99928 | 0.7657 |
| Logistic Regression | 0.99916 | 0.7209 |

> All models score ≈99% accuracy — a direct consequence of the 0.17% fraud rate rather than a meaningful sign of quality. **F1 score is the more informative metric here**, and ranks the models in the order above.

### Confusion Matrices

| Model | TN (Non-fraud → Non-fraud) | FP (Non-fraud → Fraud) | FN (Fraud → Non-fraud) | TP (Fraud → Fraud) |
|-------|----|----|----|----|
| **Decision Tree** | 56,849 | 12 | 24 | **77** |
| K-Nearest Neighbors | 56,854 | 7 | 31 | 70 |
| Random Forest | 56,854 | 7 | 33 | 67 |
| Logistic Regression | 56,852 | 9 | 39 | 62 |

**Decision Tree** correctly identified the most fraud cases (77 of 101 fraud cases in the test set, FN=24) while keeping false positives reasonably low (12), giving it both the highest F1 score and the best fraud-recall trade-off among the four models tested. **Logistic Regression** caught the fewest fraud cases (62, FN=39) and was the weakest performer overall.

---

## 🧭 Critical Analysis & Next Steps

This project is a solid first-pass baseline that correctly identifies the central challenge of the dataset — **severe class imbalance** — and chooses F1 score over accuracy accordingly. That said, a few areas would meaningfully strengthen the analysis in a follow-up iteration:

- **Accuracy is reported but barely informative here.** All four models land within 0.0005 of each other on accuracy (99.92%–99.97%), which tells us almost nothing — a trivial "always predict non-fraud" classifier would score 99.83%. Leading with F1, and ideally **precision-recall AUC (AUPRC)**, would better reflect real model quality on this dataset.
- **No resampling techniques were applied.** With only 492 fraud cases against 284,315 legitimate ones, techniques like **SMOTE, random undersampling, or class-weighted loss** (`class_weight='balanced'` in scikit-learn) typically improve minority-class recall substantially and would be a natural next experiment.
- **Shallow trees (`max_depth=4`) for Decision Tree and Random Forest** were not tuned — these depths were likely chosen to avoid overfitting on the minority class, but a proper **`GridSearchCV` with stratified k-fold cross-validation** (`StratifiedKFold`) would validate whether this depth is optimal or overly conservative.
- **`StandardScaler` was identified as necessary for the `Amount` feature but its application to the modeling pipeline isn't explicitly shown** — confirming it's applied consistently to train and test sets (fit on train only, to avoid leakage) would be an important verification step.
- **Single train/test split, no cross-validation.** Given how few fraud examples exist (492 total, ~101 in a 20% test split), results can vary noticeably depending on which fraud cases land in the test set. Cross-validation would give a more stable estimate of each model's true performance.
- **Threshold tuning was not explored.** Logistic Regression and Random Forest output probabilities — adjusting the classification threshold (instead of the default 0.5) could shift the precision/recall trade-off in favor of catching more fraud, which is often the business priority even at the cost of more false positives.
- **Untried approaches:** Gradient boosting models (XGBoost, LightGBM) with `scale_pos_weight`, and anomaly-detection-based methods (Isolation Forest, Local Outlier Factor) are commonly used on this exact dataset and would be worthwhile comparisons.
- **Two days of data is a narrow window**, as noted in the original analysis — temporal patterns (time-of-day fraud spikes, etc.) couldn't be captured here, and a production system would need continuous retraining as fraud patterns evolve.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
├── tree_cm_plot.png         # Confusion matrix - Decision Tree
├── knn_cm_plot.png          # Confusion matrix - KNN
├── lr_cm_plot.png           # Confusion matrix - Logistic Regression
├── rf_cm_plot.png           # Confusion matrix - Random Forest
└── README.md
```

---

## ▶️ How to Run

1. Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory (update the file path in the notebook if needed).
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn termcolor
   ```
3. Run the notebook end-to-end. It will print case counts, amount statistics, accuracy/F1 scores for all four models, and save confusion matrix plots for each.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Modeling:** scikit-learn (`DecisionTreeClassifier`, `KNeighborsClassifier`, `LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`)
- **Evaluation:** Accuracy score, F1 score, confusion matrix

---

## 📚 References

1. [Credit Card Fraud Detection with Machine Learning in Python – Medium](https://medium.com/datazen/credit-card-fraud-detectionwith-machine-learning-in-python-ac7281991d87)
2. [Data Science Project – Credit Card Fraud Detection – DataFlair](https://data-flair.training/blogs/data-science-machine-learning-project-credit-card-fraud-detection/)
3. [ML | Credit Card Fraud Detection – GeeksforGeeks](https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/)
4. Maniraj, S.P., Saini, A., Sarkar, S.D., Ahmed, S. — *Credit Card Fraud Detection using Machine Learning and Data Science*, IJERT V8I090031.
5. Géron, A. — *Hands-On Machine Learning with Scikit-Learn and TensorFlow*.

---

## 👤 Author

**[Your Name]**
📧 [your.email@example.com] | 🔗 [LinkedIn](https://linkedin.com/in/your-profile) | 💼 [Portfolio](https://your-portfolio.com)

---

⭐ If you found this project useful or interesting, consider giving it a star!
