# Credit Card Fraud Detection — End-to-End Machine Learning

An end-to-end machine learning project for detecting fraudulent credit card transactions using classical supervised learning techniques.

This project was created as a final practical consolidation of my classical Machine Learning fundamentals, covering the complete workflow from data understanding and EDA to model comparison, cross-validation, hyperparameter tuning, error analysis, threshold tuning, and model interpretation.

---

## Problem Statement

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions are extremely rare compared to legitimate transactions.

The goal of this project is to predict whether a transaction is:

* `0` → Legitimate
* `1` → Fraudulent

The main challenge is not simply achieving high accuracy, but detecting as many fraudulent transactions as possible while keeping false fraud alerts under control.

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset containing transactions made by European cardholders.

### Dataset characteristics

* 284K+ transactions after cleaning
* 30+ predictive features
* Highly imbalanced target
* Fraud transactions represent only about **0.17%** of the dataset
* Features include anonymized PCA components (`V1`–`V28`), `Time`, and `Amount`

> The original dataset is not included in this repository because of its large file size. A dataset download/source link can be added to the repository separately.

---

## Project Workflow

The notebook follows a complete end-to-end ML workflow:

1. Problem Understanding
2. Data Loading & Basic Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Feature Selection
7. Stratified Train-Test Split
8. Preprocessing & Scaling
9. Baseline Model
10. Model Comparison
11. Cross Validation
12. Hyperparameter Tuning
13. Final Model Evaluation
14. Error Analysis
15. Threshold Tuning
16. Model Interpretation
17. Final Conclusion

---

## Data Cleaning

The dataset was checked for:

* Missing values
* Duplicate records
* Basic data consistency

No missing values were found.

Duplicate rows were identified and removed before model development.

---

## Exploratory Data Analysis

The EDA focused on:

* Target class distribution
* Numerical feature distributions
* Transaction amount distribution
* Transaction time distribution
* Fraud vs legitimate transaction comparison
* Feature-target relationships
* Correlation analysis
* Outlier investigation

### Key observations

* The target was extremely imbalanced, with approximately **99.83% legitimate** and **0.17% fraudulent** transactions.
* Transaction `Amount` was highly right-skewed.
* Fraudulent transactions showed different patterns in `Time` and `Amount` compared with legitimate transactions.
* Features such as `V14`, `V17`, `V12`, `V10`, and `V4` showed relatively stronger relationships with the fraud target.

---

## Feature Engineering

The `Amount` feature was highly right-skewed, so a logarithmic transformation was created:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

This reduced the influence of extreme transaction amounts while keeping the original `Amount` feature available for comparison.

---

## Feature Selection

Feature selection checks included:

* Constant / near-constant feature detection
* Highly correlated feature pairs
* Feature-target correlation

No major feature removal was performed based only on simple correlation checks because weak linear correlation does not necessarily mean that a feature is useless for a nonlinear model.

---

## Handling Class Imbalance

Because fraud represented only around **0.17%** of all transactions, accuracy was not treated as the primary metric.

The project focused mainly on:

* Precision
* Recall
* F1-score
* PR-AUC
* ROC-AUC
* Confusion Matrix

Class imbalance was also considered during model training using class weighting for the Random Forest and positive-class weighting for XGBoost.

---

## Models Tested

The project included the following major supervised learning approaches:

### Logistic Regression

Used as the baseline model.

### Random Forest

Used as the main ensemble model.

### XGBoost

Used as the main boosting model.

The goal was not to run every possible algorithm, but to compare models that provide meaningfully different approaches to the classification problem.

---

## Cross Validation

Because the target was highly imbalanced, **Stratified K-Fold Cross Validation** was used.

The selected metric for comparison was **F1-score**, because it balances precision and recall.

Random Forest achieved a higher mean cross-validation F1-score than XGBoost.

---

## Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Random Forest model.

Best configuration found:

```text
class_weight      = balanced_subsample
max_depth         = None
min_samples_leaf  = 1
min_samples_split = 5
n_estimators      = 107
```

Best cross-validation F1-score:

```text
0.8522
```

---

## Final Model

The final model selected for the project was:

### Tuned Random Forest Classifier

Default threshold (`0.5`) test performance:

| Metric    | Score |
| --------- | ----: |
| Precision |  0.97 |
| Recall    |  0.73 |
| F1-score  |  0.83 |
| ROC-AUC   | 0.950 |
| PR-AUC    | 0.820 |

Confusion matrix:

```text
TN = 56649
FP = 2
FN = 26
TP = 69
```

---

## Threshold Tuning

Since fraud detection is sensitive to missed fraud cases, different probability thresholds were investigated.

The best observed F1-score was obtained at a threshold of **0.2**:

| Metric    | Threshold = 0.2 |
| --------- | --------------: |
| Precision |           0.915 |
| Recall    |           0.789 |
| F1-score  |           0.847 |

Lowering the threshold increased recall while maintaining relatively high precision.

For a production system, the threshold should be selected using a validation set or cross-validation based on the actual business cost of false positives and false negatives.

---

## Model Interpretation

Random Forest feature importance was used to understand which features contributed most to the model's decisions.

Top influential features included:

1. `V14`
2. `V10`
3. `V12`
4. `V4`
5. `V17`
6. `V11`
7. `V3`

These features also showed relatively strong relationships with the target during EDA.

However, feature importance does not imply causation.

---

## Key Learnings

This project helped consolidate the complete classical ML workflow:

* Data cleaning
* EDA
* Feature engineering
* Feature selection
* Train-test splitting
* Stratified sampling
* Feature scaling
* Baseline modelling
* Model comparison
* Ensemble learning
* Boosting
* Cross-validation
* Hyperparameter tuning
* Imbalanced classification
* Error analysis
* Threshold tuning
* Model interpretation

The project also reinforced an important lesson:

> For highly imbalanced classification problems, accuracy alone can be misleading. Model selection should consider the trade-off between precision, recall, and the actual business cost of false positives and false negatives.

---

## Future Improvements

Possible next steps include:

* Validation-based threshold optimization
* SMOTE / alternative imbalance-handling strategies
* Precision-Recall curve analysis
* SHAP-based model interpretation
* More systematic feature selection
* Model deployment using an API or Streamlit
* Monitoring model performance on new transactions

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook / Google Colab

---

## Project Status

**Classical Machine Learning End-to-End Project — Completed**

This project serves as a practical wrap-up of my classical supervised machine learning learning phase before moving towards **Deep Learning and modern AI/GenAI concepts**.
