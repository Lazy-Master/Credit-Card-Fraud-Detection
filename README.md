# Credit Card Fraud Detection

Comparative study of machine learning techniques for fraud detection on highly imbalanced transaction data. Implements and evaluates multiple classifiers with ensemble methods, threshold optimization, and cost-sensitive learning.

## Overview

| Aspect | Details |
|--------|---------|
| Dataset | European Credit Card Transactions (284,807 records) |
| Class Imbalance | 0.17% fraud (492 out of 284,807) |
| Models | Logistic Regression, Random Forest, XGBoost, DNN |
| Ensemble Methods | Soft Voting, Hard Voting, Stacking |
| Evaluation | Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC |

## Techniques

### Models Implemented

- **Logistic Regression** — L2 regularized linear classifier with LBFGS optimizer
- **Random Forest** — Bagging ensemble with Gini criterion, 100 estimators
- **XGBoost** — Gradient boosting with L1/L2 regularization
- **Deep Neural Network** — 3-layer MLP (64→32→16→1) with dropout and early stopping

### Imbalance Handling

- **SMOTE** — Synthetic Minority Over-sampling on training data only
- **Cost-Sensitive Learning** — XGBoost with `scale_pos_weight` adjustment
- **Leakage-Free Pipeline** — Split → Fit scaler on train → Transform → SMOTE on train

### Ensemble Methods

- **Soft Voting** — Average predicted probabilities across models
- **Hard Voting** — Majority vote (3 of 4 models must agree)
- **Stacking** — Logistic Regression meta-learner on base model predictions

### Evaluation & Validation

- **Threshold Optimization** — Youden's J statistic for optimal decision boundary
- **5-Fold Stratified Cross-Validation** — SMOTE applied inside each fold
- **McNemar's Test** — Statistical significance between model pairs
- **Feature Importance** — Random Forest Gini importance analysis

## Preprocessing Pipeline

```
Raw Data → Train/Test Split (80/20, stratified)
         → RobustScaler (fit on train only)
         → SMOTE (train only)
         → Model Training
```

RobustScaler uses median and IQR, making it resistant to outliers in transaction amounts.

## Project Structure

```
Credit-Card-Fraud-Detection/
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
tensorflow
joblib
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

## License

MIT
