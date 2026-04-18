# Notebook Notes

The original exploratory workflow for this project lives in the local notebook:

`Credit_Card_Fraud_Detection.ipynb`

This repository now packages that notebook work into reusable Python modules under `src/fraud_detection/`, so the project is easier to run, explain, and extend.

The notebook-based workflow covers:
- exploratory data analysis
- class imbalance handling with SMOTE
- model comparison across Logistic Regression, Random Forest, XGBoost, and a DNN
- artifact saving for trained models and summary reports
