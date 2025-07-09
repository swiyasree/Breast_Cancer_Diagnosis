# Breast Cancer Diagnosis using L1-Regularized SVM and Semi-Supervised Learning

This notebook implements a classification pipeline for breast cancer diagnosis using a Linear SVM with L1 regularization and evaluates model performance over multiple runs.

## Objectives
- Train an L1-penalized Linear SVM with hyperparameter tuning
- Evaluate performance across 30 runs to assess model robustness
- Use precision, recall, F1-score, and AUC as metrics
- Visualize confusion matrices and ROC curves
- Explore semi-supervised learning using partially labeled data

## Key Techniques
- Hyperparameter tuning via GridSearchCV
- Model robustness via repeated training
- ROC curve and AUC analysis
- Use of StandardScaler, confusion matrices
- Semi-supervised setup for realistic training conditions

## Libraries Used
- scikit-learn
- numpy, pandas
- matplotlib
