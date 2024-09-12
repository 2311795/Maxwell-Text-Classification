# Multi-Label Job Description Classification with BERT, SMOTEENN, and Ensemble Models

## Overview

This project implements a multi-label classification pipeline to classify job descriptions into multiple categories based on responses to several questions. The pipeline incorporates BERT embeddings for text processing, SMOTEENN for handling imbalanced datasets, and various machine learning models including Random Forest, Gradient Boosting, and XGBoost. Model interpretability is achieved using SHAP (SHapley Additive exPlanations).

Two versions of the code are provided:
- **Code 1**: Uses the original dataset (`Matrix.csv`).
- **Code 2**: Uses a modified dataset with synthetic data added to improve model accuracy (`Matrix_synthetic.csv`).

## Prerequisites

Ensure you have Python installed along with the required libraries. You can install the necessary libraries using `pip`:

```bash
pip install shap pandas scikit-learn transformers torch imbalanced-learn xgboost
```
## Datasets

The project uses two datasets:

- **Original Dataset**: `Matrix.csv` - Contains job descriptions and responses to several questions.
- **Synthetic Dataset**: `Matrix_synthetic.csv` - This dataset includes synthetic data added to the original dataset to address class imbalance and improve accuracy.

Both datasets should include the following columns:

- **`Job description`**: Text describing the job.
- **`Question 12`, `Question 13`, `Question 14`, `Question 15`**: Columns representing the responses to different questions, with responses labeled as 'A', 'B', 'C', or 'D'.
