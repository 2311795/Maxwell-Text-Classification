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
