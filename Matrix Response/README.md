# Multi-Label Job Description Classification with BERT, SMOTEENN, and Ensemble Models

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Project Structure](#projectstructure)
- [Usage](#usage)
- [License](#license)
- [References](#references
  
## Overview

This project implements a multi-label classification pipeline to classify job descriptions into multiple categories based on responses to several questions. The pipeline incorporates BERT embeddings for text processing, SMOTEENN for handling imbalanced datasets, and various machine learning models including Random Forest, Gradient Boosting, and XGBoost. Model interpretability is achieved using SHAP (SHapley Additive exPlanations).

Two versions of the code are provided:
- **Code 1**: Uses the original dataset (`Matrix.csv`).
- **Code 2**: Uses a modified dataset with synthetic data added to improve model accuracy (`Matrix_synthetic.csv`).

## Installation

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

## Project Structure

The project follows these main steps:

1. **Load the Dataset**: Read the CSV file into a pandas DataFrame.
2. **Encode Responses**: Convert categorical responses ('A', 'B', 'C', 'D') into numeric labels (0, 1, 2, 3) for compatibility with machine learning models.
3. **Convert Text to BERT Embeddings**: Use BERT to convert job descriptions into embeddings.
4. **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets.
5. **Create a Pipeline**: Use SMOTEENN to handle class imbalance and a classifier for prediction.
6. **Hyperparameter Tuning**: Use RandomizedSearchCV to optimize model parameters.
7. **Model Stacking**: Combine multiple models using a stacking classifier for better performance.
8. **Model Interpretation with SHAP**: Use SHAP to explain model predictions.
9. **Predicting New Job Descriptions**: Use the trained model to predict responses for new job descriptions.
10. **Evaluation Metrics**: Calculate accuracy, classification report, and confusion matrix for each question.

## Usage

### Clone the Repository:

```bash
git clone https://github.com/2311795/Maxwell-Text-Classification.git
cd Maxwell-Text-Classification
```
##### Run the Script:
Execute the script corresponding to the dataset you want to use:
```bash
Matrix Response.ipynb
```
#### Evaluate the Model:
The script will output the averaged accuracy, precision, recall, and F1 score across all questions. It will also print the predicted responses for a sample new job description and visualize SHAP values and confusion matrices for the predictions.

#### Model Interpretability with SHAP
SHAP (SHapley Additive exPlanations) is used to interpret the model predictions. The script generates SHAP summary plots for the models used in the stacking ensemble (Random Forest, Gradient Boosting, and XGBoost).

#####Improvements with Synthetic Data
In Code 2, synthetic data was added to the original dataset to improve class balance. This led to better model performance, as reflected in higher accuracy and other evaluation metrics.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. **Pandas Documentation**: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
2. **Scikit-Learn User Guide**: [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
3. **Transformers Documentation**: [Transformers Documentation](https://huggingface.co/transformers/)
4. **SHAP Documentation**: [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
5. **Imbalanced-Learn Documentation**: [Imbalanced-Learn Documentation](https://imbalanced-learn.org/stable/)
6. **XGBoost Documentation**: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
