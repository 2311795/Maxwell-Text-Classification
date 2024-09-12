# Multi-Label Job Description Classification with Gradient Boosting

## Overview

This project implements a machine learning pipeline to classify job descriptions into multiple categories based on responses to a set of predefined questions. The model uses a Gradient Boosting classifier wrapped in a `MultiOutputClassifier` to handle the multi-label classification task. The project also includes hyperparameter tuning using `GridSearchCV` and a custom rule to adjust predictions.

## Installation

Ensure you have Python installed along with the required libraries. You can install the necessary libraries using `pip`:

```bash
pip install pandas scikit-learn seaborn
```

## Dataset

The project assumes you have a CSV file named `Target Response DB.csv` located in your file system. This file should contain job descriptions and their corresponding responses to multiple questions. The relevant columns include:

- **`Job description`**: Text describing the job.
- **`Question 7`, `Question 8`, `Question 9`, `Question 10`, `Question 11`**: Columns representing the responses to different questions, with responses labeled as 'A', 'B', 'C', or 'D'.

## Project Structure

The project follows these main steps:

1. **Load the Dataset**: Read the CSV file into a pandas DataFrame.
2. **Encode Responses**: Convert categorical responses ('A', 'B', 'C', 'D') into numeric labels (0, 1, 2, 3) for compatibility with machine learning models.
3. **Text Preprocessing and Vectorization**: Convert job descriptions into TF-IDF features, using bigrams and removing common English stopwords.
4. **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets.
5. **Model Definition**: Define a Gradient Boosting model wrapped in a `MultiOutputClassifier`.
6. **Hyperparameter Tuning**: Use `GridSearchCV` to optimize model parameters.
7. **Model Training**: Train the model on the training set using the best hyperparameters.
8. **Prediction and Evaluation**: Predict on the test set and evaluate model performance using accuracy, precision, recall, and F1 score.
9. **Custom Rule for Adjusting Predictions**: Implement a custom rule where once an 'A' is found, all preceding responses are set to 'D'.
10. **Example Prediction**: Use the model to predict responses for a new job description.
11. **Visualization**: Generate and visualize confusion matrices for the predicted responses.

## Usage

### Clone the Repository:

```bash
git clone https://github.com/yourusername/job-description-classification.git
cd job-description-classification
