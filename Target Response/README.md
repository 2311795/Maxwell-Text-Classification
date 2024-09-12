# Multi-Label Job Description Classification with Gradient Boosting

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#projectstructure)
- [Usage](#usage)
- [License](#license)
- [References](#references)

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

##### Clone the Repository:

```bash

git clone https://github.com/2311795/Maxwell-Text-Classification.git
cd Maxwell-Text-Classification
```
##### Run the Script:
Execute the script to perform the classification:
```bash
Target Response.ipynb
```
##### Evaluate the Model:
The script will output the averaged accuracy, precision, recall, and F1 score across all questions. It will also print the predicted responses for a sample new job description and visualize confusion matrices for the predictions.

The script will produce output like:
```
Job Description: To provide an effective Joinery resource to ensure the University fabric is efficiently maintained...
Predicted Responses: {'Question 7': 'A', 'Question 8': 'C', 'Question 9': 'B', 'Question 10': 'D', 'Question 11': 'A'}

Accuracy: 0.8524
Precision: 0.8413
Recall: 0.8379
F1 Score: 0.8396
```
##### Custom Prediction Adjustment Rule
The script includes a custom rule where, if a job description is predicted to have an 'A' in any response, all preceding responses are automatically set to 'D'. This rule is applied after the initial predictions and can be adjusted according to specific business needs.

##### Visualization
Confusion matrices are generated for each question to visualize the performance of the model. The matrices help in understanding the distribution of true and predicted labels.

## References

1. **Pandas Documentation**: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
2. **Scikit-Learn User Guide**: [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
3. **Seaborn Documentation**: [Seaborn Documentation](https://seaborn.pydata.org/)
4. **Gradient Boosting in Scikit-Learn**: [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
