# Job Description Classification Project

This repository contains a collection of Python scripts designed to classify job descriptions based on various criteria such as **Single Response**, **Matrix Response**, and **Target Response**. Each script corresponds to a specific classification task and uses its respective dataset.

## Project Overview

The objective of this project is to analyze job descriptions and classify them into predefined categories based on the questions provided in the dataset. The project leverages:
- **TF-IDF Vectorization** for text feature extraction.
- **Logistic Regression** as the classification algorithm.
- **GridSearchCV** for hyperparameter optimization.
- 

Each script is tailored to address a specific element of job descriptions, such as:
- Liaison and Networking
- Decision Making
- Teamwork and Motivation
- Work Environment
- Initiative and Problem Solving
- Teaching and Learning Support
- Analysis and Research
- Planning and Organising Resources
- Service Delivery
- Teamwork and Motivation
- Knowledge and Experience
  
### Datasets
Each dataset is stored in the `data/` folder and contains:
- **Job Description**: Text describing the job role.
- **Target Variables**: Questions or criteria (Single, Matrix & Target Response) for classification (e.g., Q12, Q13, etc.).

### Scripts
Each Python script performs the following tasks:
1. **Data Preprocessing**: Cleans and prepares the dataset.
2. **Model Training**: Trains a Logistic Regression model with hyperparameter tuning.
3. **Evaluation**: Reports classification metrics.
4. **Prediction**: Makes predictions for new job descriptions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/2311795/Maxwell-Text-Classification



## Installation

To install the necessary libraries, run:

```bash
pip install -r requirements.txt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2311795/Maxwell-Text-Classification/blob/main/main.ipynb)

