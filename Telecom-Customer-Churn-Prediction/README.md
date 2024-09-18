# Customer Churn Prediction Using Machine Learning Algorithms

This project focuses on predicting customer churn using different machine learning algorithms. The dataset contains customer details such as account length, usage, and service plans. Several models like Logistic Regression, Decision Tree, and KNN are trained and tested for accuracy. Additionally, hyperparameter tuning is performed to improve model performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
  - Logistic Regression
  - Decision Tree
  - KNN
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The goal of this project is to develop machine learning models that can predict customer churn based on various customer features. The data is preprocessed, visualized, and used to train different models, which are then evaluated for accuracy and performance. This is a common business problem where companies aim to identify customers likely to leave their services.

## Dataset

The dataset consists of customer-related attributes such as:

- State
- Account Length
- International Plan
- Voice Mail Plan
- Total Day/Evening/Night Minutes, Calls, and Charges
- Customer Service Calls
- Churn (Target variable)

## Requirements

To run this project, install the following dependencies:

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install the dependencies using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```
Navigate to the project directory:

```bash
cd customer-churn-prediction
```
Install the required libraries as shown above.

## Data Preprocessing
Label Encoding: 
Categorical features like 'International Plan', 'Voice Mail Plan', and 'Churn' are transformed into numerical representations using Label Encoding.
Missing Values Check: The dataset is checked for missing values, though none were found in this case.

## Exploratory Data Analysis
Visualizations: The relationship between churn and various features like 'Total Day Minutes' is explored using bar plots to gain insights.

## Machine Learning Models
### Logistic Regression
Logistic Regression is used as the initial model to predict churn.
The model is trained on the training dataset and predictions are made on both training and test sets.
### Decision Tree Classifier
A Decision Tree Classifier is trained on the dataset and evaluated based on accuracy.
### K-Nearest Neighbors (KNN)
KNN is implemented with the Elbow Method to find the optimal value for 'k'.
### Model Evaluation
Each model's performance is evaluated using the following metrics:

- Accuracy: Measured on both training and test datasets.
- Confusion Matrix: To evaluate the true positives, true negatives, false positives, and false negatives.
- Classification Report: Provides precision, recall, and F1 score.
## Hyperparameter Tuning
Grid Search is used to fine-tune hyperparameters for the KNN model.
The optimal number of neighbors (k) is found to improve accuracy.

## Results
Logistic Regression, Decision Tree, and KNN models are evaluated.
KNN with hyperparameter tuning yielded the best results based on accuracy.

## Conclusion
This project showcases how different machine learning algorithms can be applied to predict customer churn. The KNN model with tuned hyperparameters performed the best. Future improvements could include trying more complex models like Random Forest or Neural Networks.

