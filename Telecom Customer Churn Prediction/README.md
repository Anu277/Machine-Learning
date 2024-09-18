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
