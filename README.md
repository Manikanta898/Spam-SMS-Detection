# Spam-SMS-Detection

A Machine Learning project that classifies SMS messages as spam or ham using Natural Language Processing (NLP) and Supervised Learning.

## Table of Contents
- [Getting Started](#getting-started)
- [Data Sources](#data-sources)
- [File Descriptions](#file-descriptions)
- [Machine Learning Model](#machine-learning-model)
- [Technologies Used](#technologies-used)
- [Usage](#usage)

## Getting Started
To run the analysis locally, clone this repository and install the dependencies.

## Data Sources
The data used in this project was sourced from kaggle.com

## File Descriptions
- Spam_SMS_Detection.ipynb : Colab notebook containing the entire implementation.
- sms_spam.csv : Dataset containing SMS messages and their corresponding labels.

## Machine Learning Model

In spam_sms_detection.ipynb, we performed the following steps:

- Data Cleaning & Preprocessing: Converted text to lowercase, removed punctuation & stopwords, and applied stemming.
- Feature Engineering: Used TF-IDF vectorization to convert text into numerical features.
- Model Training & Evaluation: Trained Multinomial Naive Bayes for classification.
- Results: 96.77% accuracy, high precision (99%), lower recall (77%).
- Performance Metrics: Precision, Recall, F1-score, Confusion Matrix for model evaluation.

## Technologies Used
- Python 3
- Colab Notebook
- Pandas, NLTK (Natural Language Processing Toolkit)

## Usage
To reproduce the model training and evaluation:
- open Spam_SMS_Detection.ipynb in Colab Notebook and execute the cells sequentially.
