# Spam Message Detection Using NLP and Machine Learning

## 📌 Project Overview

This project is a Natural Language Processing (NLP) based application that detects whether an SMS message is **Spam** or **Ham (Not Spam)**.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text messages into numerical features and **Multinomial Naive Bayes** for classification.

The project is implemented in **Python using Google Colab**.

---

## 🎯 Objectives

- To understand the basics of Natural Language Processing.
- To preprocess and clean SMS text data.
- To convert text into numerical features using TF-IDF.
- To train a machine learning model for spam detection.
- To evaluate the performance of the classification model.
- To predict whether a new SMS message is Spam or Ham.

---

## 🧠 Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Regular Expressions (re)

---

## 📂 Dataset

### SMS Spam Collection Dataset

The project uses the **SMS Spam Collection Dataset**.

- **Dataset Size:** 5,574 SMS messages
- **Source:** UCI Machine Learning Repository
- **Target Classes:**
  - `ham` – Normal message
  - `spam` – Unwanted/spam message

The dataset contains SMS messages along with their corresponding labels.

---

## 🔄 Methodology

The project follows these steps:

```text
SMS Dataset
     ↓
Text Preprocessing
     ↓
TF-IDF Feature Extraction
     ↓
Train-Test Split
     ↓
Multinomial Naive Bayes
     ↓
Prediction
     ↓
Model Evaluation
