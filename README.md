# Spam Message Detection Using NLP and Machine Learning

## 1. Project Overview

Spam messages are unwanted messages that may contain advertisements, fraudulent offers, misleading information, or suspicious links. Manually identifying spam messages can be difficult when the number of messages is large.

This project develops a **Spam Message Detection System** using **Natural Language Processing (NLP)** and **Machine Learning**. The system analyzes the text of an SMS message and classifies it into one of two categories:

- **Ham** – Normal or legitimate message
- **Spam** – Unwanted or suspicious message

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and **Multinomial Naive Bayes** as the machine learning classification algorithm.

The project is implemented using **Python in Google Colab**.

---

## 2. Problem Statement

The objective of this project is to automatically identify whether an SMS message is spam or a normal message.

Since SMS messages are unstructured text data, Natural Language Processing techniques are required to process the text and convert it into a format that can be understood by a machine learning algorithm.

The system takes an SMS message as input and predicts whether the message is **Spam** or **Ham**.

---

## 3. Objectives

The main objectives of this project are:

1. To understand the application of Natural Language Processing in text classification.
2. To preprocess and clean SMS text data.
3. To convert text messages into numerical features using TF-IDF.
4. To train a machine learning model for spam message classification.
5. To predict whether new SMS messages are Spam or Ham.
6. To evaluate the performance of the trained model using standard classification metrics.
7. To understand the limitations and possible improvements of the system.

---

## 4. Technologies Used

The following technologies and Python libraries are used in this project:

| Technology / Library | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development and execution platform |
| Pandas | Data loading and data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Regular Expressions (`re`) | Text preprocessing |

---

## 5. Dataset

### SMS Spam Collection Dataset

The project uses the **SMS Spam Collection Dataset**.

**Dataset Source:** UCI Machine Learning Repository

**Dataset Size:** 5,574 SMS messages

The dataset contains SMS messages labelled as either `ham` or `spam`.

### Dataset Attributes

| Attribute | Description |
|---|---|
| `label` | Class of the SMS message |
| `message` | Actual SMS text |

### Target Classes

| Class | Meaning |
|---|---|
| Ham | Normal / legitimate SMS |
| Spam | Unwanted / spam SMS |

The dataset is stored in the `dataset` folder of this repository.

---

## 6. Natural Language Processing

Natural Language Processing is used in this project because the input data consists of human language in the form of SMS messages.

The main NLP steps used are:

1. Text preprocessing
2. Text normalization
3. TF-IDF feature extraction
4. Machine learning classification

These steps allow the machine learning model to learn patterns commonly found in spam and normal messages.

---

## 7. Methodology

The overall workflow of the project is:

```text
SMS Dataset
     |
     v
Data Loading
     |
     v
Text Preprocessing
     |
     v
Train-Test Split
     |
     v
TF-IDF Feature Extraction
     |
     v
Multinomial Naive Bayes
     |
     v
Prediction
     |
     v
Model Evaluation
8. Text Preprocessing

Before training the machine learning model, the SMS messages are cleaned.

The following preprocessing operations are performed:

Convert text to lowercase.
Remove special characters and numbers.
Remove extra spaces.
Prepare the cleaned text for feature extraction.

For example:

Original:
"Congratulations! You have WON $1000!!!"

After preprocessing:
"congratulations you have won"

The cleaned messages are then used for TF-IDF feature extraction.

9. TF-IDF Feature Extraction

Machine learning algorithms cannot directly understand raw text. Therefore, the SMS messages need to be converted into numerical features.

This project uses TF-IDF (Term Frequency-Inverse Document Frequency).

TF-IDF assigns numerical importance to words based on how frequently they occur in a message and how common or uncommon they are across the dataset.

The project uses:

TfidfVectorizer(stop_words="english", max_features=5000)

The TF-IDF vectorizer is fitted using the training data and then used to transform both training and testing messages.

10. Train-Test Split

The dataset is divided into two parts:

80% Training Data
20% Testing Data

The training data is used to train the machine learning model, while the testing data is used to evaluate its performance on unseen messages.

The dataset is split using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
11. Machine Learning Model
Multinomial Naive Bayes

The classification algorithm used in this project is Multinomial Naive Bayes.

Multinomial Naive Bayes is commonly used for text classification because it works well with word-based features such as TF-IDF.

The model is trained using the TF-IDF feature vectors obtained from the SMS messages.

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

After training, the model predicts whether unseen messages belong to the Spam or Ham class.

12. Model Evaluation

The model is evaluated using the following metrics:

Accuracy

Accuracy represents the percentage of correctly classified messages out of all test messages.

Precision

Precision measures how many messages predicted as Spam were actually Spam.

Recall

Recall measures how many actual Spam messages were correctly detected.

F1-Score

F1-Score is the harmonic mean of Precision and Recall.

Confusion Matrix

A confusion matrix shows the number of correct and incorrect predictions for each class.

13. Results

The trained Multinomial Naive Bayes model achieved the following results on the test dataset:

Evaluation Metric	Result
Accuracy	96.59%
Precision	100.00%
Recall	74.50%
F1-Score	85.38%
Result Analysis

The model achieved an accuracy of 96.59%, which indicates that most of the test SMS messages were classified correctly.

The precision score of 100% indicates that the messages predicted as Spam were correctly identified as Spam in the evaluated test set.

The recall score of 74.50% indicates that some actual Spam messages were not detected by the model.

The F1-Score of 85.38% represents the balance between precision and recall.

14. Sample Prediction

The trained model can also classify new SMS messages entered by the user.

Example:

Input:
Congratulations! You have won a free prize. Claim now!

Output:

Prediction: SPAM

Another example:

Input:
Hey, are you coming to college today?

Output:

Prediction: HAM
15. Project Screenshots

Screenshots demonstrating the project execution and results are available in the screenshots folder.

The screenshots include:

Dataset preview
Dataset distribution
Model performance metrics
Confusion matrix
New message prediction
16. Project Structure
Spam-Message-Detection-NLP/
│
├── dataset/
│   └── spam.tsv
│
├── screenshots/
│   ├── 01_dataset.png
│   ├── 02_dataset_distribution.png
│   ├── 03_model_performance.png
│   ├── 04_confusion_matrix.png
│   └── 05_new_message_prediction.png
│
├── report/
│   └── REPORT Spam Message Detection.pdf
│
├── README.md
├── requirements.txt
├── source_code.py
└── Spam_Message_Detection_NLP.ipynb
17. How to Run the Project
Option 1: Google Colab
Open Spam_Message_Detection_NLP.ipynb.
Open the notebook in Google Colab.
Make sure the dataset is available in the required location.
Run the notebook cells from top to bottom.
The dataset will be loaded and preprocessed.
TF-IDF features will be generated.
The Multinomial Naive Bayes model will be trained.
The model performance will be displayed.
Enter a new SMS message to obtain a prediction.
Option 2: Run Using Python

Install the required libraries:

pip install -r requirements.txt

Then run:

python source_code.py

The program will train the model and allow the user to test new SMS messages.

18. Example Output

The system produces output in the following format:

Enter a message: Congratulations! You won a free prize!

Prediction: SPAM

For a normal message:

Enter a message: Are you coming to class today?

Prediction: HAM
19. Limitations

Although the model provides good results, it has some limitations:

The model may not correctly classify unusual or very short messages.
The dataset may not represent all types of modern spam messages.
The dataset mainly contains English SMS messages.
Spelling variations and slang may affect classification.
New spam patterns may not be recognized if they were not present in the training data.
The model uses traditional machine learning and does not understand the deeper semantic meaning of text.
20. Future Scope

The project can be improved in the future by:

Using a larger and more recent SMS dataset.
Including messages in multiple languages.
Applying advanced NLP preprocessing techniques.
Comparing Multinomial Naive Bayes with other machine learning algorithms such as Logistic Regression and Support Vector Machine.
Using deep learning models such as LSTM.
Exploring Transformer-based NLP models.
Developing a web application for real-time spam detection.
Adding URL and sender-related features for improved spam detection.
21. Conclusion

This project demonstrates the use of Natural Language Processing and Machine Learning for SMS spam detection.

The SMS messages are cleaned using text preprocessing techniques and converted into numerical features using TF-IDF. These features are then classified using the Multinomial Naive Bayes algorithm.

The model achieved an accuracy of 96.59%, demonstrating that traditional NLP and machine learning techniques can be effective for SMS spam classification.

The project also provides an understanding of the complete NLP workflow, including dataset preparation, preprocessing, feature extraction, model training, prediction, and evaluation.

22. Files in This Repository
File / Folder	Description
dataset/	Contains the SMS dataset
screenshots/	Contains project execution and result screenshots
report/	Contains the project report PDF
source_code.py	Python source code
Spam_Message_Detection_NLP.ipynb	Google Colab/Jupyter Notebook
requirements.txt	Required Python libraries
README.md	Project documentation
23. Project Information

Project Title: Spam Message Detection Using NLP and Machine Learning

Project Type: Mini NLP Project

Programming Language: Python

Platform: Google Colab

NLP Technique: TF-IDF

Machine Learning Algorithm: Multinomial Naive Bayes

Dataset: SMS Spam Collection Dataset

Dataset Size: 5,574 messages

Classification: Binary Classification

Classes: Ham and Spam

Accuracy: 96.59%

24. Author

Joseph Toby

This project was developed as part of an academic Mini NLP Project.


### ⚠️ Before you commit

One important correction from your previous screenshot: your uploaded screenshots were named **`.png.png`**. Rename them if you can so they exactly match the README:

```text
01_dataset.png
02_dataset_distribution.png
03_model_performance.png
04_confusion_matrix.png
05_new_message_prediction.png
