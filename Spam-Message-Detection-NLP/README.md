# Spam Message Detection Using NLP and Machine Learning

## Project Overview
This mini NLP project classifies SMS messages as **Spam** or **Ham (Not Spam)**.

## Techniques Used
- Text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes
- Accuracy, Precision, Recall and F1-Score
- Confusion Matrix

## Methodology
SMS Dataset → Text Preprocessing → TF-IDF → Train-Test Split → Naive Bayes → Prediction → Evaluation

## Platform
Google Colab

## Project Structure
```text
Spam-Message-Detection-NLP/
├── README.md
├── source_code.py
├── Spam_Message_Detection_NLP.ipynb
├── dataset/
├── screenshots/
├── report/
└── requirements.txt
```

## Dataset
SMS Spam Collection dataset, with messages labelled as `ham` or `spam`.

## How to Run
1. Open the `.ipynb` file in Google Colab.
2. Run cells from top to bottom.
3. Check the evaluation results.
4. Enter a new SMS message to test the classifier.

## Limitations
- Unseen messages may be incorrectly classified.
- Sarcasm and unusual language can be difficult to detect.
- New spam patterns may not always be identified.

## Future Scope
- Larger dataset
- Comparison of multiple ML models
- Advanced NLP models such as BERT
- Multilingual support
- Web/mobile deployment
