Abusive Language Detection in Kannada and Tulu Code-Mixed Text
📌 Project Overview

This project focuses on detecting abusive and offensive language in Kannada and Tulu code-mixed social media comments. Low-resource Dravidian languages present unique challenges such as data scarcity, code-mixing, spelling variations, and transliteration issues.

We implemented and evaluated ensemble machine learning models using TF–IDF feature extraction, combining Linear SVM, Logistic Regression, and Naive Bayes classifiers to improve robustness and performance.

Kannada: Ensemble model (SVM + LR + NB)

Tulu: Ensemble model (SVM + LR + NB)

🚀 Features

Handles low-resource languages (Kannada & Tulu).

Text preprocessing with cleaning, normalization, and TF–IDF vectorization.

Uses ensemble learning for improved prediction accuracy.

Outputs predictions on test data with ID and Predicted Label.

Lightweight models → Easy deployment on limited hardware.

🛠️ Tech Stack

Language: Python

Libraries:

pandas, numpy

scikit-learn

re (for regex-based cleaning)

Platform: Google Colab

📂 Dataset

The datasets consist of social media comments annotated for abusive and non-abusive content.

Kannada: kannada_offensive_train.csv, kannada_offensive_dev.csv, kannada_offensive_test_without_labels.csv

Tulu: Tulu_offensive_train.csv, Tulu_offensive_dev.csv, Tulu_test_data_without_label.csv

⚙️ Methodology

Data Cleaning: Remove URLs, special characters, extra spaces, and normalize text.

Feature Extraction: Apply TF–IDF vectorizer with unigrams & bigrams.

Model Training:

Train SVM, Logistic Regression, and Naive Bayes classifiers.

Combine them using Voting Classifier (hard voting).

Evaluation: Measure accuracy on dev sets.

Prediction: Generate predictions on test data and save output as CSV.

📊 Result Summary

Both Kannada and Tulu achieved consistent performance using the ensemble approach, outperforming individual classifiers.

The models proved effective in handling sparse, high-dimensional text features, but remain limited in deep contextual understanding.

⚠️ Limitatiions

Data Imbalance: Uneven class distributions may bias predictions.

Code-Mixed Complexity: Spelling variations, grammar inconsistencies, and transliteration errors reduce accuracy.

Generalization: Models may not adapt well to unseen datasets or different social platforms.

TF–IDF Limitation: Captures surface-level features only, ignoring semantic context.

Contextual Understanding: Sarcasm and implicit abuse remain difficult for feature-based models.

🔮 Future Work

Apply the transformer-based models (BERT, RoBERTa, XLM-Roberta).

Explore contextual embeddings (FastText, Word2Vec).

Use data augmentation to balance underrepresented classes.

Investigate domain-adaptive pretraining for Kannada & Tulu.

▶️ How to Run
1. Install dependencies
pip install pandas scikit-learn

2. Run Kannada Model (Ensemble)
!python kannada_model.py

3. Run Tulu Model (Ensemble)
!python tulu_model.py

4. Output

Predictions saved as:

kannada_test_predictions_with_id.csv

Tulu_test_predictions_with_id.csv
