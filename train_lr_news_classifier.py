"""
News Category Classification using Logistic Regression + Bag-of-Words (BoW)
============================================================================
Dataset : News_Category_Dataset_v3.json (Kaggle, rmisra)
          or preprocessed_news_data.csv (output of preprocess_news_data.py)
Input   : headline + short_description
Model   : Logistic Regression
Features: Bag-of-Words (CountVectorizer)

BoW vs TF-IDF:
- BoW counts how many times each word appears in a document (raw counts)
- TF-IDF additionally weights words by how rare they are across all documents
- BoW is simpler and faster; TF-IDF usually gives slightly better accuracy

Run:
    python train_lr_news_classifier.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import joblib

from preprocess_news_data import (
    load_raw_data,
    merge_categories,
    build_text_column,
    encode_and_split,
    load_preprocessed,
)

DATA_PATH = "preprocessed_news_data.csv"   # raw .json or preprocessed .csv
MODEL_PATH = "./lr_bow_news_classifier.joblib"
LABEL_PATH = "./lr_bow_label_mapping.json"
RESULTS_PATH = "./lr_bow_training_results.json"
RANDOM_STATE = 42


print("=" * 65)
print("NEWS CATEGORY CLASSIFICATION — LOGISTIC REGRESSION + BoW")
print("=" * 65)

if DATA_PATH.endswith(".csv"):
    print("\n[1/6] Loading preprocessed dataset...")
    train_texts, test_texts, train_labels, test_labels, label_map = load_preprocessed(DATA_PATH)
    label_names = list(label_map.values())
    num_labels  = len(label_names)
    print(f"{num_labels} classes | Train: {len(train_texts):,} | Test {len(test_texts):,}")
else:
    print("\n[1/6] Loading dataset...")
    df = load_raw_data(DATA_PATH)
    print(f"Loaded {len(df):,} articles")

    print("\n[2/6] Merging categories... (41 original → 19 classes)")
    df = merge_categories(df)
    print(f"Loaded {len(df):,} articles after merging categories")
    print("\nClass distribution after merging categories:")
    print(df["merged_category"].value_counts().to_string())

    print("\n[3/6] Preprocessing text data...")
    df = build_text_column(df)
    print(f"Text ready. Sample: '{df['text'].iloc[0][:80]}...'")

    print("\n[4/6] Encoding labels and splitting dataset...")
    df, label_map = encode_and_split(df)
    label_names   = list(label_map.values())
    num_labels    = len(label_names)

    train_texts  = df.loc[df["split"] == "train", "text"].tolist()
    test_texts   = df.loc[df["split"] == "test", "text"].tolist()
    train_labels = df.loc[df["split"] == "train", "label"].tolist()
    test_labels  = df.loc[df["split"] == "test", "label"].tolist()
    print(f"{num_labels} classes | Train: {len(train_texts):,} | Test {len(test_texts):,}")

with open(LABEL_PATH, "w") as f:
    json.dump(label_map, f, indent=2)


print("\n[5/6] Building pipeline: Training BoW + LogisticRegression pipeline...")
print(" This typically finishes in 2-8 minutes on CPU.")

pipeline = Pipeline([
    ("bow", CountVectorizer(
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        binary=False,
    )),
    ("lr", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

pipeline.fit(train_texts, train_labels)
print(" Training complete. Saving model...")

print("\n[6/6] Evaluating model on test set...")
preds = pipeline.predict(test_texts)
accuracy = accuracy_score(test_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, preds, average="weighted"
)

print("\n" + "=" * 65)
print("EVALUATION RESULTS  (Logistic Regression + Bag-of-Words)")
print("=" * 65)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy:.2%})")
print(f"  Precision : {precision:.4f}  ({precision:.2%})")
print(f"  Recall    : {recall:.4f}  ({recall:.2%})")
print(f"  F1-Score  : {f1:.4f}  ({f1:.2%})")
print("=" * 65)

print("\nPer-class Classification Report:")
print(classification_report(test_labels, preds, target_names=label_names))

report = classification_report(
    test_labels, preds, target_names=label_names, output_dict=True
)

results = {
    "model" : "Logistic Regression + Bag-of-Words",
    "accuracy" : round(accuracy, 4),
    "precision" : round(precision, 4),
    "recall" : round(recall, 4),
    "f1" : round(f1, 4),
    "per_class" : report,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTraining results saved to {RESULTS_PATH}")

joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved -> {MODEL_PATH}")
print(f"Labels saved -> {LABEL_PATH}")
print(f"\nDone! You can now use the trained model to classify news articles.")