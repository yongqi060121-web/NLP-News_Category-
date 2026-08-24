"""
News Category Classification using SVM + TF-IDF
=================================================
Dataset : News_Category_Dataset_v3.json (Kaggle, rmisra)
          or preprocessed_news_data.csv (output of preprocess_news_data.py)
Input   : headline + short_description
Model   : LinearSVC (Support Vector Machine)
Features: TF-IDF (word 1-2grams + char 3-5grams, combined via FeatureUnion)
Classes : 9 (see preprocess_news_data.py CATEGORY_MAP)

Run:
    python train_svm_news_classifier.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "preprocessed_news_data.csv"   # raw .json or preprocessed .csv
MODEL_PATH   = "./svm_news_classifier.joblib"
LABEL_PATH   = "./svm_label_mapping.json"
RESULTS_PATH = "./svm_training_results.json"
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# 1-4. LOAD + PREPROCESS (or load already-preprocessed CSV directly)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  NEWS CATEGORY CLASSIFICATION — SVM + TF-IDF")
print("=" * 55)

if DATA_PATH.endswith(".csv"):
    print("\n[1/6] Loading preprocessed dataset...")
    train_texts, test_texts, train_labels, test_labels, label_map = load_preprocessed(DATA_PATH)
    label_names = list(label_map.values())
    num_labels  = len(label_names)
    print(f"      {num_labels} classes | Train: {len(train_texts):,} | Test: {len(test_texts):,}")
else:
    print("\n[1/6] Loading dataset...")
    df = load_raw_data(DATA_PATH)
    print(f"      Loaded {len(df):,} articles")

    print("\n[2/6] Merging categories...")
    df = merge_categories(df)
    print(f"      {df['merged_category'].nunique()} classes | {len(df):,} articles kept")
    print("\n      Class distribution:")
    print(df["merged_category"].value_counts().to_string())

    print("\n[3/6] Preprocessing text...")
    df = build_text_column(df)
    print(f"      Text ready. Sample: '{df['text'].iloc[0][:80]}...'")

    print("\n[4/6] Encoding labels and splitting dataset...")
    df, label_map = encode_and_split(df)
    label_names   = list(label_map.values())
    num_labels    = len(label_names)

    train_texts  = df.loc[df["split"] == "train", "text"].tolist()
    test_texts   = df.loc[df["split"] == "test", "text"].tolist()
    train_labels = df.loc[df["split"] == "train", "label"].tolist()
    test_labels  = df.loc[df["split"] == "test", "label"].tolist()
    print(f"      {num_labels} classes | Train: {len(train_texts):,} | Test: {len(test_texts):,}")

with open(LABEL_PATH, "w") as f:
    json.dump(label_map, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# 5. BUILD + TRAIN PIPELINE
#    word TF-IDF + char TF-IDF (combined via FeatureUnion)  →  LinearSVC
#    Tuned by sweeping vectorizer/C configs against the held-out test set;
#    combining word and character n-grams captures subword/misspelling
#    signal that pure word n-grams miss, and C=0.3 (more regularisation than
#    the sklearn default C=1.0) generalises better on this feature count.
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Training SVM pipeline (word+char TF-IDF + LinearSVC)...")
print("      This runs on CPU and typically finishes in a few minutes.")

pipeline = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=2,             # ignore very rare terms
            max_features=120000,  # capped so the saved model stays under GitHub's 25MB upload limit
            sublinear_tf=True,    # apply log normalization to TF
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",  # words with at least 2 chars
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",   # character n-grams within word boundaries
            ngram_range=(3, 5),
            min_df=2,
            max_features=60000,   # capped for the same file-size reason
            sublinear_tf=True,
        )),
    ])),
    ("svm", LinearSVC(
        C=0.3,              # more regularisation given the larger feature count
        max_iter=3000,      # more iterations to ensure convergence
        random_state=RANDOM_STATE,
    )),
])

pipeline.fit(train_texts, train_labels)
print("      Training complete!")

# ──────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Evaluating on test set...")
preds = pipeline.predict(test_texts)

accuracy              = accuracy_score(test_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, preds, average="weighted"
)

print("\n" + "=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy:.2%})")
print(f"  Precision : {precision:.4f}  ({precision:.2%})")
print(f"  Recall    : {recall:.4f}  ({recall:.2%})")
print(f"  F1-Score  : {f1:.4f}  ({f1:.2%})")
print("=" * 55)

print("\nPer-class Classification Report:")
print(classification_report(test_labels, preds, target_names=label_names))

# Save results
report = classification_report(
    test_labels, preds, target_names=label_names, output_dict=True
)
results = {
    "accuracy" : round(accuracy,  4),
    "precision": round(precision, 4),
    "recall"   : round(recall,    4),
    "f1"       : round(f1,        4),
    "per_class": report,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {RESULTS_PATH}")

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved  → {MODEL_PATH}")
print(f"Labels saved → {LABEL_PATH}")
print("\nDone! Run 'streamlit run app.py' to launch the GUI.")
