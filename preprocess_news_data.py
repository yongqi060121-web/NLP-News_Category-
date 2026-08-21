"""
News Dataset Preprocessing
===========================
Dataset : News_Category_Dataset_v3.json (Kaggle, rmisra)
Purpose : Standalone preprocessing pipeline for the raw dataset. Produces a
          cleaned, ready-to-train CSV so the SVM (train_svm_news_classifier.py)
          and Logistic Regression (train_lr_news_classifier.py) scripts don't
          need to repeat these steps or re-read the raw JSON.

Pipeline:
  1. Load raw JSON            - newline-delimited JSON, one article per line
  2. Select + drop missing    - keep headline, short_description, category
  3. Merge categories         - 41 raw labels -> 19 classes (drop "FIFTY")
  4. Build text field         - headline + short_description combined
  5. Clean text                - lowercase, strip punctuation, normalise spaces
  6. Encode labels + split    - LabelEncoder + stratified 80/20 train/test

Run:
    python preprocess_news_data.py
"""

import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "News_Category_Dataset_v3.json"
OUTPUT_CSV        = "./preprocessed_news_data.csv"
LABEL_MAP_PATH    = "./category_label_mapping.json"
REPORT_PATH        = "./preprocessing_report.json"
TEST_SIZE          = 0.2
RANDOM_STATE        = 42

# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY MERGE MAP  (41 original -> 19 classes)
# Same mapping used by train_svm_news_classifier.py / train_lr_news_classifier.py
# so the outputs of this script are drop-in compatible with both.
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "POLITICS":"POLITICS","WELLNESS":"WELLNESS","HEALTHY LIVING":"WELLNESS",
    "ENTERTAINMENT":"ENTERTAINMENT","ARTS":"ENTERTAINMENT",
    "ARTS & CULTURE":"ENTERTAINMENT","CULTURE & ARTS":"ENTERTAINMENT",
    "TRAVEL":"TRAVEL","STYLE & BEAUTY":"STYLE & BEAUTY","STYLE":"STYLE & BEAUTY",
    "PARENTING":"PARENTING","PARENTS":"PARENTING",
    "QUEER VOICES":"VOICES","BLACK VOICES":"VOICES",
    "WOMEN":"VOICES","LATINO VOICES":"VOICES",
    "FOOD & DRINK":"FOOD & DRINK","TASTE":"FOOD & DRINK",
    "BUSINESS":"BUSINESS","MONEY":"BUSINESS","COMEDY":"COMEDY","SPORTS":"SPORTS",
    "HOME & LIVING":"HOME & LIVING",
    "THE WORLDPOST":"WORLD NEWS","WORLDPOST":"WORLD NEWS","WORLD NEWS":"WORLD NEWS",
    "WEDDINGS":"WEDDINGS & DIVORCE","DIVORCE":"WEDDINGS & DIVORCE",
    "CRIME":"U.S. NEWS","MEDIA":"U.S. NEWS",
    "U.S. NEWS":"U.S. NEWS","WEIRD NEWS":"U.S. NEWS",
    "IMPACT":"IMPACT","GOOD NEWS":"IMPACT","RELIGION":"IMPACT",
    "GREEN":"ENVIRONMENT","ENVIRONMENT":"ENVIRONMENT",
    "SCIENCE":"SCIENCE & TECH","TECH":"SCIENCE & TECH",
    "COLLEGE":"EDUCATION","EDUCATION":"EDUCATION",
    "FIFTY":None,
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the newline-delimited JSON dataset and keep only the needed columns."""
    df = pd.read_json(path, lines=True)
    df = df[["headline", "short_description", "category"]].dropna(
        subset=["headline", "category"]
    )
    return df


def merge_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map the 41 raw categories onto 19 merged classes, dropping unmapped rows."""
    df = df.copy()
    df["merged_category"] = df["category"].map(CATEGORY_MAP)
    df = df[df["merged_category"].notna()].reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for single strings
    (e.g. at inference time); build_text_column() below does the same thing
    vectorised over a whole DataFrame column for speed."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Combine headline + short_description into a single cleaned text field."""
    df = df.copy()
    df["short_description"] = df["short_description"].fillna("")
    df["text"] = (
        df["headline"].str.strip() + " " + df["short_description"].str.strip()
    ).str.strip()
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"[^a-z0-9\s]", " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def encode_and_split(df: pd.DataFrame):
    """Encode merged_category as integer labels and do a stratified train/test split."""
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df["merged_category"])
    label_names = list(le.classes_)
    label_map = {str(i): name for i, name in enumerate(label_names)}

    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    return df, label_map


def load_preprocessed(csv_path: str, label_map_path: str = LABEL_MAP_PATH):
    """Load the already-preprocessed CSV (produced by main() below) directly,
    skipping load_raw_data/merge_categories/build_text_column/encode_and_split.
    Returns (train_texts, test_texts, train_labels, test_labels, label_map)."""
    df = pd.read_csv(csv_path)
    with open(label_map_path) as f:
        label_map = json.load(f)

    train_texts  = df.loc[df["split"] == "train", "text"].tolist()
    test_texts   = df.loc[df["split"] == "test", "text"].tolist()
    train_labels = df.loc[df["split"] == "train", "label"].tolist()
    test_labels  = df.loc[df["split"] == "test", "label"].tolist()

    return train_texts, test_texts, train_labels, test_labels, label_map


def main():
    print("=" * 55)
    print("  NEWS DATASET PREPROCESSING")
    print("=" * 55)

    print("\n[1/5] Loading raw dataset...")
    df = load_raw_data(DATA_PATH)
    raw_count = len(df)
    print(f"      Loaded {raw_count:,} articles with headline + category")

    print("\n[2/5] Merging 41 categories into 19 classes...")
    df = merge_categories(df)
    merged_count = len(df)
    print(f"      {df['merged_category'].nunique()} classes | "
          f"{merged_count:,} articles kept "
          f"({raw_count - merged_count:,} dropped, e.g. 'FIFTY')")
    print("\n      Class distribution:")
    print(df["merged_category"].value_counts().to_string())

    print("\n[3/5] Building and cleaning text (headline + short_description)...")
    df = build_text_column(df)
    clean_count = len(df)
    print(f"      {clean_count:,} articles remain after removing empty text")
    print(f"      Sample: '{df['text'].iloc[0][:80]}...'")

    print("\n[4/5] Encoding labels and creating stratified train/test split...")
    df, label_map = encode_and_split(df)
    train_count = (df["split"] == "train").sum()
    test_count = (df["split"] == "test").sum()
    print(f"      Train: {train_count:,} | Test: {test_count:,} "
          f"({TEST_SIZE:.0%} held out, stratified by class)")

    print("\n[5/5] Saving preprocessed outputs...")
    out_df = df[["text", "merged_category", "label", "split"]]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"      Preprocessed data -> {OUTPUT_CSV}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"      Label mapping      -> {LABEL_MAP_PATH}")

    report = {
        "raw_articles": raw_count,
        "after_category_merge": merged_count,
        "after_text_cleaning": clean_count,
        "num_classes": len(label_map),
        "train_size": int(train_count),
        "test_size": int(test_count),
        "test_split_fraction": TEST_SIZE,
        "class_distribution": df["merged_category"].value_counts().to_dict(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"      Preprocessing report -> {REPORT_PATH}")

    print("\nDone! Both train_svm_news_classifier.py and train_lr_news_classifier.py")
    print("can load preprocessed_news_data.csv directly instead of the raw JSON.")


if __name__ == "__main__":
    main()
