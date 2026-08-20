"""
News Category Classifier — Combined App
========================================
Models:
  1. SVM + TF-IDF       (LinearSVC + TfidfVectorizer)
  2. LR  + Bag-of-Words (LogisticRegression + CountVectorizer)

Run with:  streamlit run app_combined.py
Requires:  pip install streamlit plotly pandas scikit-learn numpy joblib
"""

import os, json, threading, time, re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.banner {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0d1b2a 100%);
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 12px 12px;
}
.banner h1 { font-family:'Playfair Display',serif; font-size:2.4rem; color:#f1faee; margin:0 0 0.3rem 0; }
.banner p  { color:#a8b2c1; font-size:0.95rem; margin:0; font-weight:300; }

.svm-tag  { display:inline-block; background:#2196f3; color:#fff; font-size:0.7rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; padding:2px 10px; border-radius:2px; margin-bottom:0.6rem; }
.bow-tag  { display:inline-block; background:#f39c12; color:#fff; font-size:0.7rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; padding:2px 10px; border-radius:2px; margin-bottom:0.6rem; }

/* Metric cards */
.metric-svm { background:#1a1f2e; border:1px solid #2196f3; border-radius:10px; padding:1.1rem 1.2rem; text-align:center; }
.metric-svm .val { font-size:1.8rem; font-weight:700; color:#2196f3; line-height:1; }
.metric-bow { background:#1a1f2e; border:1px solid #f39c12; border-radius:10px; padding:1.1rem 1.2rem; text-align:center; }
.metric-bow .val { font-size:1.8rem; font-weight:700; color:#f39c12; line-height:1; }
.metric-lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; color:#7a8499; margin-top:0.3rem; }

/* Prediction boxes */
.pred-svm { background:linear-gradient(135deg,#1a1f2e,#0f1117); border:1.5px solid #2196f3; border-radius:12px; padding:1.2rem 1.5rem; text-align:center; }
.pred-bow { background:linear-gradient(135deg,#1a1f2e,#0f1117); border:1.5px solid #f39c12; border-radius:12px; padding:1.2rem 1.5rem; text-align:center; }
.pred-cat  { font-family:'Playfair Display',serif; font-size:1.8rem; color:#f1faee; margin:0 0 0.2rem 0; }
.pred-conf { color:#a8b2c1; font-size:0.88rem; }

/* Compare bar */
.compare-bar { background:#1a1f2e; border:1px solid #2a3040; border-radius:10px; padding:1rem 1.4rem; margin:1rem 0; }

.log-box { background:#0f1117; border:1px solid #2a3040; border-radius:8px; padding:1rem; font-family:monospace; font-size:0.8rem; color:#a8d8a8; max-height:280px; overflow-y:auto; }
.pill-green  { display:inline-block; background:#145a32; color:#58d68d; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.pill-red    { display:inline-block; background:#641e16; color:#e74c3c; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.pill-yellow { display:inline-block; background:#4a3b00; color:#f4d03f; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
[data-testid="stSidebar"] { background:#0f1117; border-right:1px solid #2a3040; }
[data-testid="stSidebar"] * { color:#c5cdd8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
SVM_MODEL    = "./svm_news_classifier.joblib"
SVM_LABELS   = "./svm_label_mapping.json"
SVM_RESULTS  = "./svm_training_results.json"

BOW_MODEL    = "./lr_bow_news_classifier.joblib"
BOW_LABELS   = "./lr_bow_label_mapping.json"
BOW_RESULTS  = "./lr_bow_training_results.json"

DEFAULT_DATA = "News_Category_Dataset_v3.json"
RANDOM_STATE = 42

CATEGORY_MAP = {
    "POLITICS":"POLITICS","WELLNESS":"WELLNESS","HEALTHY LIVING":"WELLNESS",
    "ENTERTAINMENT":"ENTERTAINMENT","ARTS":"ENTERTAINMENT",
    "ARTS & CULTURE":"ENTERTAINMENT","CULTURE & ARTS":"ENTERTAINMENT",
    "TRAVEL":"TRAVEL","STYLE & BEAUTY":"STYLE & BEAUTY","STYLE":"STYLE & BEAUTY",
    "PARENTING":"PARENTING","PARENTS":"PARENTING",
    "QUEER VOICES":"VOICES","BLACK VOICES":"VOICES","WOMEN":"VOICES","LATINO VOICES":"VOICES",
    "FOOD & DRINK":"FOOD & DRINK","TASTE":"FOOD & DRINK",
    "BUSINESS":"BUSINESS","MONEY":"BUSINESS","COMEDY":"COMEDY","SPORTS":"SPORTS",
    "HOME & LIVING":"HOME & LIVING",
    "THE WORLDPOST":"WORLD NEWS","WORLDPOST":"WORLD NEWS","WORLD NEWS":"WORLD NEWS",
    "WEDDINGS":"WEDDINGS & DIVORCE","DIVORCE":"WEDDINGS & DIVORCE",
    "CRIME":"U.S. NEWS","MEDIA":"U.S. NEWS","U.S. NEWS":"U.S. NEWS","WEIRD NEWS":"U.S. NEWS",
    "IMPACT":"IMPACT","GOOD NEWS":"IMPACT","RELIGION":"IMPACT",
    "GREEN":"ENVIRONMENT","ENVIRONMENT":"ENVIRONMENT",
    "SCIENCE":"SCIENCE & TECH","TECH":"SCIENCE & TECH",
    "COLLEGE":"EDUCATION","EDUCATION":"EDUCATION","FIFTY":None,
}

CATEGORY_ICONS = {
    "POLITICS":"🏛️","WELLNESS":"🧘","ENTERTAINMENT":"🎬","TRAVEL":"✈️",
    "STYLE & BEAUTY":"💄","PARENTING":"👶","VOICES":"🗣️","FOOD & DRINK":"🍽️",
    "BUSINESS":"💼","COMEDY":"😄","SPORTS":"⚽","HOME & LIVING":"🏠",
    "WORLD NEWS":"🌍","WEDDINGS & DIVORCE":"💍","U.S. NEWS":"🗽",
    "IMPACT":"💡","ENVIRONMENT":"🌿","SCIENCE & TECH":"🔬","EDUCATION":"📚",
}

# ─────────────────────────────────────────────────────────────────
# Global variables for training threads
# ─────────────────────────────────────────────────────────────────
_svm_log, _svm_running, _svm_done, _svm_results = [], False, False, None
_bow_log, _bow_running, _bow_done, _bow_results = [], False, False, None

# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────
for key, val in [
    ("svm_pipeline", None), ("svm_labels", None), ("svm_results", None),
    ("bow_pipeline", None), ("bow_labels", None), ("bow_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_svm():
    pipe = joblib.load(SVM_MODEL)
    with open(SVM_LABELS) as f: lm = json.load(f)
    return pipe, lm

@st.cache_resource
def load_bow():
    pipe = joblib.load(BOW_MODEL)
    with open(BOW_LABELS) as f: lm = json.load(f)
    return pipe, lm

def svm_exists(): return os.path.exists(SVM_MODEL) and os.path.exists(SVM_LABELS)
def bow_exists(): return os.path.exists(BOW_MODEL) and os.path.exists(BOW_LABELS)

def predict_svm(headline, description):
    text    = clean_text(headline + " " + description)
    pipe    = st.session_state.svm_pipeline
    lm      = st.session_state.svm_labels
    scores  = pipe.decision_function([text])[0]
    exp     = np.exp(scores - np.max(scores))
    probs   = exp / exp.sum()
    idx     = int(np.argmax(probs))
    return lm[str(idx)], {lm[str(i)]: float(p) for i, p in enumerate(probs)}

def predict_bow(headline, description):
    text  = clean_text(headline + " " + description)
    pipe  = st.session_state.bow_pipeline
    lm    = st.session_state.bow_labels
    probs = pipe.predict_proba([text])[0]
    idx   = int(np.argmax(probs))
    return lm[str(idx)], {lm[str(i)]: float(p) for i, p in enumerate(probs)}

def get_top_bow_words(headline, description, top_n=8):
    text  = clean_text(headline + " " + description)
    vec   = st.session_state.bow_pipeline.named_steps["bow"]
    vocab = vec.vocabulary_
    counts = vec.transform([text]).toarray()[0]
    wc    = [(w, counts[i]) for w, i in vocab.items() if counts[i] > 0]
    wc.sort(key=lambda x: x[1], reverse=True)
    return wc[:top_n]

# ─────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────
def _common_load(data_path):
    """Shared data loading and preprocessing."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_json(data_path, lines=True)
    df = df[["headline","short_description","category"]].dropna(subset=["headline","category"])
    df["merged_category"] = df["category"].map(CATEGORY_MAP)
    df = df[df["merged_category"].notna()].reset_index(drop=True)
    df["short_description"] = df["short_description"].fillna("")
    df["text"] = (df["headline"].str.strip() + " " + df["short_description"].str.strip()).str.strip()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["merged_category"])
    label_names = list(le.classes_)
    label_map   = {str(i): n for i, n in enumerate(label_names)}
    train_x, test_x, train_y, test_y = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    return train_x, test_x, train_y, test_y, label_names, label_map


def train_svm_thread(data_path, c_val, max_feat, ngram_max):
    global _svm_log, _svm_running, _svm_done, _svm_results
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

    def log(m): _svm_log.append(m)
    try:
        log("📂 Loading and preprocessing dataset...")
        train_x, test_x, train_y, test_y, label_names, label_map = _common_load(data_path)
        log(f"✅ {len(train_x)+len(test_x):,} articles | {len(label_names)} classes")
        log(f"   Train: {len(train_x):,} | Test: {len(test_x):,}")

        log(f"⚙️  Building TF-IDF + LinearSVC  (C={c_val}, features={max_feat:,}, ngram=1-{ngram_max})")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,ngram_max), max_features=max_feat,
                                      sublinear_tf=True, min_df=2,
                                      strip_accents="unicode", token_pattern=r"\w{2,}")),
            ("svm",  LinearSVC(C=c_val, max_iter=2000, random_state=RANDOM_STATE)),
        ])
        log("🚀 Training SVM... (1–5 minutes)")
        pipe.fit(train_x, train_y)
        log("✅ Training complete!")

        preds = pipe.predict(test_x)
        acc   = accuracy_score(test_y, preds)
        p,r,f1,_ = precision_recall_fscore_support(test_y, preds, average="weighted")
        report = classification_report(test_y, preds, target_names=label_names, output_dict=True)
        log(f"📈 Accuracy:{acc:.4f}  F1:{f1:.4f}  Precision:{p:.4f}  Recall:{r:.4f}")

        results = {"accuracy":round(acc,4),"f1":round(f1,4),"precision":round(p,4),"recall":round(r,4),"per_class":report}
        with open(SVM_RESULTS,"w") as f: json.dump(results, f, indent=2)
        joblib.dump(pipe, SVM_MODEL)
        with open(SVM_LABELS,"w") as f: json.dump(label_map, f, indent=2)
        _svm_results = results
        _svm_done    = True
        log("🎉 SVM model saved! Click 'Load SVM Model'.")
    except Exception as e:
        import traceback; log(f"❌ {e}"); log(traceback.format_exc())
    finally:
        _svm_running = False


def train_bow_thread(data_path, c_val, max_feat, ngram_max, binary):
    global _bow_log, _bow_running, _bow_done, _bow_results
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

    def log(m): _bow_log.append(m)
    try:
        log("📂 Loading and preprocessing dataset...")
        train_x, test_x, train_y, test_y, label_names, label_map = _common_load(data_path)
        log(f"✅ {len(train_x)+len(test_x):,} articles | {len(label_names)} classes")
        log(f"   Train: {len(train_x):,} | Test: {len(test_x):,}")

        mode = "binary" if binary else "counts"
        log(f"⚙️  Building BoW + Logistic Regression  (C={c_val}, features={max_feat:,}, ngram=1-{ngram_max}, mode={mode})")
        pipe = Pipeline([
            ("bow", CountVectorizer(ngram_range=(1,ngram_max), max_features=max_feat,
                                    min_df=2, strip_accents="unicode",
                                    token_pattern=r"\w{2,}", binary=binary)),
            ("lr",  LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs",
                                    
                                    random_state=RANDOM_STATE)),
        ])
        log("🚀 Training Logistic Regression... (2–8 minutes)")
        pipe.fit(train_x, train_y)
        log("✅ Training complete!")

        preds = pipe.predict(test_x)
        acc   = accuracy_score(test_y, preds)
        p,r,f1,_ = precision_recall_fscore_support(test_y, preds, average="weighted")
        report = classification_report(test_y, preds, target_names=label_names, output_dict=True)
        log(f"📈 Accuracy:{acc:.4f}  F1:{f1:.4f}  Precision:{p:.4f}  Recall:{r:.4f}")

        results = {"accuracy":round(acc,4),"f1":round(f1,4),"precision":round(p,4),"recall":round(r,4),"per_class":report}
        with open(BOW_RESULTS,"w") as f: json.dump(results, f, indent=2)
        joblib.dump(pipe, BOW_MODEL)
        with open(BOW_LABELS,"w") as f: json.dump(label_map, f, indent=2)
        _bow_results = results
        _bow_done    = True
        log("🎉 BoW model saved! Click 'Load BoW Model'.")
    except Exception as e:
        import traceback; log(f"❌ {e}"); log(traceback.format_exc())
    finally:
        _bow_running = False

# ─────────────────────────────────────────────────────────────────
# Auto-load models if saved on disk
# ─────────────────────────────────────────────────────────────────
if svm_exists() and st.session_state.svm_pipeline is None:
    try:
        p,lm = load_svm()
        st.session_state.svm_pipeline = p
        st.session_state.svm_labels   = lm
    except: pass

if bow_exists() and st.session_state.bow_pipeline is None:
    try:
        p,lm = load_bow()
        st.session_state.bow_pipeline = p
        st.session_state.bow_labels   = lm
    except: pass

if os.path.exists(SVM_RESULTS) and st.session_state.svm_results is None:
    with open(SVM_RESULTS) as f: st.session_state.svm_results = json.load(f)

if os.path.exists(BOW_RESULTS) and st.session_state.bow_results is None:
    with open(BOW_RESULTS) as f: st.session_state.bow_results = json.load(f)

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
<h1>📰 News Category Classifier</h1>
<p>
    <span class="svm-tag">SVM · TF-IDF</span>&nbsp;
    <span class="bow-tag">Logistic Regression · Bag-of-Words</span>
    &nbsp;· 19 categories · HuffPost News Dataset
</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Status")

    st.markdown("**🔵 SVM + TF-IDF**")
    if st.session_state.svm_pipeline:
        st.markdown('<span class="pill-green">✔ Loaded</span>', unsafe_allow_html=True)
    elif svm_exists():
        st.markdown('<span class="pill-yellow">⚠ On disk</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-red">✘ Not trained</span>', unsafe_allow_html=True)

    st.markdown("**🟠 LR + Bag-of-Words**")
    if st.session_state.bow_pipeline:
        st.markdown('<span class="pill-green">✔ Loaded</span>', unsafe_allow_html=True)
    elif bow_exists():
        st.markdown('<span class="pill-yellow">⚠ On disk</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-red">✘ Not trained</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 19 Categories")
    for cat, icon in CATEGORY_ICONS.items():
        st.markdown(f"<small>{icon} {cat}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────
tab_predict, tab_train, tab_results = st.tabs([
    "🔍 Predict & Compare", "🚀 Train", "📊 Results"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & COMPARE
# ══════════════════════════════════════════════════════════════════
with tab_predict:
    neither = st.session_state.svm_pipeline is None and st.session_state.bow_pipeline is None
    if neither:
        st.info("⚠️ No models loaded yet. Go to the **Train** tab to train both models first.")
    else:
        st.markdown("#### Enter a news article")
        c1, c2 = st.columns([1,1], gap="large")
        with c1:
            headline = st.text_input("Headline", placeholder="e.g. Senate passes new climate bill…")
        with c2:
            description = st.text_area("Short Description (optional)", height=95,
                                    placeholder="Brief summary of the article…")

        st.markdown("###### Quick examples")
        ex_cols  = st.columns(4)
        examples = [
            ("💼 Business",     "Markets rally as inflation cools",   "Stocks surged after CPI data came in below expectations."),
            ("⚽ Sports",       "World Cup Final: Argentina wins",     "Argentina beat France on penalties in a thrilling final."),
            ("🔬 Science",      "NASA discovers water on Mars",        "Scientists confirm liquid water beneath the Martian south polar ice cap."),
            ("🎬 Entertainment","Oscars 2024: Best Picture winner",    "Oppenheimer swept the Oscars taking home seven awards."),
        ]
        for col, (lbl, h, d) in zip(ex_cols, examples):
            if col.button(lbl, use_container_width=True):
                headline    = h
                description = d

        st.markdown("---")
        if st.button("🔍 Classify with Both Models", type="primary"):
            if not headline.strip() and not description.strip():
                st.warning("Please enter at least a headline.")
            else:
                svm_pred = bow_pred = None
                svm_probs = bow_probs = None

                with st.spinner("Running both models…"):
                    if st.session_state.svm_pipeline:
                        svm_pred, svm_probs = predict_svm(headline, description)
                    if st.session_state.bow_pipeline:
                        bow_pred, bow_probs = predict_bow(headline, description)

                # ── Side-by-side prediction boxes ──
                left, right = st.columns(2, gap="large")

                with left:
                    if svm_pred:
                        icon = CATEGORY_ICONS.get(svm_pred, "📰")
                        conf = svm_probs[svm_pred]
                        st.markdown(f"""
                        <div class="pred-svm">
                        <div style="color:#2196f3;font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">🔵 SVM + TF-IDF</div>
                        <div class="pred-cat">{icon} {svm_pred}</div>
                        <div class="pred-conf">Confidence: {conf:.1%}</div>
                        </div>""", unsafe_allow_html=True)

                        top5_svm = sorted(svm_probs.items(), key=lambda x:x[1], reverse=True)[:8]
                        fig = go.Figure(go.Bar(
                            x=[v for _,v in top5_svm],
                            y=[f"{CATEGORY_ICONS.get(c,'📰')} {c}" for c,_ in top5_svm],
                            orientation="h",
                            marker_color=["#2196f3" if c==svm_pred else "#2a3040" for c,_ in top5_svm],
                            text=[f"{v:.1%}" for _,v in top5_svm],
                            textposition="outside", textfont=dict(color="#c5cdd8", size=11),
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c5cdd8", family="Inter"),
                            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(autorange="reversed", showgrid=False),
                            margin=dict(l=0,r=70,t=10,b=10), height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("SVM model not loaded.")

                with right:
                    if bow_pred:
                        icon = CATEGORY_ICONS.get(bow_pred, "📰")
                        conf = bow_probs[bow_pred]
                        st.markdown(f"""
                        <div class="pred-bow">
                        <div style="color:#f39c12;font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">🟠 Logistic Regression + BoW</div>
                        <div class="pred-cat">{icon} {bow_pred}</div>
                        <div class="pred-conf">Probability: {conf:.1%}</div>
                        </div>""", unsafe_allow_html=True)

                        top5_bow = sorted(bow_probs.items(), key=lambda x:x[1], reverse=True)[:8]
                        fig2 = go.Figure(go.Bar(
                            x=[v for _,v in top5_bow],
                            y=[f"{CATEGORY_ICONS.get(c,'📰')} {c}" for c,_ in top5_bow],
                            orientation="h",
                            marker_color=["#f39c12" if c==bow_pred else "#2a3040" for c,_ in top5_bow],
                            text=[f"{v:.1%}" for _,v in top5_bow],
                            textposition="outside", textfont=dict(color="#c5cdd8", size=11),
                        ))
                        fig2.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c5cdd8", family="Inter"),
                            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(autorange="reversed", showgrid=False),
                            margin=dict(l=0,r=70,t=10,b=10), height=300,
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        # BoW top words
                        st.markdown("**🔤 Top words detected (BoW)**")
                        top_words = get_top_bow_words(headline, description)
                        if top_words:
                            wdf = pd.DataFrame(top_words, columns=["Word","Count"])
                            fig3 = go.Figure(go.Bar(
                                x=wdf["Count"], y=wdf["Word"], orientation="h",
                                marker_color="#f39c12",
                                text=wdf["Count"], textposition="outside",
                                textfont=dict(color="#c5cdd8"),
                            ))
                            fig3.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#c5cdd8", family="Inter"),
                                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                yaxis=dict(autorange="reversed", showgrid=False),
                                margin=dict(l=0,r=40,t=5,b=5), height=240,
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("BoW model not loaded.")

                # ── Agreement banner ──
                if svm_pred and bow_pred:
                    if svm_pred == bow_pred:
                        st.success(f"✅ Both models agree: **{CATEGORY_ICONS.get(svm_pred,'📰')} {svm_pred}**")
                    else:
                        st.warning(f"⚠️ Models disagree — SVM says **{svm_pred}**, BoW says **{bow_pred}**")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN
# ══════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("Train each model independently using the same dataset.")

    # ── SVM section ──
    with st.expander("🔵 SVM + TF-IDF", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            svm_data   = st.text_input("Dataset path (SVM)", value=DEFAULT_DATA, key="svm_data")
            svm_c      = st.select_slider("C value", options=[0.01,0.1,0.5,1.0,2.0,5.0,10.0], value=1.0, key="svm_c")
        with sc2:
            svm_feat   = st.select_slider("Max TF-IDF features", options=[10000,50000,100000,200000], value=100000, key="svm_feat")
            svm_ngram  = st.radio("N-gram range", [1,2], index=1, horizontal=True, key="svm_ngram")

        if _svm_running:
            st.button("⏳ SVM training…", disabled=True, use_container_width=True, key="svm_btn_dis")
        else:
            if st.button("🚀 Train SVM", type="primary", key="svm_train_btn"):
                if not os.path.exists(svm_data):
                    st.error(f"Dataset not found: {svm_data}")
                else:
                    _svm_log, _svm_done, _svm_running, _svm_results = [], False, True, None
                    threading.Thread(target=train_svm_thread,
                                    args=(svm_data, svm_c, svm_feat, svm_ngram),
                                    daemon=True).start()
                    st.rerun()

        if _svm_log:
            st.markdown("<div class='log-box'>"+"<br>".join(_svm_log)+"</div>", unsafe_allow_html=True)
            if _svm_running:
                time.sleep(2); st.rerun()
            if _svm_done:
                if _svm_results: st.session_state.svm_results = _svm_results
                st.success("✅ SVM training complete!")
                if st.button("📂 Load SVM Model", key="load_svm"):
                    load_svm.clear()
                    p,lm = load_svm()
                    st.session_state.svm_pipeline = p
                    st.session_state.svm_labels   = lm
                    st.rerun()

    st.markdown("---")

    # ── BoW section ──
    with st.expander("🟠 Logistic Regression + Bag-of-Words", expanded=True):
        bc1, bc2 = st.columns(2)
        with bc1:
            bow_data   = st.text_input("Dataset path (BoW)", value=DEFAULT_DATA, key="bow_data")
            bow_c      = st.select_slider("C value", options=[0.01,0.1,0.5,1.0,2.0,5.0,10.0], value=5.0, key="bow_c")
        with bc2:
            bow_feat   = st.select_slider("Max BoW features", options=[10000,50000,100000,200000], value=100000, key="bow_feat")
            bow_ngram  = st.radio("N-gram range", [1,2], index=1, horizontal=True, key="bow_ngram")

        bow_binary = st.toggle("Binary mode (word presence only)", value=False, key="bow_binary")

        if _bow_running:
            st.button("⏳ BoW training…", disabled=True, use_container_width=True, key="bow_btn_dis")
        else:
            if st.button("🚀 Train LR + BoW", type="primary", key="bow_train_btn"):
                if not os.path.exists(bow_data):
                    st.error(f"Dataset not found: {bow_data}")
                else:
                    _bow_log, _bow_done, _bow_running, _bow_results = [], False, True, None
                    threading.Thread(target=train_bow_thread,
                                    args=(bow_data, bow_c, bow_feat, bow_ngram, bow_binary),
                                    daemon=True).start()
                    st.rerun()

        if _bow_log:
            st.markdown("<div class='log-box'>"+"<br>".join(_bow_log)+"</div>", unsafe_allow_html=True)
            if _bow_running:
                time.sleep(2); st.rerun()
            if _bow_done:
                if _bow_results: st.session_state.bow_results = _bow_results
                st.success("✅ BoW training complete!")
                if st.button("📂 Load BoW Model", key="load_bow"):
                    load_bow.clear()
                    p,lm = load_bow()
                    st.session_state.bow_pipeline = p
                    st.session_state.bow_labels   = lm
                    st.rerun()

# ══════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ══════════════════════════════════════════════════════════════════
with tab_results:
    svm_r = st.session_state.svm_results
    bow_r = st.session_state.bow_results

    if svm_r is None and bow_r is None:
        st.info("No results yet. Train both models first.")
    else:
        # ── Overall metric comparison ──
        st.markdown("#### Overall Performance Comparison")
        header = st.columns([2,1,1,1,1])
        header[0].markdown("**Model**")
        for h, label in zip(header[1:], ["Accuracy","F1-Score","Precision","Recall"]):
            h.markdown(f"**{label}**")

        for model_name, tag, res, color in [
            ("SVM + TF-IDF",              "svm-tag", svm_r, "#2196f3"),
            ("Logistic Regression + BoW", "bow-tag", bow_r, "#f39c12"),
        ]:
            if res:
                row = st.columns([2,1,1,1,1])
                row[0].markdown(f'<span class="{tag}">{model_name}</span>', unsafe_allow_html=True)
                for cell, key in zip(row[1:], ["accuracy","f1","precision","recall"]):
                    cell.markdown(f"**{res[key]:.2%}**")

        # ── Side-by-side F1 bar chart ──
        if svm_r and bow_r:
            st.markdown("---")
            st.markdown("#### Per-Class F1 Score Comparison")

            svm_pc = {k:v for k,v in svm_r["per_class"].items()
                    if k not in ("accuracy","macro avg","weighted avg")}
            bow_pc = {k:v for k,v in bow_r["per_class"].items()
                    if k not in ("accuracy","macro avg","weighted avg")}
            cats    = sorted(svm_pc.keys())
            svm_f1s = [svm_pc[c]["f1-score"] for c in cats]
            bow_f1s = [bow_pc.get(c,{}).get("f1-score",0) for c in cats]
            labels  = [f"{CATEGORY_ICONS.get(c,'📰')} {c}" for c in cats]

            fig = go.Figure([
                go.Bar(name="SVM + TF-IDF",     x=labels, y=svm_f1s,
                    marker_color="#2196f3",
                    text=[f"{v:.2f}" for v in svm_f1s], textposition="outside",
                    textfont=dict(color="#c5cdd8", size=10)),
                go.Bar(name="LR + BoW", x=labels, y=bow_f1s,
                    marker_color="#f39c12",
                    text=[f"{v:.2f}" for v in bow_f1s], textposition="outside",
                    textfont=dict(color="#c5cdd8", size=10)),
            ])
            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c5cdd8", family="Inter"),
                yaxis=dict(range=[0,1.18], gridcolor="#2a3040", title="F1-Score"),
                xaxis=dict(tickangle=-35),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a3040", borderwidth=1),
                margin=dict(l=0,r=0,t=20,b=140), height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Detailed table ──
            st.markdown("#### Detailed Metrics Table")
            rows = []
            for c in sorted(svm_pc.keys()):
                sf = svm_pc.get(c,{})
                bf = bow_pc.get(c,{})
                rows.append({
                    "Category"     : f"{CATEGORY_ICONS.get(c,'📰')} {c}",
                    "SVM F1"       : f"{sf.get('f1-score',0):.4f}",
                    "SVM Precision": f"{sf.get('precision',0):.4f}",
                    "SVM Recall"   : f"{sf.get('recall',0):.4f}",
                    "BoW F1"       : f"{bf.get('f1-score',0):.4f}",
                    "BoW Precision": f"{bf.get('precision',0):.4f}",
                    "BoW Recall"   : f"{bf.get('recall',0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            # Only one model trained — show its per-class chart
            res   = svm_r or bow_r
            color = "#2196f3" if svm_r else "#f39c12"
            name  = "SVM + TF-IDF" if svm_r else "LR + BoW"
            pc    = {k:v for k,v in res["per_class"].items()
                    if k not in ("accuracy","macro avg","weighted avg")}
            rows  = sorted(pc.items(), key=lambda x: x[1]["f1-score"], reverse=True)
            names = [f"{CATEGORY_ICONS.get(k,'📰')} {k}" for k,_ in rows]
            f1s   = [v["f1-score"] for _,v in rows]
            fig   = go.Figure(go.Bar(x=names, y=f1s, marker_color=color,
                                    text=[f"{v:.2f}" for v in f1s],
                                    textposition="outside",
                                    textfont=dict(color="#c5cdd8")))
            fig.update_layout(
                title=name,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c5cdd8", family="Inter"),
                yaxis=dict(range=[0,1.1], gridcolor="#2a3040", title="F1-Score"),
                xaxis=dict(tickangle=-35),
                margin=dict(l=0,r=0,t=30,b=130), height=440,
            )
            st.plotly_chart(fig, use_container_width=True)