"""
News Category Classifier — Combined App
========================================
Models:
  1. SVM + TF-IDF       (LinearSVC + TfidfVectorizer)
  2. LR  + Bag-of-Words (LogisticRegression + CountVectorizer)

Tabs:
  🔍 Predict & Compare  — run both models on one article, side by side
  🔵 SVM + TF-IDF       — that model on its own, with feature attributions
  🟠 LR + Bag-of-Words  — that model on its own, with feature attributions
  📊 Results            — held-out test-set metrics for both models

Run with:  streamlit run app.py
Requires:  pip install -r requirements.txt
"""

import os, json, re
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

CATEGORY_ICONS = {
    "POLITICS & NEWS":"📰","WELLNESS":"🧘","ENTERTAINMENT":"🎬","TRAVEL":"✈️",
    "STYLE & BEAUTY":"💄","FOOD & DRINK":"🍽️","SPORTS":"⚽",
    "HOME & LIVING":"🏠","WEDDINGS & DIVORCE":"💍",
}

# Per-model presentation + copy, so the three prediction tabs share one code path.
MODEL_CFG = {
    "svm": {
        "name"      : "SVM + TF-IDF",
        "short"     : "SVM",
        "dot"       : "🔵",
        "color"     : "#2196f3",
        "tag"       : "svm-tag",
        "pred_css"  : "pred-svm",
        "metric_css": "metric-svm",
        # LinearSVC has no predict_proba — the number shown is a softmax over
        # the one-vs-rest decision margins, which ranks correctly but is NOT a
        # calibrated probability. Label it honestly.
        "score_lbl" : "Score",
        "score_note": "Softmax over one-vs-rest SVM margins — ranks the classes "
                      "correctly, but is **not** a calibrated probability, so it "
                      "is not comparable to the LR probability.",
        "how"       : "**LinearSVC** over a `FeatureUnion` of two TF-IDF blocks: "
                      "word 1–2grams (120k features) and character 3–5grams "
                      "(60k features, `char_wb`). TF-IDF down-weights words that "
                      "appear everywhere, and the character n-grams give the model "
                      "signal from misspellings and word fragments. `C=0.3`.",
    },
    "bow": {
        "name"      : "Logistic Regression + Bag-of-Words",
        "short"     : "LR + BoW",
        "dot"       : "🟠",
        "color"     : "#f39c12",
        "tag"       : "bow-tag",
        "pred_css"  : "pred-bow",
        "metric_css": "metric-bow",
        "score_lbl" : "Probability",
        "score_note": "A real calibrated probability from `predict_proba` — the "
                      "nine values sum to 100%.",
        "how"       : "**LogisticRegression** over a `CountVectorizer` "
                      "bag-of-words: raw counts of word 1–2grams (120k features, "
                      "`min_df=2`). No IDF weighting — a word that appears in "
                      "every article counts the same as a rare one. `C=0.3`.",
    },
}

EXAMPLES = [
    ("📰 Politics & News",    "President signs new infrastructure spending bill", "The $1.2 trillion package will fund roads, bridges and broadband nationwide."),
    ("🧘 Wellness",           "Study finds daily walking cuts heart disease risk by 30%", "Researchers tracked 10,000 adults over five years."),
    ("🎬 Entertainment",      "Oscars 2024: Best Picture winner", "Oppenheimer swept the Oscars taking home seven awards."),
    ("✈️ Travel",             "Top 10 hidden gem destinations for 2024", "These under-the-radar spots offer stunning views without the crowds."),
    ("💄 Style & Beauty",     "Fall fashion week highlights bold colors and oversized silhouettes", "Designers embraced maximalism on the runway this season."),
    ("🍽️ Food & Drink",       "This 20-minute pasta recipe is going viral", "Home cooks are raving about the creamy garlic sauce."),
    ("⚽ Sports",             "World Cup Final: Argentina wins", "Argentina beat France on penalties in a thrilling final."),
    ("🏠 Home & Living",      "Small space, big style: 5 tips for tiny apartments", "Designers share their favorite tricks for maximizing square footage."),
    ("💍 Weddings & Divorce", "Celebrity couple announces surprise wedding", "The intimate ceremony took place at a private estate."),
]

# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────
for key, val in [
    ("svm_pipeline", None), ("svm_labels", None), ("svm_results", None), ("svm_error", None),
    ("bow_pipeline", None), ("bow_labels", None), ("bow_results", None), ("bow_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def clean_text(text):
    """Must stay identical to build_text_column()/clean_text() in
    preprocess_news_data.py — the models were trained on text cleaned that way,
    so any drift here silently degrades every prediction."""
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


@st.cache_resource
def svm_feature_names():
    """180k names — built once, reused for every attribution chart."""
    return load_svm()[0].named_steps["features"].get_feature_names_out()


@st.cache_resource
def bow_feature_names():
    """120k names — same reason as svm_feature_names()."""
    return load_bow()[0].named_steps["bow"].get_feature_names_out()


def svm_exists(): return os.path.exists(SVM_MODEL) and os.path.exists(SVM_LABELS)
def bow_exists(): return os.path.exists(BOW_MODEL) and os.path.exists(BOW_LABELS)


def _score_dict(pipe, label_map, values):
    """Map a model's per-column scores onto category names.

    Uses pipe.classes_ rather than assuming column i == encoded label i, so the
    app stays correct if a model is ever retrained on a subset of classes.
    """
    return {label_map[str(int(c))]: float(v) for c, v in zip(pipe.classes_, values)}


def predict_svm(headline, description):
    text   = clean_text(headline + " " + description)
    pipe   = st.session_state.svm_pipeline
    lm     = st.session_state.svm_labels
    scores = pipe.decision_function([text])[0]
    exp    = np.exp(scores - np.max(scores))
    probs  = exp / exp.sum()
    idx    = int(np.argmax(probs))
    pred   = lm[str(int(pipe.classes_[idx]))]
    # margin gap is the honest "how sure is the SVM" signal
    ordered = np.sort(scores)[::-1]
    margin  = float(ordered[0] - ordered[1])
    return pred, _score_dict(pipe, lm, probs), margin


def predict_bow(headline, description):
    text  = clean_text(headline + " " + description)
    pipe  = st.session_state.bow_pipeline
    lm    = st.session_state.bow_labels
    probs = pipe.predict_proba([text])[0]
    idx   = int(np.argmax(probs))
    pred  = lm[str(int(pipe.classes_[idx]))]
    return pred, _score_dict(pipe, lm, probs), None


def predict(mkey, headline, description):
    return (predict_svm if mkey == "svm" else predict_bow)(headline, description)


def _contributions(pipe, feature_values, feature_names, class_row, top_n, keep=None):
    """Rank features by their signed push toward the predicted class
    (feature value x that class's coefficient)."""
    coef  = pipe.steps[-1][1].coef_[class_row]
    rows  = feature_values.tocoo()
    pairs = []
    for j, v in zip(rows.col, rows.data):
        name = feature_names[j]
        if keep and not name.startswith(keep):
            continue
        if keep:
            name = name[len(keep):]
        pairs.append((name, float(v) * float(coef[j])))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def word_contributions(mkey, headline, description, pred, top_n=10):
    """Which words pushed this article toward its predicted category."""
    text = clean_text(headline + " " + description)
    pipe = st.session_state[f"{mkey}_pipeline"]
    lm   = st.session_state[f"{mkey}_labels"]
    # row of coef_ belonging to the predicted class
    encoded   = next(int(k) for k, v in lm.items() if v == pred)
    class_row = int(np.where(pipe.classes_ == encoded)[0][0])

    if mkey == "svm":
        fu = pipe.named_steps["features"]
        return _contributions(pipe, fu.transform([text]), svm_feature_names(),
                              class_row, top_n, keep="word_tfidf__")
    vec = pipe.named_steps["bow"]
    return _contributions(pipe, vec.transform([text]), bow_feature_names(),
                          class_row, top_n)


# ── chart builders ───────────────────────────────────────────────
def score_figure(scores, pred, color, height=300):
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
    fig = go.Figure(go.Bar(
        x=[v for _, v in top],
        y=[f"{CATEGORY_ICONS.get(c,'📰')} {c}" for c, _ in top],
        orientation="h",
        marker_color=[color if c == pred else "#2a3040" for c, _ in top],
        text=[f"{v:.1%}" for _, v in top],
        textposition="outside", textfont=dict(color="#c5cdd8", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c5cdd8", family="Inter"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=0, r=70, t=10, b=10), height=height,
    )
    return fig


def contribution_figure(pairs, color, height=280):
    """Signed bars: model colour = pushed toward the prediction, grey = against."""
    words = [w for w, _ in pairs]
    vals  = [v for _, v in pairs]
    fig = go.Figure(go.Bar(
        x=vals, y=words, orientation="h",
        marker_color=[color if v > 0 else "#e74c3c" for v in vals],
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside", textfont=dict(color="#c5cdd8", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c5cdd8", family="Inter"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True,
                   zerolinecolor="#2a3040"),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=0, r=70, t=10, b=10), height=height,
    )
    return fig


# ── shared UI blocks ─────────────────────────────────────────────
def article_inputs(prefix):
    """Headline + description inputs. `prefix` keeps widget keys unique — every
    tab renders on every rerun, so duplicate keys would raise."""
    hk, dk = f"{prefix}_headline", f"{prefix}_description"
    st.session_state.setdefault(hk, "")
    st.session_state.setdefault(dk, "")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        headline = st.text_input("Headline", key=hk,
                                 placeholder="e.g. Senate passes new climate bill…")
    with c2:
        description = st.text_area("Short Description (optional)", key=dk, height=95,
                                   placeholder="Brief summary of the article…")
    return headline, description


def _fill_example(prefix, h, d):
    st.session_state[f"{prefix}_headline"]    = h
    st.session_state[f"{prefix}_description"] = d
    st.session_state.pop(f"{prefix}_result", None)   # stale result no longer matches


def example_buttons(prefix):
    st.markdown("###### Quick examples")
    for row_start in range(0, len(EXAMPLES), 3):
        cols = st.columns(3)
        for i, (col, (lbl, h, d)) in enumerate(zip(cols, EXAMPLES[row_start:row_start + 3])):
            col.button(lbl, width="stretch", key=f"{prefix}_example_{row_start + i}",
                       on_click=_fill_example, args=(prefix, h, d))


def prediction_box(mkey, pred, score):
    cfg  = MODEL_CFG[mkey]
    icon = CATEGORY_ICONS.get(pred, "📰")
    st.markdown(f"""
    <div class="{cfg['pred_css']}">
    <div style="color:{cfg['color']};font-size:0.75rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">{cfg['dot']} {cfg['name']}</div>
    <div class="pred-cat">{icon} {pred}</div>
    <div class="pred-conf">{cfg['score_lbl']}: {score:.1%}</div>
    </div>""", unsafe_allow_html=True)


def model_metric_cards(mkey):
    res = st.session_state[f"{mkey}_results"]
    if not res:
        return
    cfg  = MODEL_CFG[mkey]
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("accuracy", "Accuracy"), ("f1", "F1-Score"),
                                        ("precision", "Precision"), ("recall", "Recall")]):
        col.markdown(f"""
        <div class="{cfg['metric_css']}">
          <div class="val">{res[key]:.1%}</div>
          <div class="metric-lbl">{label}</div>
        </div>""", unsafe_allow_html=True)


def class_scorecard(mkey, pred):
    """This model's held-out performance on the category it just predicted."""
    res = st.session_state[f"{mkey}_results"]
    if not res:
        return
    pc = res["per_class"].get(pred)
    if not pc:
        return
    st.caption(
        f"On the held-out test set this model scores **F1 {pc['f1-score']:.3f}** "
        f"on {CATEGORY_ICONS.get(pred,'📰')} {pred} "
        f"(precision {pc['precision']:.3f} · recall {pc['recall']:.3f} · "
        f"{int(pc['support']):,} test articles)."
    )


# ─────────────────────────────────────────────────────────────────
# Auto-load models if saved on disk
# ─────────────────────────────────────────────────────────────────
if svm_exists() and st.session_state.svm_pipeline is None:
    try:
        p, lm = load_svm()
        st.session_state.svm_pipeline = p
        st.session_state.svm_labels   = lm
    except Exception as e:                      # surfaced in the sidebar, not swallowed
        st.session_state.svm_error = f"{type(e).__name__}: {e}"

if bow_exists() and st.session_state.bow_pipeline is None:
    try:
        p, lm = load_bow()
        st.session_state.bow_pipeline = p
        st.session_state.bow_labels   = lm
    except Exception as e:
        st.session_state.bow_error = f"{type(e).__name__}: {e}"

for path, key in [(SVM_RESULTS, "svm_results"), (BOW_RESULTS, "bow_results")]:
    if os.path.exists(path) and st.session_state[key] is None:
        try:
            with open(path) as f:
                st.session_state[key] = json.load(f)
        except Exception as e:
            st.warning(f"Could not read {path}: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
<h1>📰 News Category Classifier</h1>
<p>
    <span class="svm-tag">SVM · TF-IDF</span>&nbsp;
    <span class="bow-tag">Logistic Regression · Bag-of-Words</span>
    &nbsp;· 9 categories · HuffPost News Dataset
</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Status")

    for mkey, exists in [("svm", svm_exists()), ("bow", bow_exists())]:
        cfg = MODEL_CFG[mkey]
        st.markdown(f"**{cfg['dot']} {cfg['name']}**")
        if st.session_state[f"{mkey}_pipeline"] is not None:
            st.markdown('<span class="pill-green">✔ Loaded</span>', unsafe_allow_html=True)
        elif exists:
            st.markdown('<span class="pill-yellow">⚠ Failed to load</span>', unsafe_allow_html=True)
            if st.session_state[f"{mkey}_error"]:
                st.caption(st.session_state[f"{mkey}_error"])
        else:
            st.markdown('<span class="pill-red">✘ Not found</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 9 Categories")
    for cat, icon in CATEGORY_ICONS.items():
        st.markdown(f"<small>{icon} {cat}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────
tab_compare, tab_svm, tab_bow, tab_results = st.tabs([
    "🔍 Predict & Compare",
    "🔵 SVM + TF-IDF",
    "🟠 LR + Bag-of-Words",
    "📊 Results",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & COMPARE
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    if st.session_state.svm_pipeline is None and st.session_state.bow_pipeline is None:
        st.info("⚠️ Model files not found. Make sure svm_news_classifier.joblib and "
                "lr_bow_news_classifier.joblib are present in the app directory.")
    else:
        st.markdown("#### Enter a news article")
        headline, description = article_inputs("cmp")
        example_buttons("cmp")

        st.markdown("---")
        if st.button("🔍 Classify with Both Models", type="primary", key="cmp_go"):
            if not headline.strip() and not description.strip():
                st.warning("Please enter at least a headline.")
            else:
                with st.spinner("Running both models…"):
                    out = {}
                    for mkey in ("svm", "bow"):
                        if st.session_state[f"{mkey}_pipeline"] is not None:
                            p, s, m = predict(mkey, headline, description)
                            out[mkey] = {"pred": p, "scores": s, "margin": m,
                                         "words": word_contributions(mkey, headline, description, p)}
                    st.session_state.cmp_result = out

        result = st.session_state.get("cmp_result")
        if result:
            left, right = st.columns(2, gap="large")
            for col, mkey in [(left, "svm"), (right, "bow")]:
                cfg = MODEL_CFG[mkey]
                with col:
                    r = result.get(mkey)
                    if not r:
                        st.warning(f"{cfg['name']} not loaded.")
                        continue
                    prediction_box(mkey, r["pred"], r["scores"][r["pred"]])
                    st.plotly_chart(score_figure(r["scores"], r["pred"], cfg["color"]),
                                    key=f"cmp_scores_{mkey}")
                    st.markdown("**🔤 Words driving this prediction**")
                    if r["words"]:
                        st.plotly_chart(contribution_figure(r["words"], cfg["color"]),
                                        key=f"cmp_words_{mkey}")
                    else:
                        st.caption("No words from this article are in the model's vocabulary.")

            if "svm" in result and "bow" in result:
                if result["svm"]["pred"] == result["bow"]["pred"]:
                    p = result["svm"]["pred"]
                    st.success(f"✅ Both models agree: **{CATEGORY_ICONS.get(p,'📰')} {p}**")
                else:
                    st.warning(f"⚠️ Models disagree — SVM says **{result['svm']['pred']}**, "
                               f"BoW says **{result['bow']['pred']}**")
                st.caption(
                    "⚠️ The two percentages are **not** on the same scale. The LR number is a "
                    "real probability; the SVM number is a softmax over decision margins "
                    "(LinearSVC has no `predict_proba`), which reads systematically lower. "
                    "Compare the *rankings*, not the raw percentages."
                )

# ══════════════════════════════════════════════════════════════════
# TABS 2 & 3 — SINGLE-MODEL PREDICTION
# ══════════════════════════════════════════════════════════════════
def render_model_tab(mkey):
    cfg  = MODEL_CFG[mkey]
    pipe = st.session_state[f"{mkey}_pipeline"]

    st.markdown(f"#### {cfg['dot']} {cfg['name']}")
    st.caption(cfg["how"])

    if pipe is None:
        st.info(f"⚠️ {cfg['name']} is not loaded — no prediction possible in this tab. "
                f"See **Model Status** in the sidebar.")
        return

    model_metric_cards(mkey)
    st.markdown("---")

    st.markdown("#### Enter a news article")
    headline, description = article_inputs(mkey)
    example_buttons(mkey)

    st.markdown("---")
    if st.button(f"🔍 Classify with {cfg['short']}", type="primary", key=f"{mkey}_go"):
        if not headline.strip() and not description.strip():
            st.warning("Please enter at least a headline.")
        else:
            with st.spinner(f"Running {cfg['short']}…"):
                p, s, m = predict(mkey, headline, description)
                st.session_state[f"{mkey}_result"] = {
                    "pred": p, "scores": s, "margin": m,
                    "words": word_contributions(mkey, headline, description, p, top_n=12),
                }

    r = st.session_state.get(f"{mkey}_result")
    if not r:
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        prediction_box(mkey, r["pred"], r["scores"][r["pred"]])
        st.caption(cfg["score_note"])
        if r["margin"] is not None:
            st.caption(f"Margin between the top two classes: **{r['margin']:.2f}** "
                       f"(bigger = more decisive; below ~0.2 is a close call).")
        class_scorecard(mkey, r["pred"])
        st.markdown("##### All 9 categories")
        st.plotly_chart(score_figure(r["scores"], r["pred"], cfg["color"], height=360),
                        key=f"{mkey}_scores")

    with right:
        st.markdown("##### 🔤 Words driving this prediction")
        st.caption(f"Each bar is a word's feature value × its weight for "
                   f"**{r['pred']}**. Coloured bars pushed toward that category, "
                   f"red bars pushed away.")
        if r["words"]:
            st.plotly_chart(contribution_figure(r["words"], cfg["color"], height=420),
                            key=f"{mkey}_words")
            wdf = pd.DataFrame(r["words"], columns=["Feature", "Contribution"])
            wdf["Contribution"] = wdf["Contribution"].map(lambda v: f"{v:+.4f}")
            with st.expander("Show as table"):
                st.dataframe(wdf, hide_index=True)
        else:
            st.info("None of the words in this article are in the model's vocabulary, "
                    "so the prediction falls back to the class priors.")


with tab_svm:
    render_model_tab("svm")

with tab_bow:
    render_model_tab("bow")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
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

        for mkey, res in [("svm", svm_r), ("bow", bow_r)]:
            if res:
                cfg = MODEL_CFG[mkey]
                row = st.columns([2,1,1,1,1])
                row[0].markdown(f'<span class="{cfg["tag"]}">{cfg["name"]}</span>',
                                unsafe_allow_html=True)
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
            st.plotly_chart(fig, key="results_f1_compare")

            # ── Detailed table ──
            st.markdown("#### Detailed Metrics Table")
            rows = []
            for c in cats:
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
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        else:
            # Only one model trained — show its per-class chart
            mkey  = "svm" if svm_r else "bow"
            res   = svm_r or bow_r
            cfg   = MODEL_CFG[mkey]
            pc    = {k:v for k,v in res["per_class"].items()
                    if k not in ("accuracy","macro avg","weighted avg")}
            rows  = sorted(pc.items(), key=lambda x: x[1]["f1-score"], reverse=True)
            names = [f"{CATEGORY_ICONS.get(k,'📰')} {k}" for k,_ in rows]
            f1s   = [v["f1-score"] for _,v in rows]
            fig   = go.Figure(go.Bar(x=names, y=f1s, marker_color=cfg["color"],
                                    text=[f"{v:.2f}" for v in f1s],
                                    textposition="outside",
                                    textfont=dict(color="#c5cdd8")))
            fig.update_layout(
                title=cfg["name"],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c5cdd8", family="Inter"),
                yaxis=dict(range=[0,1.1], gridcolor="#2a3040", title="F1-Score"),
                xaxis=dict(tickangle=-35),
                margin=dict(l=0,r=0,t=30,b=130), height=440,
            )
            st.plotly_chart(fig, key="results_single_model")
