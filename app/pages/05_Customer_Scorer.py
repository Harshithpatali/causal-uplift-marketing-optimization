import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from theme import inject_theme, eyebrow, rule, split_bar, PINE, GOLD  # noqa: E402

inject_theme()
MD = ROOT / "artifacts/models"

pre = joblib.load(MD / "preprocessor.joblib")
mt = joblib.load(MD / "tlearner_mens_treatment.joblib")
mc = joblib.load(MD / "tlearner_mens_control.joblib")
wt = joblib.load(MD / "tlearner_womens_treatment.joblib")
wc = joblib.load(MD / "tlearner_womens_control.joblib")

eyebrow("SINGLE CUSTOMER · LIVE T-LEARNER SCORE")
st.title("Customer Scorer")

c1, c2, c3 = st.columns(3)
with c1:
    recency = st.number_input("Recency", 0, 100, 10)
    history = st.number_input("History", 0.0, 5000.0, 150.0)
    mens = st.selectbox("Men's history", [0, 1])
    womens = st.selectbox("Women's history", [0, 1])
with c2:
    zip_code = st.selectbox("Zip code", ["Urban", "Surburban", "Rural"])
    newbie = st.selectbox("Newbie", [0, 1])
    channel = st.selectbox("Channel", ["Web", "Phone", "Multichannel"])
with c3:
    history_segment = st.selectbox(
        "History segment",
        ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
         "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"],
    )

if st.button("Score customer", type="primary"):
    row = pd.DataFrame([{
        "recency": recency, "history": history, "mens": mens, "womens": womens,
        "zip_code": zip_code, "newbie": newbie, "channel": channel, "history_segment": history_segment,
    }])
    X = pre.transform(row)
    pc = mc.predict_proba(X)[:, 1][0]
    pm = mt.predict_proba(X)[:, 1][0]
    pw = wt.predict_proba(X)[:, 1][0]
    um, uw = pm - pc, pw - pc
    action = "Mens E-Mail" if um > max(uw, 0) else ("Womens E-Mail" if uw > max(um, 0) else "No E-Mail")

    rule()
    st.subheader(f"Recommended action: {action}")

    best_p = max(pc, pm, pw)
    split_bar("Best treatment probability", best_p * 100, "Control probability", pc * 100,
              left_color=GOLD, right_color=PINE, unit=" pp", decimals=2)

    cols = st.columns(3)
    cols[0].metric("Control", f"{pc:.2%}")
    cols[1].metric("Men's / uplift", f"{pm:.2%} / {um:+.2%}")
    cols[2].metric("Women's / uplift", f"{pw:.2%} / {uw:+.2%}")
    st.caption("Recommendation = the treatment with the highest model-estimated incremental conversion probability, if that probability is positive.")
