import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theme import inject_theme, eyebrow, rule, split_bar, PINE, STONE  # noqa: E402

st.set_page_config(page_title="Causal Uplift Marketing Intelligence", page_icon="✉️", layout="wide")
inject_theme()

M = ROOT / "artifacts/metrics"

eyebrow("RANDOMIZED CONTROLLED TRIAL · HILLSTROM EMAIL DATASET")
st.title("Causal Uplift Marketing Intelligence")
st.caption("Estimating who a marketing email actually *changes the mind of* — not who was going to convert anyway.")

# Hero stat: the trial's own randomization split, rendered with the console's signature element.
try:
    groups = pd.read_csv(M / "group_summary.csv")
    total = int(groups.customers.sum())
    email_customers = int(groups.loc[groups.segment != "No E-Mail", "customers"].sum()) if "segment" in groups else None
    control_customers = total - email_customers if email_customers is not None else None

    c = st.columns(3)
    c[0].metric("Customers randomized", f"{total:,}")
    if email_customers is not None:
        c[1].metric("Assigned to an email arm", f"{email_customers:,}")
        c[2].metric("Assigned to control", f"{control_customers:,}")
        split_bar("Email arms", email_customers, "Control", control_customers, left_color=PINE, right_color=STONE)
except FileNotFoundError:
    st.warning("No trial artifacts found yet — run `python run_project.py` once to generate them.")

rule()

st.markdown(
    """
    ### What this measures
    A predictive model answers *"who's likely to buy?"* An uplift model answers a different,
    harder question: *"who buys **because** we emailed them, and who would have bought anyway?"*
    Only the second question tells you who's worth spending a send on.

    ### How to read this console
    """
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**01 · Experiment Results**  \nThe raw randomized evidence — visit, conversion, and spend lift with confidence intervals.")
    st.markdown("**02 · Uplift Models**  \nHow well each model ranks customers by their true incremental response.")
with c2:
    st.markdown("**03 · Targeting Simulator**  \nDrag the targeting depth and watch incremental revenue move in real time.")
    st.markdown("**04 · Customer Scorer**  \nScore one customer and see the recommended action, live.")
with c3:
    st.markdown("**05 · Methodology**  \nThe causal reasoning behind every number on this console.")

rule()
st.caption("Run `python run_project.py` once before opening this dashboard, to build the model artifacts it reads from.")
