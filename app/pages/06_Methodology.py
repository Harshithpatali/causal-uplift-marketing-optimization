import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from theme import inject_theme, eyebrow, rule  # noqa: E402

inject_theme()

eyebrow("HOW TO READ THE NUMBERS ON THIS CONSOLE")
st.title("Methodology")

sections = [
    ("Randomized experiment",
     "Customers were assigned to Men's Email, Women's Email, or No Email. Random assignment is what "
     "lets the numbers on this console be read as cause and effect, not just correlation."),
    ("Average treatment effect",
     "The average treatment effect is the difference between a treatment group's outcome mean and "
     "the control group's outcome mean. This project reports it for visit, conversion, and spend, "
     "each with a confidence interval."),
    ("Heterogeneous treatment effect",
     "No single customer's causal effect is directly observable — you only ever see one of their two "
     "possible outcomes. The uplift models estimate how that effect *varies* across customers, using "
     "pre-treatment features."),
    ("Two-model / T-learner",
     "Separate outcome models are fit for treatment and for control. Estimated uplift is "
     "P(Y=1 | X, treatment) − P(Y=1 | X, control)."),
    ("Class transformation",
     "For each randomized binary comparison, treatment assignment and its empirical propensity are "
     "used to transform the outcome. A regression model then learns that transformed signal directly."),
    ("Qini and AUUC",
     "Customers are ranked by predicted uplift, and cumulative incremental response is estimated with "
     "inverse-propensity weighting. Qini is the maximum cumulative gain; AUUC is the trapezoidal area "
     "under that curve."),
    ("Targeting policy",
     "The policy compares each customer's estimated Men's and Women's uplift. If neither is positive, "
     "the customer stays on No Email. The threshold is configurable in `src/config.py`."),
    ("Spend",
     "Spend is zero-inflated, so separate treatment and control regression models estimate conditional "
     "spend uplift. Treat this as an estimated business-value signal, not a guaranteed revenue forecast."),
]

for i, (title, body) in enumerate(sections, start=1):
    st.markdown(f"**{i:02d} · {title}**")
    st.write(body)
    if i < len(sections):
        rule()

rule()
st.subheader("Limitations")
st.markdown(
    """
    - Individual treatment effects are estimates, not observed truths.
    - Held-out policy estimates carry sampling uncertainty.
    - Email costs aren't provided here, so this reports incremental revenue, not net profit.
    - Production deployment can face distribution shift and operational constraints this trial didn't have.
    """
)
