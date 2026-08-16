import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from theme import inject_theme, eyebrow, rule, split_bar, plotly_layout, PINE, STONE  # noqa: E402

inject_theme()
M = ROOT / "artifacts/metrics"

ate = pd.read_csv(M / "ate_results.csv")
groups = pd.read_csv(M / "group_summary.csv")

eyebrow("RANDOMIZATION CHECK · ARM SIZES AND OUTCOME DIFFERENCES")
st.title("Experiment Results")

email_customers = int(groups.loc[groups.segment != "No E-Mail", "customers"].sum())
control_customers = int(groups.loc[groups.segment == "No E-Mail", "customers"].sum())
split_bar("Email arms", email_customers, "Control arm", control_customers, left_color=PINE, right_color=STONE, decimals=0)
st.caption("The randomization split itself — how customers were actually divided before any modeling happened.")

rule()

fig = px.bar(
    groups, x="segment", y="customers", title="Customers by experimental arm",
    color="segment", color_discrete_sequence=[PINE, "#5B7A8C", STONE],
)
st.plotly_chart(plotly_layout(fig, height=360), use_container_width=True)

rule()

for outcome in ["visit", "conversion", "spend"]:
    d = ate[ate.outcome == outcome]
    st.subheader(outcome.title())
    st.dataframe(
        d.style.format({
            "treatment_mean": "{:.4f}", "control_mean": "{:.4f}", "absolute_lift": "{:.4f}",
            "relative_lift": "{:.2%}", "ci_low": "{:.4f}", "ci_high": "{:.4f}", "p_value": "{:.3g}",
        }),
        use_container_width=True,
    )

st.caption("The average treatment effect is identified by random assignment. Confidence intervals use a large-sample normal approximation for the difference in means.")
