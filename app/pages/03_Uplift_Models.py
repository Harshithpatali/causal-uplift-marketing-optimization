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
models = pd.read_csv(M / "model_comparison.csv")

eyebrow("MODEL BAKE-OFF · TWO-MODEL, CLASS TRANSFORMATION, AND VARIANTS")
st.title("Uplift Models")

choice = st.selectbox("Comparison", models.comparison.unique())
d = models[models.comparison == choice]

st.dataframe(
    d.style.format({
        "auuc": "{:.6f}", "qini": "{:.6f}",
        "uplift_at_10": "{:.3%}", "uplift_at_20": "{:.3%}", "uplift_at_30": "{:.3%}",
    }),
    use_container_width=True,
)

best = d.loc[d.auuc.idxmax()]
st.success(f"Best model for **{choice}**: **{best.model}** (AUUC {best.auuc:.6f}).")

rule()

slug = {"Mens E-Mail vs Control": "mens", "Womens E-Mail vs Control": "womens", "Any Email vs Control": "any"}[choice]
q = pd.read_csv(M / f"qini_{slug}_two_model.csv")
fig1 = px.line(q, x="fraction", y="incremental_response", title=f"Qini curve — cumulative incremental response ({choice})")
st.plotly_chart(plotly_layout(fig1), use_container_width=True)

pred = pd.read_csv(ROOT / "artifacts/predictions/customer_predictions.csv")
col = {
    "Mens E-Mail vs Control": "uplift_mens",
    "Womens E-Mail vs Control": "uplift_womens",
    "Any Email vs Control": "uplift_any_email",
}[choice]

positive = int((pred[col] > 0).sum())
non_positive = int((pred[col] <= 0).sum())
split_bar("Predicted uplift > 0", positive, "Predicted uplift ≤ 0", non_positive, left_color=PINE, right_color=STONE, decimals=0)
st.caption(f"Share of customers where {best.model} expects {choice.split(' vs')[0]} to move the needle at all.")

fig2 = px.histogram(pred, x=col, nbins=60, title="Customer-level estimated uplift distribution")
st.plotly_chart(plotly_layout(fig2), use_container_width=True)
