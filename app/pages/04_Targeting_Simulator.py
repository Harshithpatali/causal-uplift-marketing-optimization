import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from theme import inject_theme, eyebrow, rule, split_bar, plotly_layout, PINE, STONE, GOLD  # noqa: E402

inject_theme()
M = ROOT / "artifacts/metrics"

pred = pd.read_csv(ROOT / "artifacts/predictions/customer_predictions.csv")
curve = pd.read_csv(M / "targeting_curve.csv")

eyebrow("POLICY SIMULATOR · SET A DEPTH, SEE THE INCREMENTAL PAYOFF")
st.title("Targeting Policy Simulator")

depth = st.slider("Targeting depth", 1, 100, 50, 1) / 100
d = pred.sort_values("max_uplift", ascending=False).head(int(len(pred) * depth))
active = d[d.max_uplift > 0]

split_bar("Active sends", len(active), "Held back", len(pred) - len(active), left_color=PINE, right_color=STONE, decimals=0)

c = st.columns(5)
c[0].metric("Ranked", f"{len(d):,}")
c[1].metric("Active targets", f"{len(active):,}")
c[2].metric("Men's", f"{(active.recommended_action == 'Mens E-Mail').sum():,}")
c[3].metric("Women's", f"{(active.recommended_action == 'Womens E-Mail').sum():,}")
c[4].metric("No email", f"{len(pred) - len(active):,}")

r = curve.iloc[(curve.targeting_depth - depth).abs().argsort()[:1]].iloc[0]
c = st.columns(3)
c[0].metric("Incremental conversions", f"{r.expected_incremental_conversions:.1f}")
c[1].metric("Incremental revenue", f"${r.expected_incremental_revenue:,.0f}")
c[2].metric("Average uplift", f"{r.average_uplift:.2%}")

rule()

fig1 = px.line(
    curve, x="targeting_depth", y=["expected_incremental_conversions", "expected_incremental_revenue"],
    markers=True, title="Held-out targeting curve",
)
st.plotly_chart(plotly_layout(fig1), use_container_width=True)

pc = pd.read_csv(M / "policy_comparison.csv")
fig2 = px.bar(pc, x="policy", y="incremental_revenue", title="Policy comparison at 20% depth", color_discrete_sequence=[GOLD])
st.plotly_chart(plotly_layout(fig2, height=360), use_container_width=True)

st.dataframe(
    pc.style.format({
        "expected_conversion_rate": "{:.3%}", "incremental_conversions": "{:.1f}",
        "expected_revenue": "${:.2f}", "incremental_revenue": "${:.2f}",
    }),
    use_container_width=True,
)
