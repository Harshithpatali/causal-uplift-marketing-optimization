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
models = pd.read_csv(M / "model_comparison.csv")
curve = pd.read_csv(M / "targeting_curve.csv")

conv = ate[ate.outcome == "conversion"]
m = conv[conv.treatment.isin(["Mens E-Mail", "Womens E-Mail"])]
best_t = m.loc[m.absolute_lift.idxmax(), "treatment"]
best_model = models.loc[models.auuc.idxmax()]
best_depth = curve.loc[curve.expected_incremental_revenue.idxmax()]

eyebrow("TRIAL READOUT · MEN'S VS WOMEN'S EMAIL VS CONTROL")
st.title("Executive Overview")

c = st.columns(5)
c[0].metric("Customers", f"{groups.customers.sum():,}")
c[1].metric("Men's lift", f"{m[m.treatment == 'Mens E-Mail'].absolute_lift.iloc[0] * 100:.2f} pp")
c[2].metric("Women's lift", f"{m[m.treatment == 'Womens E-Mail'].absolute_lift.iloc[0] * 100:.2f} pp")
c[3].metric("Best treatment", best_t)
c[4].metric("Best model AUUC", f"{best_model.auuc:.5f}")

rule()

st.subheader("Recommendation")
st.success(
    f"On held-out data, the highest estimated incremental revenue lands at roughly "
    f"**{best_depth.targeting_depth:.0%} targeting depth** — {best_depth.customers_targeted:,} active sends. "
    f"Estimated incremental conversions: **{best_depth.expected_incremental_conversions:.1f}**. "
    f"Estimated incremental revenue: **${best_depth.expected_incremental_revenue:,.0f}**."
)

not_targeted = groups.customers.sum() - best_depth.customers_targeted
split_bar(
    "Recommended sends", best_depth.customers_targeted,
    "Held back", not_targeted,
    left_color=PINE, right_color=STONE, decimals=0,
)
st.caption("Inverse-propensity-weighted estimates on held-out randomized observations — a read on the trial, not a guarantee about the future.")

rule()

fig1 = px.bar(
    conv[conv.treatment != "Any Email"], x="treatment", y="absolute_lift", color="treatment",
    title="Conversion lift vs. control", color_discrete_sequence=[PINE, STONE, "#AD8A2E"],
)
st.plotly_chart(plotly_layout(fig1), use_container_width=True)

fig2 = px.line(
    curve, x="targeting_depth", y=["expected_incremental_conversions", "expected_incremental_revenue"],
    markers=True, title="Incremental impact by targeting depth",
)
st.plotly_chart(plotly_layout(fig2), use_container_width=True)
