"""
Bayesian & Sequential Experimentation Platform
Main Streamlit application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Ensure src is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.generators import (
    generate_bernoulli_ab,
    generate_normal_ab,
    generate_sequential_stream,
    generate_multi_variant,
)
from src.frequentist.power_analysis import (
    sample_size_proportion,
    power_proportion,
    power_curve_proportion,
)
from src.frequentist.sequential import sequential_test_proportion, two_proportion_ztest
from src.frequentist.multiple_testing import apply_multiple_testing_correction
from src.bayesian.ab_testing import bayesian_ab_test
from src.utils.metrics import summarize_ab_result, effect_size_cohen_h

st.set_page_config(
    page_title="Bayesian & Sequential Experimentation Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Bayesian & Sequential Experimentation Platform")
st.markdown(
    """
    A teaching- and research-oriented platform for **rigorous A/B testing**.  
    All experiments are driven by **synthetic data generators** with known ground truth,
    enabling proper evaluation of Type I error, power, and calibration.
    """
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Synthetic Data Lab",
        "2. Power Analysis",
        "3. Frequentist A/B Test",
        "4. Bayesian A/B Test",
        "5. Sequential Testing",
        "6. Multiple Testing",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Built for statisticians & applied mathematicians. "
    "Focus: correct inference under optional stopping and multiple comparisons."
)


# =============================================================================
# PAGE 1: Synthetic Data Lab
# =============================================================================
if page == "1. Synthetic Data Lab":
    st.header("Synthetic Data Generators")
    st.markdown(
        "Generate controlled A/B experiments with **known ground truth**. "
        "This is the foundation for all subsequent statistical evaluations."
    )

    col1, col2 = st.columns(2)
    with col1:
        metric_type = st.selectbox("Metric type", ["Binary (conversion)", "Continuous (Normal)"])
        n_per_group = st.slider("Sample size per group", 100, 10000, 2000, step=100)
        seed = st.number_input("Random seed", value=42, step=1)

    with col2:
        if metric_type.startswith("Binary"):
            p_control = st.slider("Control rate (p₀)", 0.01, 0.50, 0.10, 0.01)
            uplift = st.slider("Absolute uplift", -0.10, 0.20, 0.02, 0.005)
            p_treatment = float(np.clip(p_control + uplift, 0.0, 1.0))
            st.metric("Treatment rate (p₁)", f"{p_treatment:.3f}")
        else:
            mu_control = st.number_input("Control mean (μ₀)", value=0.0)
            mu_treatment = st.number_input("Treatment mean (μ₁)", value=0.2)
            sigma = st.number_input("Common σ", value=1.0, min_value=0.1)

    if st.button("Generate Dataset", type="primary"):
        if metric_type.startswith("Binary"):
            df, truth = generate_bernoulli_ab(
                n_per_group, n_per_group, p_control, p_treatment, seed=int(seed)
            )
            st.session_state["ab_df"] = df
            st.session_state["ab_truth"] = truth
            st.session_state["metric_col"] = "conversion"
        else:
            df, truth = generate_normal_ab(
                n_per_group, n_per_group, mu_control, mu_treatment, sigma, seed=int(seed)
            )
            st.session_state["ab_df"] = df
            st.session_state["ab_truth"] = truth
            st.session_state["metric_col"] = "value"

        st.success("Dataset generated and stored in session.")

    if "ab_df" in st.session_state:
        df = st.session_state["ab_df"]
        truth = st.session_state["ab_truth"]
        metric_col = st.session_state["metric_col"]

        st.subheader("Ground Truth")
        st.json(truth)

        st.subheader("Sample Preview")
        st.dataframe(df.head(10), use_container_width=True)

        summary = summarize_ab_result(df, metric_col)
        st.subheader("Empirical Summary")
        st.json(summary)

        fig = px.histogram(
            df, x=metric_col, color="group", barmode="overlay",
            opacity=0.65, nbins=40, title="Distribution by Group"
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 2: Power Analysis
# =============================================================================
elif page == "2. Power Analysis":
    st.header("Power Analysis & Sample Size Planning")

    col1, col2, col3 = st.columns(3)
    with col1:
        p1 = st.slider("Baseline rate (p₀)", 0.01, 0.5, 0.10, 0.01)
        uplift = st.slider("Minimum detectable effect (absolute)", 0.005, 0.15, 0.02, 0.005)
        p2 = float(np.clip(p1 + uplift, 0, 1))
    with col2:
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        target_power = st.slider("Target power (1-β)", 0.70, 0.95, 0.80, 0.05)
    with col3:
        st.metric("Treatment rate under H₁", f"{p2:.3f}")
        st.metric("Cohen's h", f"{effect_size_cohen_h(p1, p2):.3f}")

    n_required = sample_size_proportion(p1, p2, alpha=alpha, power=target_power)
    st.success(f"**Required sample size per group:** {n_required:,}")

    # Power curve
    n_range = np.unique(np.linspace(100, max(n_required * 2, 5000), 40).astype(int))
    powers = power_curve_proportion(p1, uplift, n_range, alpha=alpha)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_range, y=powers, mode="lines+markers", name="Power"))
    fig.add_hline(y=target_power, line_dash="dash", annotation_text=f"Target {target_power:.0%}")
    fig.add_vline(x=n_required, line_dash="dash", annotation_text=f"n = {n_required}")
    fig.update_layout(
        title="Power Curve (Two-Proportion Test)",
        xaxis_title="Sample size per group",
        yaxis_title="Statistical Power",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        **Interpretation**  
        To detect an absolute uplift of **{uplift:.1%}** from a baseline of **{p1:.1%}** 
        with **{target_power:.0%}** power at **α = {alpha}**, you need approximately 
        **{n_required:,}** observations per group.
        """
    )


# =============================================================================
# PAGE 3: Frequentist A/B Test
# =============================================================================
elif page == "3. Frequentist A/B Test":
    st.header("Frequentist Two-Proportion Test")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("n per group", 200, 20000, 2000, 100)
        p_c = st.slider("True p_control", 0.01, 0.4, 0.10, 0.01, key="freq_pc")
        uplift = st.slider("True absolute uplift", -0.05, 0.15, 0.025, 0.005, key="freq_up")
    with col2:
        alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="freq_alpha")
        seed = st.number_input("Seed", 1, 9999, 42, key="freq_seed")

    p_t = float(np.clip(p_c + uplift, 0, 1))
    df, truth = generate_bernoulli_ab(n, n, p_c, p_t, seed=int(seed))

    g = df.groupby("group")["conversion"]
    s_c, n_c = int(g.get_group("control").sum()), int(g.get_group("control").count())
    s_t, n_t = int(g.get_group("treatment").sum()), int(g.get_group("treatment").count())

    z, p_val = two_proportion_ztest(s_c, n_c, s_t, n_t)
    p_hat_c, p_hat_t = s_c / n_c, s_t / n_t

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control rate", f"{p_hat_c:.3%}")
    c2.metric("Treatment rate", f"{p_hat_t:.3%}")
    c3.metric("Absolute diff", f"{p_hat_t - p_hat_c:+.3%}")
    c4.metric("p-value", f"{p_val:.4f}", delta="Significant" if p_val < alpha else "Not significant")

    st.write(f"**z-statistic:** {z:.3f}")
    st.write(f"**Ground truth uplift:** {truth['absolute_uplift']:+.3%}")

    # Simple CI
    se = np.sqrt(p_hat_c*(1-p_hat_c)/n_c + p_hat_t*(1-p_hat_t)/n_t)
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_low = (p_hat_t - p_hat_c) - z_crit * se
    ci_high = (p_hat_t - p_hat_c) + z_crit * se
    st.write(f"**{100*(1-alpha):.0f}% CI for difference:** [{ci_low:.4f}, {ci_high:.4f}]")


# =============================================================================
# PAGE 4: Bayesian A/B Test
# =============================================================================
elif page == "4. Bayesian A/B Test":
    st.header("Bayesian A/B Testing (Beta-Binomial)")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("n per group", 200, 20000, 1500, 100, key="bayes_n")
        p_c = st.slider("True p_control", 0.01, 0.4, 0.10, 0.01, key="bayes_pc")
        uplift = st.slider("True absolute uplift", -0.05, 0.15, 0.03, 0.005, key="bayes_up")
    with col2:
        prior_strength = st.slider("Prior strength (α=β)", 0.5, 20.0, 1.0, 0.5)
        decision_threshold = st.slider("Decision threshold P(pₜ > p₀)", 0.80, 0.99, 0.95, 0.01)
        seed = st.number_input("Seed", 1, 9999, 123, key="bayes_seed")

    p_t = float(np.clip(p_c + uplift, 0, 1))
    df, truth = generate_bernoulli_ab(n, n, p_c, p_t, seed=int(seed))
    g = df.groupby("group")["conversion"]
    s_c, n_c = int(g.get_group("control").sum()), int(g.get_group("control").count())
    s_t, n_t = int(g.get_group("treatment").sum()), int(g.get_group("treatment").count())

    result = bayesian_ab_test(
        s_c, n_c, s_t, n_t,
        prior_alpha=prior_strength,
        prior_beta=prior_strength,
        n_sim=40_000,
        seed=int(seed),
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("P(treatment > control)", f"{result.prob_treatment_better:.1%}")
    c2.metric("E[loss] if choose treatment", f"{result.expected_loss_choose_treatment:.5f}")
    c3.metric("E[loss] if choose control", f"{result.expected_loss_choose_control:.5f}")

    decision = "Choose Treatment" if result.prob_treatment_better >= decision_threshold else "Keep Control / Continue"
    st.success(f"**Decision (threshold = {decision_threshold:.0%}): {decision}**")

    st.markdown("#### Posterior Summaries")
    st.write(
        f"- Control: mean = {result.mean_control:.4f}, "
        f"95% CrI = [{result.ci_control[0]:.4f}, {result.ci_control[1]:.4f}]"
    )
    st.write(
        f"- Treatment: mean = {result.mean_treatment:.4f}, "
        f"95% CrI = [{result.ci_treatment[0]:.4f}, {result.ci_treatment[1]:.4f}]"
    )

    # Posterior density plot
    x = np.linspace(0, max(0.4, p_t + 0.1), 500)
    dens_c = stats.beta.pdf(x, result.alpha_c, result.beta_c)
    dens_t = stats.beta.pdf(x, result.alpha_t, result.beta_t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dens_c, name="Control posterior", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=x, y=dens_t, name="Treatment posterior", fill="tozeroy"))
    fig.update_layout(title="Posterior Distributions of Conversion Rates", xaxis_title="Rate", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 5: Sequential Testing
# =============================================================================
elif page == "5. Sequential Testing":
    st.header("Sequential Testing & Optional Stopping")
    st.markdown(
        "Compare classical p-values (invalid under optional stopping) "
        "with an approximate **always-valid p-value**."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_total = st.slider("Total n per group", 1000, 10000, 4000, 500)
        batch = st.slider("Batch size (look every)", 50, 500, 200, 50)
        p_c = st.slider("True p_control", 0.05, 0.3, 0.10, 0.01, key="seq_pc")
    with col2:
        uplift = st.slider("True absolute uplift", 0.0, 0.08, 0.02, 0.005, key="seq_up")
        alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01, key="seq_alpha")
        seed = st.number_input("Seed", 1, 9999, 7, key="seq_seed")

    p_t = float(np.clip(p_c + uplift, 0, 1))
    batches, truth = generate_sequential_stream(
        n_total_per_group=n_total,
        p_control=p_c,
        p_treatment=p_t,
        batch_size=batch,
        seed=int(seed),
    )

    results = sequential_test_proportion(batches, alpha=alpha)

    df_seq = pd.DataFrame(
        {
            "look": results["n_looks"],
            "n_per_group": results["n_control"],
            "p_classic": results["p_classic"],
            "p_always_valid": results["p_always_valid"],
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_seq["n_per_group"], y=df_seq["p_classic"], name="Classic p-value", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df_seq["n_per_group"], y=df_seq["p_always_valid"], name="Always-valid p-value", mode="lines+markers"))
    fig.add_hline(y=alpha, line_dash="dash", annotation_text=f"α = {alpha}")
    fig.update_layout(
        title="p-values Across Sequential Looks",
        xaxis_title="Sample size per group",
        yaxis_title="p-value",
        yaxis_type="log",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**First rejection (classic):** look {results['first_reject_classic']}  \n"
        f"**First rejection (always-valid):** look {results['first_reject_always_valid']}"
    )
    st.info(
        "Classic p-values are not valid under repeated looks / optional stopping. "
        "Always-valid p-values provide a conservative bound that controls Type I error even with continuous monitoring."
    )


# =============================================================================
# PAGE 6: Multiple Testing
# =============================================================================
elif page == "6. Multiple Testing":
    st.header("Multiple Testing Correction")
    st.markdown("Simulate multi-variant experiments and apply Bonferroni / Holm / Benjamini-Hochberg.")

    n_per = st.slider("n per variant", 300, 3000, 800, 100)
    seed = st.number_input("Seed", 1, 9999, 99, key="mt_seed")
    method = st.selectbox("Correction method", ["fdr_bh", "holm", "bonferroni"])

    # Control + 4 treatments (one true effect)
    rates = [0.10, 0.102, 0.098, 0.135, 0.101]  # only treatment_3 has real uplift
    df, truth = generate_multi_variant(n_per_group=n_per, rates=rates, seed=int(seed))

    # Compute pairwise p-values vs control
    control = df[df["group"] == "control"]["conversion"]
    pvals = []
    labels = []
    for g in sorted(df["group"].unique()):
        if g == "control":
            continue
        treat = df[df["group"] == g]["conversion"]
        # two-proportion test
        s_c, n_c = int(control.sum()), len(control)
        s_t, n_t = int(treat.sum()), len(treat)
        _, p = two_proportion_ztest(s_c, n_c, s_t, n_t)
        pvals.append(p)
        labels.append(g)

    correction = apply_multiple_testing_correction(pvals, alpha=0.05, method=method)

    res_df = pd.DataFrame(
        {
            "variant": labels,
            "raw_p": correction["raw_pvalues"],
            "adjusted_p": correction["adjusted_pvalues"],
            "reject": correction["reject"],
            "true_rate": rates[1:],
        }
    )
    st.dataframe(res_df.style.format({"raw_p": "{:.4f}", "adjusted_p": "{:.4f}", "true_rate": "{:.3f}"}), use_container_width=True)
    st.write(f"**Number of rejections ({method}):** {correction['n_rejected']}")


# =============================================================================
# ABOUT
# =============================================================================
else:
    st.header("About this Platform")
    st.markdown(
        """
        ### Purpose
        This platform is designed for **statisticians, applied mathematicians, and quantitative researchers**
        who need to understand and teach correct experimental inference under:
        - Optional stopping / sequential monitoring
        - Multiple comparisons
        - Bayesian decision analysis

        ### Key Design Principles
        1. **Synthetic data first** — ground truth is known, so Type I error and power can be evaluated properly.
        2. **Transparent methods** — every procedure is implemented in clear, readable Python.
        3. **Teaching-oriented** — explanations and visualizations emphasize statistical concepts.

        ### Methods Implemented
        - Sample size & power analysis for two proportions
        - Classical two-proportion z-test + confidence intervals
        - Approximate always-valid p-values for sequential testing
        - Beta-Binomial Bayesian A/B testing with expected loss
        - Bonferroni, Holm, and Benjamini-Hochberg FDR control

        ### Tech Stack
        Python · NumPy · SciPy · Statsmodels · Streamlit · Plotly

        ---
        Built as a portfolio / research project demonstrating rigorous statistical practice.
        """
    )
