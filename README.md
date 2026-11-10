# Causal Uplift Modelling for Email Marketing Optimization

A decision-science project using the **64,000-customer Hillstrom MineThatData randomized email experiment** to estimate heterogeneous treatment effects and evaluate customer-targeting policies.

The central question is not simply who is likely to convert. It is **which customers are more likely to respond because they receive a particular email treatment**.

## Project scope

- Compare Men's Email, Women's Email and No Email experimental arms.
- Estimate average treatment effects (ATE) for visit, conversion and spend outcomes.
- Compare two-model / T-Learner and class-transformation uplift approaches using LightGBM.
- Evaluate ranking quality on held-out data with Qini, AUUC and Uplift@10%, @20% and @30%.
- Compare targeting depths and evaluate policy value on the randomized evaluation sample.
- Separate measured experimental results from scenario assumptions and future expectations.

## Dataset

The project uses 64,000 customers from the Hillstrom MineThatData email experiment. The three randomized arms are `Mens E-Mail`, `Womens E-Mail` and `No E-Mail`.

Pre-treatment attributes include recency, historical spend, category purchase indicators, geography, acquisition channel, history segment and customer status. Outcomes include visit, conversion and spend.

## Methodology

1. Validate the randomized experiment structure and inspect pre-treatment covariates.
2. Estimate population-level treatment effects for the email arms.
3. Train uplift models to estimate conditional treatment-effect rankings.
4. Compare models using held-out Qini/AUUC and uplift-at-depth metrics.
5. Evaluate targeting policies against the randomized evaluation data.
6. Report uncertainty and limitations alongside the results.

**Causal terminology:** ATE is a population-level average effect. Individual treatment effects are not directly observed; customer-level uplift scores are estimated heterogeneous treatment effects and should be interpreted as rankings with uncertainty.

## Reported held-out results

| Metric | Result |
|---|---:|
| Best comparison | Men's Email vs Control |
| Best model | Class Transformation |
| Best AUUC | 0.002054 |
| Best Qini | 0.004129 |
| Best Uplift@10% | 0.63% |
| Best Uplift@20% | 0.41% |
| Best Uplift@30% | 0.55% |

These are evaluation-sample metrics from the saved model comparison artifact. Policy simulations are estimates from the held-out randomized sample, not guaranteed future campaign outcomes.

## Architecture

```text
Hillstrom randomized experiment
            |
            v
Data validation and feature preparation
            |
            v
ATE analysis by treatment comparison
            |
            v
T-Learner / Class-Transformation models
            |
            v
Held-out Qini, AUUC and uplift-at-depth
            |
            v
Targeting-policy evaluation
            |
            v
Streamlit analysis and reporting
```

## Repository structure

```text
app/                 Streamlit application and pages
src/                 Data preparation, experiment analysis,
                     uplift models, evaluation and policy logic
artifacts/models/    Saved treatment and uplift models
artifacts/metrics/   ATE, model comparison and policy metrics
artifacts/figures/   Evaluation and targeting plots
tests/               Automated tests
run_project.py       Reproducible analysis entry point
```

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python run_project.py
streamlit run app/streamlit_app.py
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`. On Linux/macOS, use `source .venv/bin/activate`.

## Limitations and interpretation

- Individual treatment effects are estimated, not directly observed.
- Uplift ranking quality does not by itself establish profitability.
- Campaign costs are not present in the dataset; the project does not claim realised net profit.
- Spend is zero-inflated, and policy estimates carry sampling uncertainty.
- Distribution shift and campaign execution effects require prospective validation.
- A targeting policy should be tested in a new randomized experiment before operational use.

## What this project demonstrates

**Randomized experiments → statistical inference → heterogeneous treatment effects → uplift evaluation → targeting-policy analysis.**

The emphasis is on causal reasoning and evaluation discipline rather than treating uplift as ordinary conversion classification.

---

**Author:** Harshith Devaraja  
**Focus:** Statistical inference · Causal modelling · Decision science
