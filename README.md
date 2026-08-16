<<<<<<< HEAD
# Causal Uplift Modeling for Email Marketing Optimization

## Overview

This portfolio project uses the Hillstrom MineThatData email experiment to answer a causal question:

> Which customers should receive Men's Email, Women's Email, or No Email to maximize incremental response?

Unlike a conventional conversion classifier, the system estimates **heterogeneous treatment effects** and turns them into a customer-level targeting policy.

## Dataset

The uploaded dataset contains **64,000 customers** and the experimental arms `Mens E-Mail`, `Womens E-Mail`, and `No E-Mail`. Pre-treatment features include recency, historical spend, gender-category purchase indicators, geography, acquisition channel, history segment, and newbie status. Outcomes are visit, conversion, and spend.

## Methodology

- Randomized experiment validation
- ATE for visit, conversion, and spend
- Men's Email vs Control
- Women's Email vs Control
- Any Email vs Control
- Two-Model / T-Learner uplift modeling with LightGBM
- Class-transformation uplift modeling
- Held-out Qini / AUUC evaluation
- Uplift@10%, @20%, @30%
- Customer-level treatment policy
- Held-out inverse-propensity-weighted policy evaluation
- Conditional spend uplift as a business-value signal

### Causal terminology

**ATE** is a population-level average effect.

**Individual Treatment Effect** is not directly observed. The system estimates a conditional treatment effect from customer features.

**Estimated heterogeneous treatment effect** is therefore the preferred interpretation of customer-level uplift scores.

## Headline results from the uploaded dataset

| Metric | Result |
|---|---:|
| Men's conversion lift | 0.68% |
| Women's conversion lift | 0.31% |
| Best uplift model | Class Transformation — Mens E-Mail vs Control |
| Best AUUC | 0.002054 |
| Best Qini | 0.004129 |
| Best Uplift@10% | 0.63% |
| Best Uplift@20% | 0.41% |
| Best Uplift@30% | 0.55% |
| Highest estimated-revenue targeting depth | 50% |
| Estimated incremental conversions at that depth | 41.9 |
| Estimated incremental revenue at that depth | $6,051 |

These policy numbers are estimates from the held-out randomized evaluation sample and should not be interpreted as guaranteed future campaign results.

## Architecture

```text
hillstrom-uplift-project/
├── app/
│   ├── streamlit_app.py
│   └── pages/
├── data/
│   └── hillstrom_email_marketing.csv
├── notebooks/
│   └── 01_full_analysis.ipynb
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── experiment_analysis.py
│   ├── feature_engineering.py
│   ├── uplift_models.py
│   ├── evaluation.py
│   ├── policy.py
│   ├── business_impact.py
│   └── run_analysis.py
├── artifacts/
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── metadata/
├── tests/
├── requirements.txt
├── run_project.py
└── README.md
```

## How to run

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py
streamlit run app/streamlit_app.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_project.py
streamlit run app/streamlit_app.py
```

Training is performed by `python run_project.py`. Streamlit loads the saved artifacts and does not retrain models at startup.

## GitHub deployment

```bash
git init
git add .
git commit -m "Build causal uplift modeling project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Do not replace `YOUR_REPOSITORY_URL` until you have created your repository.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the main file to `app/streamlit_app.py`.
5. Deploy.
6. The repository already contains trained artifacts, so the dashboard can start without retraining.

## Model artifacts

Saved under `artifacts/models/`:

- T-Learner treatment/control models
- Class-transformation models
- Spend uplift regressors
- preprocessing pipeline
- feature list

Saved under `artifacts/metrics/`:

- ATE results
- model comparison
- Qini curves
- targeting curve
- policy comparison

## Testing

Run:

```bash
pytest -q
```

## Limitations

- Randomization supports identification of population-level effects, but individual treatment effects remain estimated.
- Held-out policy evaluation has statistical uncertainty.
- Spend is highly zero-inflated.
- Campaign costs are not present, so this project does not calculate net profit.
- Production deployment can experience distribution shift and operational constraints.
- A targeting policy should be prospectively validated before being used in a live campaign.

## Portfolio value

This project demonstrates:

**Experimentation → Causal Inference → Heterogeneous Effects → Uplift Modeling → Optimization → Business Decision → Deployment**

It is intentionally different from a conventional binary conversion-classification portfolio project.
=======
# Bayesian & Sequential Experimentation Platform

A teaching- and research-oriented platform for **rigorous A/B testing and sequential experimentation**, built around **synthetic data generators** with known ground truth.

Ideal for statisticians, applied mathematicians, and quantitative researchers who need correct inference under optional stopping and multiple comparisons.

---

## Features

| Module | Capabilities |
|--------|--------------|
| **Synthetic Data Lab** | Bernoulli & Normal A/B generators, sequential streams, multi-variant data |
| **Power Analysis** | Sample size calculation, power curves, effect size (Cohen's h) |
| **Frequentist A/B** | Two-proportion z-test, confidence intervals |
| **Bayesian A/B** | Beta-Binomial model, P(treatment > control), expected loss, credible intervals |
| **Sequential Testing** | Classic vs always-valid p-values under repeated looks |
| **Multiple Testing** | Bonferroni, Holm, Benjamini-Hochberg FDR |

---

## Why Synthetic Data?

Real A/B datasets almost never have known true effects.  
With synthetic generators we can:

- Evaluate **Type I error** under optional stopping
- Measure **power** and calibration
- Demonstrate the difference between classic and always-valid p-values
- Run clean Monte-Carlo studies of Bayesian decision rules

---

## Quick Start

```bash
# 1. Clone / unzip
cd bayesian-sequential-experimentation

# 2. Create environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
bayesian-sequential-experimentation/
├── app.py                     # Streamlit front-end
├── requirements.txt
├── README.md
├── src/
│   ├── data/
│   │   └── generators.py      # Synthetic data generators (core)
│   ├── frequentist/
│   │   ├── power_analysis.py
│   │   ├── sequential.py
│   │   └── multiple_testing.py
│   ├── bayesian/
│   │   └── ab_testing.py      # Beta-Binomial + expected loss
│   └── utils/
│       └── metrics.py
├── notebooks/                 # Optional exploration
└── tests/
```

---

## Example Usage (Python API)

```python
from src.data.generators import generate_bernoulli_ab, generate_sequential_stream
from src.frequentist.power_analysis import sample_size_proportion, power_proportion
from src.bayesian.ab_testing import bayesian_ab_test
from src.frequentist.sequential import sequential_test_proportion

# Sample size planning
n = sample_size_proportion(p1=0.10, p2=0.12, alpha=0.05, power=0.8)
print(f"Required n per group: {n}")

# Generate controlled experiment
df, truth = generate_bernoulli_ab(n_control=2000, n_treatment=2000,
                                  p_control=0.10, p_treatment=0.125, seed=42)

# Bayesian analysis
g = df.groupby("group")["conversion"]
res = bayesian_ab_test(
    success_c=int(g.get_group("control").sum()),
    n_c=len(g.get_group("control")),
    success_t=int(g.get_group("treatment").sum()),
    n_t=len(g.get_group("treatment")),
)
print(f"P(treatment > control) = {res.prob_treatment_better:.3f}")
print(f"Expected loss (choose treatment) = {res.expected_loss_choose_treatment:.5f}")
```

---

## Statistical Methods Summary

### Power & Sample Size
Closed-form approximation for two-proportion z-test.

### Sequential Testing
Approximate always-valid p-value based on mixture / information-adjusted bounds.  
Classic p-values are shown for contrast (they do **not** control Type I error under optional stopping).

### Bayesian A/B
Independent Beta(α, β) priors → Beta posteriors.  
Decision quantities:
- Posterior probability P(pₜ > p₀ | data)
- Expected loss of choosing each arm

### Multiple Testing
- Bonferroni
- Holm
- Benjamini-Hochberg (FDR)

---

## Extending the Platform

Ideas for further development:
- Exact always-valid p-values / mSPRT
- CUPED / variance reduction
- Hierarchical Bayesian models for multi-segment experiments
- Sequential Bayesian monitoring
- Simulation engine for Type I error & power curves under optional stopping

---

## License

MIT License – free for research, teaching, and portfolio use.

---

**Author:** Harshith Devaraja  
M.Sc. Applied Mathematics & Computing  
Focus: Statistical inference, causal & experimental methods
>>>>>>> 75a6bdf986070639ea184490b34c5168f8484e6b
