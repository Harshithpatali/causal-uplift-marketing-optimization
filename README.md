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
