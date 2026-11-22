# Model Card — Causal Uplift Modelling for Email Marketing

## Intended use

Research and decision support for estimating average treatment effects and exploring heterogeneous response to email treatments in the randomized Hillstrom MineThatData experiment. The intended decision is whether a targeting policy warrants a prospective randomized test.

## Data

- 64,000 randomized customer observations.
- Treatment arms: Men's Email, Women's Email and No Email.
- Pre-treatment customer attributes include recency, historical spend, category purchase indicators, geography, acquisition channel and customer status.
- Outcomes include visit, conversion and spend.

## Methods

- Randomized-arm descriptive analysis and average treatment effect estimation.
- Two-model / T-Learner and class-transformation uplift approaches.
- Held-out Qini, AUUC and uplift-at-capacity diagnostics.
- Policy evaluation against email-everyone and email-nobody comparators.
- Budget sensitivity scenarios with user-specified contact costs and contribution margins.

## Evaluation snapshot

The README reports the best Men's Email versus Control class-transformation comparison with AUUC 0.002054, Qini 0.004129 and Uplift@10% 0.63%. These are ranking diagnostics on the evaluation sample. The project must not translate them into guaranteed incremental revenue.

## Critical limitations

- Individual treatment effects are not observed; uplift scores are estimated conditional-effect rankings.
- The reported model policy has not demonstrated superiority to emailing everyone. In the README snapshot, IPW estimated spend per customer is $1.127 for the model policy versus $1.310 for email-everyone; the paired difference is -$0.183 with a 95% CI spanning zero.
- The dataset contains no contact cost or contribution margin. Budget outputs are assumption-driven scenarios.
- Subgroup comparisons are exploratory unless multiplicity is addressed.
- Prospective randomized validation is required before deployment.

## Business interpretation

This project asks whether customer targeting can improve on a broad email policy by identifying customers whose behavior is more responsive to treatment. The current evidence supports studying the randomized average effect; it does not establish that the uplift model improves campaign value. The next business step is a preregistered randomized pilot comparing the frozen policy with email-everyone under measured campaign costs and margins.

## Reproducibility

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python src/run_analysis.py
pytest -q
streamlit run app/streamlit_app.py
```

Check the README for the dataset acquisition path and any required environment variables.