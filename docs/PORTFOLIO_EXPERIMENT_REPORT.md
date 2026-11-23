# Randomized email uplift experiment report

**Dataset:** 64,000-customer Hillstrom MineThatData randomized email experiment.  
**Evidence source:** committed `artifacts/metrics/*.csv` outputs and the model card.  
**Interpretation:** evaluation-sample findings; not a guarantee of future campaign performance.

## Randomized average treatment effects

The committed ATE table reports the following absolute conversion-rate differences versus the no-email control:

| Treatment | Control conversion | Treatment conversion | Absolute lift | 95% interval |
|---|---:|---:|---:|---:|
| Men's Email | 0.573% | 1.253% | +0.681 pp | [+0.500, +0.861] pp |
| Women's Email | 0.573% | 0.884% | +0.311 pp | [+0.150, +0.472] pp |
| Any Email | 0.573% | 1.068% | +0.495 pp | [+0.355, +0.636] pp |

These are randomized arm-level effects, not individualized effects. The estimates support an average response to email in this experiment.

## Held-out uplift ranking

The committed model comparison artifact reports the best AUUC for Men's Email versus Control using Class Transformation:

| Metric | Result |
|---|---:|
| AUUC | 0.002054 |
| Qini | 0.004129 |
| Uplift@10% | 0.628% |
| Uplift@20% | 0.410% |
| Uplift@30% | 0.549% |

The Uplift@10% figure in the README is rounded from the artifact value. Uplift ranking metrics are evaluation-sample diagnostics; they do not by themselves show positive net campaign value.

## Policy decision caveat

The model card records an IPW estimated spend/customer of $1.127 for the model policy versus $1.310 for email-everyone, with a paired difference of -$0.183 and a 95% interval spanning zero. The current policy evidence therefore does **not** establish superiority to emailing everyone. The committed policy simulation CSV also contains expected outcomes under the project's policy estimator; these should not be described as realized revenue or causal business savings.

## Reproduce

```bash
python -m venv .venv
# Activate the environment
python -m pip install -r requirements.txt
python run_project.py
python -m pytest -q
streamlit run app/streamlit_app.py
```

The pipeline entry point is `run_project.py`, which delegates to `src.run_analysis.main`. The repository includes a GitHub Actions test workflow on pushes and pull requests to `main`. Preserve the randomization design and held-out evaluation split when modifying the analysis.

## Business interpretation

The strongest supported conclusion is that the randomized experiment shows a positive average conversion response to email, particularly the Men's Email arm. The uplift models attempt to rank heterogeneous response, but the current policy evaluation does not demonstrate that model-targeted contacts outperform emailing everyone. The next step is a preregistered prospective randomized comparison of the frozen targeting policy against the broad email policy, with measured contact costs and contribution margins.

## Limitations

- Individual treatment effects are not directly observed.
- Uplift ranking metrics are sample- and split-dependent.
- The data does not supply real campaign contact cost or contribution margin.
- Subgroup findings are exploratory unless multiplicity is controlled.
- No future campaign revenue or profitability guarantee is implied.
