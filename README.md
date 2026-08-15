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
