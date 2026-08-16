# Six-Notebook Learning Workflow

Run these notebooks in order:

1. `01_data_quality_eda.ipynb` — data quality, validation, EDA
2. `02_experiment_ate.ipynb` — randomized experiment and ATE
3. `03_feature_engineering_split.ipynb` — leakage audit, preprocessing, train/test split
4. `04_uplift_model_training.ipynb` — T-Learner, Class Transformation, Any Email, spend uplift
5. `05_uplift_evaluation.ipynb` — Qini, AUUC, Uplift@10/20/30, model comparison
6. `06_customer_prediction_policy.ipynb` — customer scores, recommendations, targeting depth, business impact

The notebooks use the project's `src/` modules and write intermediate/final outputs into `artifacts/`.
