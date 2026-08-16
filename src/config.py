from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hillstrom_email_marketing.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
PREDICTIONS_DIR = ARTIFACTS / "predictions"
METRICS_DIR = ARTIFACTS / "metrics"
FIGURES_DIR = ARTIFACTS / "figures"
METADATA_DIR = ARTIFACTS / "metadata"
RANDOM_SEED = 42
TEST_SIZE = 0.25
UPLIFT_THRESHOLD = 0.0
MODEL_PARAMS = {
    "n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31,
    "max_depth": -1, "subsample": 0.9, "colsample_bytree": 0.9,
    "reg_lambda": 1.0, "verbosity": -1
}
TREATMENTS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel", "history_segment"]
