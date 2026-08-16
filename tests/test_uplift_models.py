import joblib, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_artifacts_load():
    pre=joblib.load(ROOT/"artifacts/models/preprocessor.joblib")
    mt=joblib.load(ROOT/"artifacts/models/tlearner_mens_treatment.joblib")
    mc=joblib.load(ROOT/"artifacts/models/tlearner_mens_control.joblib")
    assert pre is not None and mt is not None and mc is not None
def test_predictions_exist():
    p=pd.read_csv(ROOT/"artifacts/predictions/customer_predictions.csv")
    assert len(p)==64000
    assert {"uplift_mens","uplift_womens","recommended_action"}.issubset(p.columns)
