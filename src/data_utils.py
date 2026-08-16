import json
import pandas as pd
import numpy as np
from pathlib import Path
from .config import DATA_PATH, METADATA_DIR, TREATMENTS

EXPECTED = ["recency","history_segment","history","mens","womens","zip_code","newbie","channel","segment","visit","conversion","spend"]

def normalize_columns(df):
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in d.columns]
    aliases = {"treatment":"segment","arm":"segment","treatment_group":"segment",
               "historysegment":"history_segment","zip":"zip_code"}
    for old,new in aliases.items():
        if old in d.columns and new not in d.columns: d = d.rename(columns={old:new})
    return d

def load_data(path=DATA_PATH):
    df = normalize_columns(pd.read_csv(path))
    return df

def validate_data(df):
    report = {"shape": list(df.shape), "columns": list(df.columns),
              "missing": df.isna().sum().to_dict(), "duplicates": int(df.duplicated().sum()),
              "invalid_binary": {}, "treatment_counts": {},
              "expected_columns_present": {c: c in df.columns for c in EXPECTED}}
    for c in ["mens","womens","newbie","visit","conversion"]:
        if c in df:
            vals = sorted(pd.Series(df[c].dropna().unique()).tolist())
            report["invalid_binary"][c] = [x for x in vals if x not in [0,1]]
    if "segment" in df: report["treatment_counts"] = df["segment"].value_counts().to_dict()
    report["valid_treatments"] = set(report["treatment_counts"]).issubset(set(TREATMENTS)) and set(TREATMENTS).issubset(set(report["treatment_counts"]))
    return report

def prepare(df):
    d = df.copy()
    for c in ["recency","history","mens","womens","newbie","visit","conversion","spend"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["history_segment","zip_code","channel","segment"]:
        d[c] = d[c].astype("string").fillna("Unknown")
    d = d.dropna(subset=["segment","conversion","visit","spend"]).reset_index(drop=True)
    return d

def save_validation(report):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR/"data_quality.json").write_text(json.dumps(report, indent=2, default=str))
