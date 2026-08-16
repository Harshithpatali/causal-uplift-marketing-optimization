import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .config import FEATURES

NUMERIC = ["recency","history","mens","womens","newbie"]
CATEGORICAL = ["zip_code","channel","history_segment"]

def make_preprocessor():
    return ColumnTransformer([
        ("num","passthrough",NUMERIC),
        ("cat",OneHotEncoder(handle_unknown="ignore", sparse_output=False),CATEGORICAL)
    ])

def fit_transform(preprocessor, df):
    X = preprocessor.fit_transform(df[FEATURES])
    return X

def transform(preprocessor, df):
    return preprocessor.transform(df[FEATURES])
