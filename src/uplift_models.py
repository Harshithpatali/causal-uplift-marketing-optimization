import numpy as np
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
from .config import MODEL_PARAMS, RANDOM_SEED, MODELS_DIR

def _classifier():
    params=dict(MODEL_PARAMS); params["random_state"]=RANDOM_SEED
    return LGBMClassifier(**params)

def train_tlearner(X, treatment, y, arm):
    tr=treatment==arm; ct=treatment=="No E-Mail"
    mt,mc=_classifier(),_classifier()
    mt.fit(X[tr],y[tr]); mc.fit(X[ct],y[ct])
    return mt,mc

def predict_tlearner(mt,mc,X):
    return mt.predict_proba(X)[:,1],mc.predict_proba(X)[:,1]

def train_class_transformation(X,treatment,y,arm):
    mask=(treatment==arm)|(treatment=="No E-Mail")
    xb=X[mask]; tb=(treatment[mask]==arm).astype(int); yb=y[mask].astype(float)
    p=float(tb.mean())
    transformed=yb*(tb-p)/(p*(1-p))
    params=dict(MODEL_PARAMS); params["random_state"]=RANDOM_SEED
    model=LGBMRegressor(**params)
    model.fit(xb,transformed)
    return model,p

def save_model(obj,name):
    path=MODELS_DIR/f"{name}.joblib"; path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(obj,path); return path
