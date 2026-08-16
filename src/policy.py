import numpy as np
import pandas as pd

def recommend(uplift_mens, uplift_womens, threshold=0.0):
    if uplift_mens > uplift_womens and uplift_mens > threshold: return "Mens E-Mail"
    if uplift_womens > uplift_mens and uplift_womens > threshold: return "Womens E-Mail"
    return "No E-Mail"

def apply_policy(df, threshold=0.0):
    d=df.copy()
    d["max_uplift"]=d[["uplift_mens","uplift_womens"]].max(axis=1)
    d["recommended_action"]=[recommend(a,b,threshold) for a,b in zip(d.uplift_mens,d.uplift_womens)]
    return d
