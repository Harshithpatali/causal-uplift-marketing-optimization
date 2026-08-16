import numpy as np
import pandas as pd
from scipy import stats

def group_summary(df):
    return df.groupby("segment").agg(
        customers=("segment","size"), visit_rate=("visit","mean"),
        conversion_rate=("conversion","mean"), avg_spend=("spend","mean")
    ).reset_index()

def ate_table(df, outcome):
    control = df[df.segment=="No E-Mail"][outcome].dropna()
    rows=[]
    for t in ["Mens E-Mail","Womens E-Mail"]:
        tr = df[df.segment==t][outcome].dropna()
        diff = tr.mean()-control.mean()
        se = np.sqrt(tr.var(ddof=1)/len(tr)+control.var(ddof=1)/len(control))
        lo,hi = diff-1.96*se,diff+1.96*se
        p=2*stats.norm.sf(abs(diff/se)) if se else np.nan
        rel = diff/control.mean() if control.mean()!=0 else np.nan
        rows.append({"treatment":t,"outcome":outcome,"treatment_mean":tr.mean(),
                     "control_mean":control.mean(),"absolute_lift":diff,
                     "relative_lift":rel,"ci_low":lo,"ci_high":hi,"p_value":p})
    anyt=df[df.segment!="No E-Mail"][outcome].dropna()
    diff=anyt.mean()-control.mean()
    se=np.sqrt(anyt.var(ddof=1)/len(anyt)+control.var(ddof=1)/len(control))
    rows.append({"treatment":"Any Email","outcome":outcome,"treatment_mean":anyt.mean(),
                 "control_mean":control.mean(),"absolute_lift":diff,
                 "relative_lift":diff/control.mean() if control.mean()!=0 else np.nan,
                 "ci_low":diff-1.96*se,"ci_high":diff+1.96*se,
                 "p_value":2*stats.norm.sf(abs(diff/se)) if se else np.nan})
    return pd.DataFrame(rows)

def all_ates(df):
    return pd.concat([ate_table(df,o) for o in ["visit","conversion","spend"]], ignore_index=True)
