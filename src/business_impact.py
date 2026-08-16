import numpy as np, pandas as pd
from .evaluation import evaluate_targeting_depth

def targeting_curve(pred, depths=None):
    if depths is None: depths=[.05,.10,.20,.30,.40,.50,1.0]
    rows=[]
    for d in depths:
        value,inc,actions=evaluate_targeting_depth(pred,d,"conversion")
        rv,inc_rev,_=evaluate_targeting_depth(pred,d,"spend")
        top=pred.sort_values("max_uplift",ascending=False).head(int(len(pred)*d))
        pos=top[top.max_uplift>0]
        active=actions!="No E-Mail"
        rows.append({"targeting_depth":d,"customers_targeted":int(active.sum()),
                     "mens_emails":int(((actions=="Mens E-Mail")).sum()),
                     "womens_emails":int(((actions=="Womens E-Mail")).sum()),
                     "no_email":int((actions=="No E-Mail").sum()),
                     "expected_incremental_conversions":inc*len(pred),
                     "expected_incremental_revenue":inc_rev*len(pred),
                     "average_uplift":float(pos.max_uplift.mean()) if len(pos) else 0})
    return pd.DataFrame(rows)
