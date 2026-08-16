import numpy as np
import pandas as pd

def uplift_curve(y, treatment, uplift, arm):
    y=np.asarray(y); treatment=np.asarray(treatment); uplift=np.asarray(uplift)
    order=np.argsort(-uplift); y=y[order]; t=treatment[order]
    p=(t==arm).mean(); c=(t=="No E-Mail").mean()
    cum=np.arange(1,len(y)+1)
    inc=(np.cumsum(y*(t==arm))/max(p,1e-9)-np.cumsum(y*(t=="No E-Mail"))/max(c,1e-9))/len(y)
    return pd.DataFrame({"fraction":cum/len(y),"incremental_response":inc})

def uplift_metrics(y, treatment, uplift, arm):
    curve=uplift_curve(y,treatment,uplift,arm)
    auuc=np.trapz(curve.incremental_response, curve.fraction)
    vals={}
    order=np.argsort(-np.asarray(uplift))
    for k in [0.10,0.20,0.30]:
        n=max(1,int(len(y)*k)); ids=order[:n]
        yy=np.asarray(y)[ids]; tt=np.asarray(treatment)[ids]
        tr=tt==arm; ct=tt=="No E-Mail"
        vals[f"uplift_at_{int(k*100)}"]=(yy[tr].mean() if tr.any() else 0)-(yy[ct].mean() if ct.any() else 0)
    return {"auuc":float(auuc),"qini":float(curve.incremental_response.max()),**{k:float(v) for k,v in vals.items()}},curve

def _ipw_policy_value(df, actions, outcome):
    probs={a:(df.segment==a).mean() for a in ["Mens E-Mail","Womens E-Mail","No E-Mail"]}
    total=0.0
    for _,r in df.iterrows():
        a=actions.loc[r.row_id]
        if r.segment==a: total += r[outcome]/max(probs[a],1e-9)
    return total/len(df)

def evaluate_targeting_depth(df, depth, outcome="conversion"):
    d=df.sort_values("max_uplift",ascending=False).copy()
    n=int(len(d)*depth)
    actions=pd.Series("No E-Mail",index=d.row_id)
    top=d.head(n)
    actions.loc[top.row_id]=np.where(top.max_uplift>0,top.recommended_action,"No E-Mail")
    value=_ipw_policy_value(d,actions,outcome)
    base=d[d.segment=="No E-Mail"][outcome].mean()
    return value, value-base, actions

def policy_table(df, depth=0.2):
    d=df.copy().reset_index(drop=True)
    d["row_id"]=np.arange(len(d))
    order=d.sort_values("max_uplift",ascending=False)
    n=int(len(d)*depth); top_ids=set(order.head(n).row_id)
    actions_dict={}
    personalized=np.where((d.row_id.isin(top_ids))&(d.max_uplift>0),d.recommended_action,"No E-Mail")
    policies={
        "Personalized":personalized,
        "Mens-only":np.where(d.row_id.isin(top_ids),"Mens E-Mail","No E-Mail"),
        "Womens-only":np.where(d.row_id.isin(top_ids),"Womens E-Mail","No E-Mail"),
        "Any Email":np.where(d.row_id.isin(top_ids),np.where(d.uplift_mens>=d.uplift_womens,"Mens E-Mail","Womens E-Mail"),"No E-Mail"),
        "Random":np.where(d.row_id.isin(top_ids),np.where(np.arange(len(d))%2==0,"Mens E-Mail","Womens E-Mail"),"No E-Mail"),
        "No Email":np.array(["No E-Mail"]*len(d))
    }
    rows=[]
    base_conv=d[d.segment=="No E-Mail"].conversion.mean()
    base_rev=d[d.segment=="No E-Mail"].spend.mean()
    for name,acts in policies.items():
        actions=pd.Series(acts,index=d.row_id)
        cv=_ipw_policy_value(d,actions,"conversion"); rv=_ipw_policy_value(d,actions,"spend")
        rows.append({"policy":name,"targeted":int((actions!="No E-Mail").sum()),
                     "expected_conversion_rate":cv,
                     "incremental_conversions":(cv-base_conv)*len(d),
                     "expected_revenue":rv,
                     "incremental_revenue":(rv-base_rev)*len(d)})
    return pd.DataFrame(rows)
