import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .config import *
from .data_utils import load_data, prepare, validate_data, save_validation
from .feature_engineering import make_preprocessor
from .experiment_analysis import all_ates, group_summary
from .uplift_models import train_tlearner, predict_tlearner, train_class_transformation, save_model
from .evaluation import uplift_metrics
from .policy import apply_policy
from .business_impact import targeting_curve

def main():
    for d in [MODELS_DIR,PREDICTIONS_DIR,METRICS_DIR,FIGURES_DIR,METADATA_DIR]: d.mkdir(parents=True,exist_ok=True)
    raw=load_data(DATA_PATH); report=validate_data(raw); save_validation(report)
    df=prepare(raw)
    a=all_ates(df); a.to_csv(METRICS_DIR/"ate_results.csv",index=False)
    group_summary(df).to_csv(METRICS_DIR/"group_summary.csv",index=False)

    idx_train,idx_test=train_test_split(np.arange(len(df)),test_size=TEST_SIZE,random_state=RANDOM_SEED,stratify=df.segment)
    train=df.iloc[idx_train].reset_index(drop=True); test=df.iloc[idx_test].reset_index(drop=True)
    pre=make_preprocessor(); Xtr=pre.fit_transform(train); Xte=pre.transform(test); Xall=pre.transform(df)
    joblib.dump(pre,MODELS_DIR/"preprocessor.joblib")
    joblib.dump(FEATURES,MODELS_DIR/"feature_columns.joblib")

    ytr=train.conversion.values; yte=test.conversion.values; tall=train.segment.values; ttest=test.segment.values
    results=[]; pred=pd.DataFrame({"row_id":df.index,"segment":df.segment,"visit":df.visit,"conversion":df.conversion,"spend":df.spend})
    for arm,slug in [("Mens E-Mail","mens"),("Womens E-Mail","womens")]:
        mt,mc=train_tlearner(Xtr,tall,ytr,arm)
        pt,pc=predict_tlearner(mt,mc,Xte); full_t,full_c=predict_tlearner(mt,mc,Xall)
        save_model(mt,f"tlearner_{slug}_treatment"); save_model(mc,f"tlearner_{slug}_control")
        ct,p=train_class_transformation(Xtr,tall,ytr,arm); save_model((ct,p),f"class_transform_{slug}")
        cu=ct.predict(Xte); full_cu=ct.predict(Xall)
        metrics,curve=uplift_metrics(yte,ttest,pt-pc,arm)
        metrics_ct,_=uplift_metrics(yte,ttest,cu,arm)
        results.append({"comparison":f"{arm} vs Control","model":"Two-Model","arm":arm,**metrics})
        results.append({"comparison":f"{arm} vs Control","model":"Class Transformation","arm":arm,**metrics_ct})
        pred[f"pred_conversion_{slug}"]=full_t; pred[f"pred_conversion_control_{slug}"]=full_c
        pred[f"uplift_{slug}"]=full_t-full_c; pred[f"uplift_{slug}_class_transform"]=full_cu
        curve.to_csv(METRICS_DIR/f"qini_{slug}_two_model.csv",index=False)

    # Any-email comparison uses a separate binary model.
    train_any=train.copy(); test_any=test.copy()
    train_any_t=np.where(train_any.segment=="No E-Mail","No E-Mail","Any Email")
    test_any_t=np.where(test_any.segment=="No E-Mail","No E-Mail","Any Email")
    mt,mc=train_tlearner(Xtr,train_any_t,ytr,"Any Email")
    pt,pc=predict_tlearner(mt,mc,Xte); full_t,full_c=predict_tlearner(mt,mc,Xall)
    save_model(mt,"tlearner_any_treatment"); save_model(mc,"tlearner_any_control")
    ct,p=train_class_transformation(Xtr,train_any_t,ytr,"Any Email"); save_model((ct,p),"class_transform_any")
    metrics,curve=uplift_metrics(yte,test_any_t,pt-pc,"Any Email")
    results.append({"comparison":"Any Email vs Control","model":"Two-Model","arm":"Any Email",**metrics})
    pred["pred_conversion_any"]=full_t; pred["pred_conversion_control_any"]=full_c; pred["uplift_any_email"]=full_t-full_c
    curve.to_csv(METRICS_DIR/"qini_any_two_model.csv",index=False)

    pred=apply_policy(pred,UPLIFT_THRESHOLD)
    # Attach source features for scorer/export.
    for c in FEATURES: pred[c]=df[c].values
    pred.to_csv(PREDICTIONS_DIR/"customer_predictions.csv",index=False)
    pd.DataFrame(results).to_csv(METRICS_DIR/"model_comparison.csv",index=False)

    # Spend uplift: regress expected spend separately for treatment/control.
    from lightgbm import LGBMRegressor
    params=dict(MODEL_PARAMS); params["random_state"]=RANDOM_SEED
    spend_results=[]
    for arm,slug in [("Mens E-Mail","mens"),("Womens E-Mail","womens"),("Any Email","any")]:
        if arm=="Any Email": ta=np.where(train.segment=="No E-Mail","No E-Mail","Any Email")
        else: ta=train.segment.values
        maskt=ta==arm; maskc=ta=="No E-Mail"
        rm_t=LGBMRegressor(**params); rm_c=LGBMRegressor(**params)
        rm_t.fit(Xtr[maskt],train.spend.values[maskt]); rm_c.fit(Xtr[maskc],train.spend.values[maskc])
        ft=rm_t.predict(Xall); fc=rm_c.predict(Xall)
        pred[f"spend_uplift_{slug}"]=ft-fc
        save_model(rm_t,f"spend_{slug}_treatment"); save_model(rm_c,f"spend_{slug}_control")
        spend_results.append({"comparison":f"{arm} vs Control","observed_spend_lift":float(a[(a.outcome=="spend")&(a.treatment==arm if arm!="Any Email" else a.treatment=="Any Email")].absolute_lift.iloc[0]) if False else np.nan})
    pred.to_csv(PREDICTIONS_DIR/"customer_predictions.csv",index=False)

    # Business targeting on held-out test is used for evaluation, with full-data predictions for scorer.
    test_pred=pred.iloc[idx_test].copy()
    curve_df=targeting_curve(test_pred); curve_df.to_csv(METRICS_DIR/"targeting_curve.csv",index=False)
    # Policy comparison at 20% using IPW evaluation.
    from .evaluation import policy_table
    policy_table(test_pred,0.20).to_csv(METRICS_DIR/"policy_comparison.csv",index=False)

    meta={"dataset":str(DATA_PATH.name),"rows":len(df),"train_rows":len(train),"test_rows":len(test),
          "random_seed":RANDOM_SEED,"test_size":TEST_SIZE,"uplift_threshold":UPLIFT_THRESHOLD,
          "features":FEATURES,"treatments":TREATMENTS}
    (METADATA_DIR/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({"rows":len(df),"train":len(train),"test":len(test),"ate":a.to_dict(orient="records"),
                      "best_model":max(results,key=lambda x:x["auuc"])},indent=2,default=float))
    return df,a,pred,pd.DataFrame(results),curve_df

if __name__=="__main__": main()
