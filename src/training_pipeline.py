import os
from dotenv import load_dotenv
import pandas as pd
import hopsworks

load_dotenv()

def get_feature_store():
    project=hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    return project.get_feature_store()

def load_features():
    fs=get_feature_store()
    fg=fs.get_feature_group(name="aqi_features_v2", version=1)
    df=fg.read()
    df=df.sort_values("timestamp").reset_index(drop=True)
    return df

def build_targets(df):
    df=df.set_index("timestamp")
    pm25_series=df["pm25"]

    def get_future(hours):
        future_times=df.index+pd.Timedelta(hours=hours)
        return future_times.map(lambda t:pm25_series.asof(t))
    df["target_24h"]=get_future(24)
    df["target_48h"]=get_future(48)
    df["target_72h"]=get_future(72)
    df=df.reset_index()
    return df

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

FEATURE_COLUMNS=[
    "hour","day","month","pm25","temp","humidity","pressure","wind_speed",
    "pm25_lag_1h","pm25_lag_24h","pm25_lag_48h","pm25_lag_72h","pm25_change_rate"
]

def prepare_training_data(df, target_col):
    data=df.dropna(subset=FEATURE_COLUMNS+[target_col]).copy()
    data=data.sort_values("timestamp")
    X=data[FEATURE_COLUMNS]
    y=data[target_col]
    split_idx=int(len(data)*0.8)
    X_train,X_test=X.iloc[:split_idx],X.iloc[split_idx:]
    y_train,y_test=y.iloc[:split_idx],y.iloc[split_idx:]
    return X_train,X_test,y_train,y_test

def train_and_evaluate(X_train,X_test,y_train,y_test,target_name):
    models = {
    "Ridge Regression":Ridge(alpha=1.0),
    "Random Forest":RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42),
    "XGBoost":XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42),
}
    results={}
    for name, model in models.items():
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        rmse=np.sqrt(mean_squared_error(y_test,preds))
        mae=mean_absolute_error(y_test,preds)
        r2=r2_score(y_test, preds)
        results[name]={"model":model,"rmse":rmse,"mae":mae,"r2":r2}
        print(f"\n[{target_name}]{name}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  R²:   {r2:.3f}")
    return results

import joblib

def save_best_model(project, model, target_name, metrics):
    model_dir=f"models/{target_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path=f"{model_dir}/model.pkl"
    joblib.dump(model, model_path)
    mr=project.get_model_registry()
    hw_model=mr.python.create_model(
        name=f"aqi_ridge_{target_name}",
        metrics={"rmse": metrics["rmse"],"mae":metrics["mae"],"r2": metrics["r2"]},
        description=f"Ridge Regression model predicting PM2.5 at {target_name}"
    )
    hw_model.save(model_dir)
    print(f"Saved {target_name} model to Model Registry.")

if __name__=="__main__":
    project=hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs=project.get_feature_store()
    df=load_features()
    print(f"Loaded {len(df)} rows from feature store.")
    df=build_targets(df)
    for target in ["target_24h","target_48h","target_72h"]:
        X_train,X_test,y_train,y_test=prepare_training_data(df,target)
        results=train_and_evaluate(X_train,X_test,y_train,y_test,target)
        best_model_info=results["Ridge Regression"]
        save_best_model(project, best_model_info["model"],target,best_model_info)


