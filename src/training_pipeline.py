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

if __name__=="__main__":
    df=load_features()
    print(f"Loaded {len(df)} rows from feature store.")
    df=build_targets(df)
    print(df[["timestamp","pm25","target_24h","target_48h","target_72h"]].tail(10))