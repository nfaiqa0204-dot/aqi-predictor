import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd
import hopsworks

load_dotenv()
AQICN_TOKEN=os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY=os.getenv("OPENWEATHER_API_KEY")
LAT,LON=33.7235,73.11822
HISTORY_FILE="data/processed/feature_history.csv"

def fetch_aqicn():
    url=f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
    resp=requests.get(url).json()
    if resp["status"]!="ok":
        raise Exception(f"AQICN error:{resp}")
    return resp["data"]

def fetch_openweather():
    url=f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    resp=requests.get(url).json()
    if resp.get("cod")!=200:
        raise Exception(f"OpenWeather error:{resp}")
    return resp

def build_raw_row():
    aqicn=fetch_aqicn()
    weather=fetch_openweather()
    now=datetime.now()
    return {
        "timestamp":now,
        "hour":now.hour,
        "day":now.day,
        "month":now.month,
        "pm25":aqicn["iaqi"].get("pm25",{}).get("v"),
        "temp":weather["main"]["temp"],
        "humidity":weather["main"]["humidity"],
        "pressure":weather["main"]["pressure"],
        "wind_speed":weather["wind"]["speed"],
    }

def add_derived_features(df):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["pm25_lag_1"]=df["pm25"].shift(1)
    df["pm25_change_rate"]=df["pm25"]-df["pm25_lag_1"]
    return df

def get_feature_store():
    project=hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    return project.get_feature_store()

def write_to_feature_store(df):
    fs=get_feature_store()
    fg=fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["timestamp"],
    description="AQI and weather features for Islamabad",
    event_time="timestamp",
    time_travel_format="HUDI"
    )
    fg.insert(df)
    print("Data written to Hopsworks feature store.")

def run():
    new_row=build_raw_row()
    if os.path.exists(HISTORY_FILE):
        history=pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
        history=pd.concat([history, pd.DataFrame([new_row])],ignore_index=True)
    else:
        history=pd.DataFrame([new_row])
    history=add_derived_features(history)
    history.to_csv(HISTORY_FILE, index=False)
    print("Latest row with derived features:")
    print(history.tail(1))
    latest_row_df=history.tail(1).reset_index(drop=True)
    write_to_feature_store(latest_row_df)

if __name__=="__main__":
    run()