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

import math

def fetch_aqicn():
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={AQICN_TOKEN}"
    resp = requests.get(url).json()
    if resp["status"] != "ok":
        raise Exception(f"AQICN error: {resp}")

    data = resp["data"]

    
    station_lat, station_lon = data["city"]["geo"]
    distance = math.sqrt((station_lat - LAT)**2 + (station_lon - LON)**2)

    if distance > 1.0: 
        raise Exception(
            f"AQICN returned a distant station ({data['city']['name']}, "
            f"{distance:.2f} degrees away) instead of Islamabad — skipping this run."
        )

    return data

def fetch_openweather():
    url=f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    resp=requests.get(url).json()
    if resp.get("cod")!=200:
        raise Exception(f"OpenWeather error:{resp}")
    return resp

def build_raw_row():
    aqicn = fetch_aqicn()
    weather = fetch_openweather()
    now = datetime.now()
    return {
        "timestamp": now,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "pm25": int(round(float(aqicn["iaqi"].get("pm25", {}).get("v")))),
        "temp": float(weather["main"]["temp"]),
        "humidity": int(round(float(weather["main"]["humidity"]))),
        "pressure": int(round(float(weather["main"]["pressure"]))),
        "wind_speed": float(weather["wind"]["speed"]),
    }

def add_derived_features(df):
    df=df.sort_values("timestamp").reset_index(drop=True)
    df=df.set_index("timestamp")
    pm25_series=df["pm25"]
    def get_lag(hours):
        lagged_times=df.index-pd.Timedelta(hours=hours)
        return lagged_times.map(lambda t: pm25_series.asof(t))
    df["pm25_lag_1h"]=get_lag(1)
    df["pm25_lag_24h"]=get_lag(24)
    df["pm25_lag_48h"]=get_lag(48)
    df["pm25_lag_72h"]=get_lag(72)
    df["pm25_change_rate"]=df["pm25"]-df["pm25_lag_1h"]
    df=df.reset_index()
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
        name="aqi_features_v2",
        version=1,
        primary_key=["timestamp"],
        description="AQI and weather features for Islamabad (with multi-window lag features)",
        event_time="timestamp",
        time_travel_format="HUDI"
    )
    fg.insert(df)
    print("Data written to Hopsworks feature store.")

def fetch_historical_weather(start_date,end_date):
    """Fetch daily historical weather for a date range from Open-Meteo."""
    url=(
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_mean"
        f"&timezone=Asia%2FKarachi"
    )
    resp=requests.get(url).json()
    if "daily" not in resp:
        raise Exception(f"Open-Meteo error:{resp}")
    weather_df=pd.DataFrame({
        "date":resp["daily"]["time"],
        "temp":resp["daily"]["temperature_2m_mean"],
        "humidity":resp["daily"]["relative_humidity_2m_mean"],
        "pressure":resp["daily"]["surface_pressure_mean"],
        "wind_speed":resp["daily"]["wind_speed_10m_mean"],
    })
    weather_df["date"]=pd.to_datetime(weather_df["date"])
    return weather_df

def run():
    try:
        new_row = build_raw_row()
    except Exception as e:
        print(f"Skipping this run: {e}")
        return

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    else:
        history = pd.DataFrame([new_row])

    history = add_derived_features(history)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history.to_csv(HISTORY_FILE, index=False)

    print("Latest row with derived features:")
    print(history.tail(1))

    latest_row_df = history.tail(1).reset_index(drop=True)
    write_to_feature_store(latest_row_df)

if __name__=="__main__":
    run()