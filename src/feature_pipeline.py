import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd
load_dotenv()
AQICN_TOKEN=os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY=os.getenv("OPENWEATHER_API_KEY")
LAT,LON=33.7235,73.11822

def fetch_aqicn():
    url=f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}"
    resp=requests.get(url).json()
    if resp["status"]!="ok":
        raise Exception(f"AQICN error:{resp}")
    return resp["data"]

def fetch_openweather():
    url=f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    resp=requests.get(url).json()
    if resp.get("cod")!=200:
        raise Exception(f"OpenWeather error: {resp}")
    return resp

def build_feature_row():
    aqicn=fetch_aqicn()
    weather=fetch_openweather()
    now=datetime.now()
    row={
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
    return row

if __name__=="__main__":
    row=build_feature_row()
    print(row)
    df=pd.DataFrame([row])
    df.to_csv("data/processed/live_feature_test.csv",index=False)
    print("Saved to data/processed/live_feature_test.csv")