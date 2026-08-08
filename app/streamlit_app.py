import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv
import hopsworks
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__),"..","src"))
from feature_pipeline import add_derived_features
load_dotenv()
st.set_page_config(page_title="Islamabad AQI Forecast", layout="centered")
st.title("🌫️ Islamabad AQI Forecast")

@st.cache_resource
def connect():
    project=hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    return project

project=connect()
fs=project.get_feature_store()
mr=project.get_model_registry()
fg=fs.get_feature_group(name="aqi_features_v2",version=1)
df=fg.read()
df=df.sort_values("timestamp").reset_index(drop=True)
df=add_derived_features(df)
FEATURE_COLUMNS=[
    "hour","day","month_sin","month_cos","pm25","temp","humidity","pressure","wind_speed",
    "pm25_lag_1h","pm25_lag_24h","pm25_lag_48h","pm25_lag_72h","pm25_change_rate"
]

import numpy as np
df["month_sin"]=np.sin(2*np.pi*df["month"]/12)
df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
latest=df.tail(1)

@st.cache_resource
def load_model(target_name):
    model_meta=mr.get_model(f"aqi_ridge_{target_name}",version=2)
    model_dir=model_meta.download()
    model_path=os.path.join(model_dir,"model.pkl")
    return joblib.load(model_path)

model_24h=load_model("target_24h")
model_48h=load_model("target_48h")
model_72h=load_model("target_72h")

def get_aqi_category(pm25):
    if pm25 <= 50:
        return "Good","#00e400"
    elif pm25 <= 100:
        return "Moderate","#ffff00"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive Groups","#ff7e00"
    elif pm25 <= 200:
        return "Unhealthy","#ff0000"
    elif pm25 <= 300:
        return "Very Unhealthy","#8f3f97"
    else:
        return "Hazardous","#7e0023"

current_pm25=latest["pm25"].values[0]
category,color=get_aqi_category(current_pm25)

st.subheader("Current Conditions")
st.markdown(
    f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:0;">{current_pm25:.0f}</h1>
        <p style="color:white; margin:0; font-size:18px;">{category}</p>
    </div>
    """,
    unsafe_allow_html=True
)
X_latest=latest[FEATURE_COLUMNS]
pred_24h=model_24h.predict(X_latest)[0]
pred_48h=model_48h.predict(X_latest)[0]
pred_72h=model_72h.predict(X_latest)[0]

st.subheader("3-Day Forecast")
forecast_data=[
    ("Tomorrow",pred_24h),
    ("In 2 days",pred_48h),
    ("In 3 days",pred_72h),
]

cols=st.columns(3)
for col,(label,value) in zip(cols,forecast_data):
    cat,col_color=get_aqi_category(value)
    with col:
        st.markdown(
            f"""
            <div style="background-color:{col_color}; padding:15px; border-radius:10px; text-align:center;">
                <p style="color:white; margin:0; font-size:14px;">{label}</p>
                <h2 style="color:white; margin:0;">{value:.0f}</h2>
                <p style="color:white; margin:0; font-size:12px;">{cat}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

max_forecast=max(pred_24h,pred_48h,pred_72h)
if max_forecast>150:
    st.markdown(
        f"""
        <div style="background-color:#ff0000; padding:15px; border-radius:10px; margin-top:20px; text-align:center;">
            <h3 style="color:white; margin:0;">⚠️ Hazardous Air Quality Alert</h3>
            <p style="color:white; margin:5px 0 0 0;">AQI is expected to reach unhealthy levels in the next 3 days. Consider limiting outdoor activity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.subheader("Trend:Last 7 Days+Forecast")
recent=df.tail(7)[["timestamp","pm25"]].copy()
recent["type"]="Historical"
last_time=df["timestamp"].max()
forecast_points=pd.DataFrame({
    "timestamp":[last_time,last_time+pd.Timedelta(hours=24),last_time+pd.Timedelta(hours=48),last_time+pd.Timedelta(hours=72)],
    "pm25":[current_pm25,pred_24h,pred_48h,pred_72h],
    "type":["Forecast","Forecast","Forecast","Forecast"]
})

fig=go.Figure()
fig.add_trace(go.Scatter(
    x=recent["timestamp"],y=recent["pm25"],
    mode="lines+markers",name="Historical",
    line=dict(color="#00bcd4", width=3)
))
fig.add_trace(go.Scatter(
    x=forecast_points["timestamp"],y=forecast_points["pm25"],
    mode="lines+markers",name="Forecast",
    line=dict(color="#ff7e00",width=3,dash="dash")
))
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Date",
    yaxis_title="PM2.5",
    height=400
)
st.plotly_chart(fig, use_container_width=True,key="aqi_trend_chart")