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
st.write("Loading latest data...")
fg=fs.get_feature_group(name="aqi_features_v2",version=1)
df=fg.read()
df=df.sort_values("timestamp").reset_index(drop=True)
latest=df.tail(1)
st.subheader("Current Conditions")
st.metric("Current PM2.5", f"{latest['pm25'].values[0]:.0f}")
st.write(latest)