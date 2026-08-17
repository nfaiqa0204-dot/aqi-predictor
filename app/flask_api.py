from flask import Flask, jsonify
import os
from dotenv import load_dotenv
import hopsworks
import joblib
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_pipeline import add_derived_features

load_dotenv()
app = Flask(__name__)

FEATURE_COLUMNS = [
    "hour", "day", "month_sin", "month_cos", "pm25", "temp", "humidity", "pressure", "wind_speed",
    "pm25_lag_1h", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_72h", "pm25_change_rate"
]


@app.route("/")
def home():
    return jsonify({"message": "Islamabad AQI Forecast API. Try /forecast for predictions."})


@app.route("/forecast")
def forecast():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(name="aqi_features_v2", version=1)
    df = fg.read().sort_values("timestamp").reset_index(drop=True)
    df = add_derived_features(df)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    latest = df.tail(1)

    X = latest[FEATURE_COLUMNS]

    results = {}
    for horizon in ["24h", "48h", "72h"]:
        model_meta = mr.get_model(f"aqi_ridge_target_{horizon}", version=2)
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        results[horizon] = round(float(model.predict(X)[0]), 1)

    return jsonify({
        "location": "Islamabad, Pakistan",
        "current_pm25": float(latest["pm25"].values[0]),
        "current_timestamp": str(latest["timestamp"].values[0]),
        "forecast": {
            "24h": results["24h"],
            "48h": results["48h"],
            "72h": results["72h"]
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)