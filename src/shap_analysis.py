import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import hopsworks
import joblib
import shap
import matplotlib.pyplot as plt

from feature_pipeline import add_derived_features

load_dotenv()

FEATURE_COLUMNS = [
    "hour", "day", "month_sin", "month_cos", "pm25", "temp", "humidity", "pressure", "wind_speed",
    "pm25_lag_1h", "pm25_lag_24h", "pm25_lag_48h", "pm25_lag_72h", "pm25_change_rate"
]


def load_data_and_model(target_name):
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(name="aqi_features_v2", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_derived_features(df)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df = df.dropna(subset=FEATURE_COLUMNS)

    model_meta = mr.get_model(f"aqi_ridge_{target_name}", version=2)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    return df, model


def run_shap_analysis(target_name="target_24h"):
    df, model = load_data_and_model(target_name)
    X = df[FEATURE_COLUMNS]

   
    X_sample = X.sample(min(200, len(X)), random_state=42)

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{target_name}.png", dpi=150)
    plt.close()

    print(f"Saved shap_summary_{target_name}.png")

    # Bar plot (mean absolute impact per feature)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_bar_{target_name}.png", dpi=150)
    plt.close()

    print(f"Saved shap_bar_{target_name}.png")


if __name__ == "__main__":
    run_shap_analysis("target_24h")