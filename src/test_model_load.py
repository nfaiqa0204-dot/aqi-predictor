import os
from dotenv import load_dotenv
import hopsworks
import joblib

load_dotenv()

project=hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)
mr=project.get_model_registry()
model_meta=mr.get_model("aqi_ridge_target_24h",version=1)
model_dir=model_meta.download()
model_path=os.path.join(model_dir,"model.pkl")
model=joblib.load(model_path)
print("Model loaded successfully:",model)
print("Model coefficients:",model.coef_)
print("Model intercept:",model.intercept_)