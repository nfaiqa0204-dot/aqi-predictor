import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features_v2", version=1)
df = fg.read()
suspicious = df[df["pressure"] > 900]
print(f"Found {len(suspicious)} suspicious rows (pressure > 900 hPa):")
print(suspicious[["timestamp", "pm25", "temp", "humidity", "pressure"]])