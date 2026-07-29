import os
from dotenv import load_dotenv
import hopsworks
load_dotenv()
project=hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)
print("Connected to project:",project.name)
fs=project.get_feature_store()
print("Feature store connected:",fs)