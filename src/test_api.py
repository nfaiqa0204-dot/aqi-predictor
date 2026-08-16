import os
from dotenv import load_dotenv
import requests
load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")
url = f"https://api.waqi.info/feed/islamabad/?token={TOKEN}"
resp = requests.get(url).json()
print(resp)