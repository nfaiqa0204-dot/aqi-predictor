import os
from dotenv import load_dotenv
import requests
load_dotenv()
KEY=os.getenv("OPENWEATHER_API_KEY")
LAT,LON=33.7235,73.11822  
url=f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={KEY}&units=metric"
resp=requests.get(url).json()
print(resp)