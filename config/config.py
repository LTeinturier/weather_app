import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    API_KEY = os.getenv('API_KEY')
    CITY = 'London' #it only works for London for now. Need to create call to geocoding API to get lon/lat from city name. Next version
    BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'
