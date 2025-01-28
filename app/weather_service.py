import requests
import pandas as pd

def fetch_weather_data(api_key, city, base_url):
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    lat, lon = 51.5156177,-0.0919983 #because hardcoded for London. Will change when making call to geocoding API. 
    url = base_url+f'lat={lat}&lon={lon}&appid={api_key}&units=metric'
    print(url)
    response = requests.get(base_url, params=params)
    data = response.json()
    return data

def process_weather_data(data):
    weather_list = []
    for item in data['list']:
        weather_list.append({
            'date': item['dt_txt'],
            'temperature': item['main']['temp'],
            'humidity': item['main']['humidity'],
            'pressure': item['main']['pressure']
        })
    df = pd.DataFrame(weather_list)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df
