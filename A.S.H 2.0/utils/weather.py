import requests ,os
from datetime import datetime,timedelta



W_API_KEY = os.getenv("W_API_KEY")
weather = ''

class kweather():
    def __init__(self,loc):
        self.KHALID="the best"
        self.apikey=W_API_KEY
        self.loc =loc

    def add_days(self,location=None):
        if not location:
            location = self.loc
        api_key = self.apikey  # Replace with your WeatherAPI.com API key
        base_url = "https://api.weatherapi.com/v1/forecast.json"
        # Make a request to the WeatherAPI.com API
        response = requests.get(base_url, params={"key": api_key, "q": location, "days": 3, "aqi": "no", "alerts": "no"})
        data = response.json()
        hourly_forecasts = data["forecast"]["forecastday"]
        for i in hourly_forecasts:
            for j in i["hour"]:
                j.update({"location":data["location"]["name"].lower()+" "+data["location"]["country"].lower()})
                weather= j
        
        return weather
    
    def accurate_weather(self,location=None):
        if not location:
            location = self.loc
        api_key = self.apikey # Replace with your WeatherAPI.com API key
        base_url = "https://api.weatherapi.com/v1/forecast.json"
        # Make a request to the WeatherAPI.com API
        response = requests.get(base_url, params={"key": api_key, "q": location, "days": 3, "aqi": "no", "alerts": "no"})
        data = response.json()
        return data["current"]
    
    def get_accurate_weather(self,location=None):
        if not location:
            location = self.loc
        data=self.accurate_weather(location=location)
        
        # Extract relevant information from the response
        if "error" in data:
            result="An error occurred:"+data["error"]["message"]
        else:
            current_temp = data["temp_c"]
            feels_like_temp = data["feelslike_c"]
            condition = data["condition"]["text"]
            wind_speed = data["wind_kph"]
            humidity = data["humidity"]
            visibility = data["vis_km"]
            pressure = data["pressure_mb"]
            result=f"Weather update for {location}:\nTemperature: {current_temp}°C\nFeels Like: {feels_like_temp}°C\nCondition: {condition}\nWind Speed: {wind_speed} kph\nHumidity: {humidity}%\nVisibility: {visibility} km\nPressure: {pressure} mb"
    
        return result    
    
    def add_forcast(self,location=None):
        if not location:
            location = self.loc
        api_key = self.apikey  # Replace with your WeatherAPI.com API key
        base_url = "https://api.weatherapi.com/v1/forecast.json"
        # Make a request to the WeatherAPI.com API
        response = requests.get(base_url, params={"key": api_key, "q": location, "days": 3, "aqi": "no", "alerts": "no"})
        data = response.json()
        hourly_forecasts = data["forecast"]["forecastday"]
        for i in hourly_forecasts:
            print(i)
            j = i["day"]
            j.update({"time":i["date"],"location":data["location"]["name"].lower()+" "+data["location"]["country"].lower()})
            forecast=j  
            return forecast 
        
    def get_weather(self,time,location=None):
        if not location:
            location = self.loc
        res=weather.find_one({"location":location,"time":time.isoformat().replace("T", " ")[:-3]})
        return res    
        
    def weather_now(self,location=None):
        if not location:
            location = self.loc
        res=self.get_weather(location, datetime.now()-timedelta(minutes=datetime.now().time().minute,seconds=datetime.now().time().second,microseconds=datetime.now().time().microsecond))
        return res

    def get_weather_update(self,time=datetime.now(),location=None):
        if not location:
            location = self.loc
        time=time-timedelta(minutes=time.time().minute,seconds=time.time().second,microseconds=time.now().time().microsecond)
        data=self.get_weather(location=location,time= time)
        if not data:
            self.add_days(location=location)
            data=self.get_weather(location= location,time= time)
        
        # Extract relevant information from the response
        if "error" in data:
            result="An error occurred:"+data["error"]["message"]
        else:
            ocation = data["location"]
            current_temp = data["temp_c"]
            feels_like_temp = data["feelslike_c"]
            condition = data["condition"]["text"]
            wind_speed = data["wind_kph"]
            humidity = data["humidity"]
            visibility = data["vis_km"]
            pressure = data["pressure_mb"]
            cor= data["chance_of_rain"]
            # Print the current weather update
            result=f"Weather update for {location}:\nTemperature: {current_temp}°C\nFeels Like: {feels_like_temp}°C\nCondition: {condition}\nWind Speed: {wind_speed} kph\nHumidity: {humidity}%\nVisibility: {visibility} km\nPressure: {pressure} mb\nChance of Rain: {cor}%"
    
        return result

    def get_forcast(self,date=datetime.now().date(),location=None):
        if not location:
            location = self.loc
        daily_forecasts = self.forcast(location=location,date=date) 
        result="Daily Forecasts:"
        # Print daily forecasts for the upcoming days
        for forecast in daily_forecasts:
            date = forecast["time"]
            min_temp = forecast["mintemp_c"]
            max_temp = forecast["maxtemp_c"]
            condition = forecast["condition"]["text"]
            chance_of_rain = forecast["daily_chance_of_rain"]
            result+=f"\nDate: {date}, Min Temperature: {min_temp}°C, Max Temperature: {max_temp}°C, Condition: {condition}, Chance of Rain: {chance_of_rain}%"
        return result
    
    # def forcast(self,date=datetime.now().date(),location=None):
    #     if not location:
    #         location = self.loc
    #     data = forecast.find_one({"time":date.isoformat(),"location":location})
    #     if not data:
    #         self.add_forcast(location=location)
    #         data=forecast.find_one({"time":date.isoformat(),"location":location})
    #     forecasts=[]
    #     for i in range(3):
    #         forecasts.append(forecast.find_one({"time":date.isoformat(),"location":location}))
    #         date=date+timedelta(days=1)
    #     return forecasts

    def get_hourly_forcast(self,time=datetime.now(),location=None):
        if not location:
            location = self.loc
        hourly_forecasts=self.hourly_forcast(location=location,time=time)
        result="Hourly Forecasts:"
        # Print hourly forecasts for the day
        for forecast in hourly_forecasts:
            time = forecast["time"]
            temperature = forecast["temp_c"]
            feels_like_temp = forecast["feelslike_c"]
            condition = forecast["condition"]["text"]
            result+=f"\nTime: {time}, Temperature: {temperature}°C, Feels Like: {feels_like_temp}°C, Condition: {condition}"

        return result

    def hourly_forcast(self,time=datetime.now(),location=None):
        if not location:
            location = self.loc
        time=time-timedelta(hours=time.time().hour, minutes=time.time().minute,seconds=time.time().second,microseconds=time.now().time().microsecond)
        data=self.get_weather(location, time)
        if not data:
            self.add_days(location=location)
            data=self.get_weather(location, time)
        
        # Extract relevant information from the response
        if "error" in data:
            result="An error occurred:"+data["error"]["message"]
        else:
            hourly_forecasts=[]
            for i in range(24):
                time=time+timedelta(hours=1)
                hourly_forecasts.append(self.get_weather(location, time))
        return hourly_forecasts
   