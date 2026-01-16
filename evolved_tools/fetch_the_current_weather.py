def fetch_the_current_weather(city: str = "London", **kwargs):
    """
    Fetches the current weather in the specified city (default is London).

    Args:
        city (str, optional): The city to fetch weather for. Defaults to "London".

    Returns:
        dict: A dictionary containing a success flag, message, and weather data.
            - {"success": bool}
            - {"message": str}
            - {"data": {"temperature": float, "description": str}}
    """
    import requests
    import json

    import requests
    import json

    try:
        # Use Open-Meteo Geocoding API to get coords for the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url)
        if geo_res.status_code != 200 or not geo_res.json().get('results'):
             return {"success": False, "message": f"City '{city}' not found.", "data": {}}
        
        location = geo_res.json()['results'][0]
        lat, lon = location['latitude'], location['longitude']
        
        # Use Open-Meteo Weather API (Free! No Key!)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(weather_url)
        data = response.json()
        
        if "current_weather" in data:
            result = data["current_weather"]
            return {
                "success": True,
                "message": f"Weather in {city}: {result['temperature']}°C, Wind: {result['windspeed']}km/h",
                "data": {
                    "temperature": result["temperature"],
                    "windspeed": result["windspeed"],
                    "winddirection": result["winddirection"],
                    "is_day": result["is_day"],
                    "time": result["time"]
                }
            }
        else:
            return {"success": False, "message": "Failed to fetch weather data", "data": data}
            
    except Exception as e:
        return {"success": False, "message": str(e), "data": {}}

TOOL_METADATA = {
    "name": "fetch_the_current_weather",
    "description": "Returns current weather in a specified city.",
    "parameters": {"city": {"type": "string", "optional": True}},
    "category": "weather"
}