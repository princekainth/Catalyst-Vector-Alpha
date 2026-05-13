def check_if_is_raining(**kwargs):
    import requests

    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast?latitude=35.6894&longitude=139.6917&current_weather=true")
        data = response.json()
        return {"success": True, "message": "Weather data fetched", "data": {"is_raining": data["current_weather"]["weathercode"] in [61, 63, 65]}}
    except Exception as e:
        return {"success": False, "message": str(e), "data": {}}

TOOL_METADATA = {
    "name": "check_if_is_raining",
    "description": "Checks if it is raining in Tokyo using Open-Meteo API",
    "parameters": {},
    "category": "weather"
}