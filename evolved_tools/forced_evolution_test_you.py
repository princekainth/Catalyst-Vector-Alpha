def forced_evolution_test_you(**kwargs):
    import requests
    from datetime import datetime

    try:
        # Fetch temperature data using Open-Meteo API
        api_url = 'https://api.open-meteo.com/v1/forecast?latitude=44.39&longitude=-79.65&current_weather=true'
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Get current temperature and time
        temperature = data['current_weather']['temperature']
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return {
            "success": True,
            "message": "Temperature fetched",
            "data": {"temperature": temperature, "time": current_time}
        }
    except requests.exceptions.RequestException as err:
        return {"success": False, "message": f"Error fetching data: {err}", "data": {}}

TOOL_METADATA = {
    "name": "forced_evolution_test_you",
    "description": "Fetches current outdoor temperature in Barrie, Ontario and the current time.",
    "parameters": {},
    "category": "environment"
}