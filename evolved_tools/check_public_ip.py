def missing_tool_for_task(**kwargs):
    import requests
    try:
        response = requests.get('https://api.ipify.org')
        response.raise_for_status()
        ip_address = response.text.strip()
        with open('ip.txt', 'w') as f:
            f.write(ip_address)
        return {"success": True, "message": "IP address fetched and saved", "data": {"ip_address": ip_address}}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": str(e), "data": {}}

TOOL_METADATA = {
    "name": "check_public_ip",
    "description": "Checks current public IP address and saves it to a file named ip.txt.",
    "parameters": {},
    "category": "utility"
}
