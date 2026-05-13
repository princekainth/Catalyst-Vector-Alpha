def get_current_price_of(**kwargs):
    import requests

    try:
        response = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot')
        response.raise_for_status()
        data = response.json()['data']
        price = data['amount']

        return {"success": True, "message": "Bitcoin price fetched", "data": {"price": price}}
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": str(e), "data": {}}

TOOL_METADATA = {
    "name": "get_current_price_of",
    "description": "Returns the current price of Bitcoin in USD",
    "parameters": {},
    "category": "finance"
}