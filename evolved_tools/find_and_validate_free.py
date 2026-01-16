import requests
from bs4 import BeautifulSoup

def find_and_validate_free(**kwargs):
    """
    Finds and validates free HTTPS proxies from the web.
    Returns a dictionary with keys: {"success": bool, "message": str, "data": dict}.
    PRIORITIZES FREE / NO-AUTH APIs. Does NOT use APIs that require a key.
    Uses scraping if no free API exists.
    """
    success = False
    message = ""
    data = {}

    try:
        # Scrape proxies from a free proxy list website (e.g., https://free-proxy-list.net/)
        url = "https://free-proxy-list.net/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        proxies = []
        proxies = []
        for tr in soup.find_all('tr')[1:]:
            td = tr.find_all('td')
            if len(td) < 2:
                continue
            proxy = {
                'ip': td[0].get_text().strip(),
                'port': td[1].get_text().strip(),
                'country': td[3].get_text().strip() if len(td) > 3 else "Unknown" # Index 3 is usually code, 4 is name
            }
            proxies.append(proxy)
            if len(proxies) >= 20: # Limit to 20 for speed
                break

        # Validate proxies by trying to connect to Google using each one
        valid_proxies = []
        for proxy in proxies:
            try:
                response = requests.get('https://www.google.com', proxies={'http': f"{proxy['ip']}:{proxy['port']}", 'https': f"{proxy['ip']}:{proxy['port']}"}, timeout=5)
                if response.status_code == 200:
                    valid_proxies.append(proxy)
            except Exception as e:
                print(f"Invalid proxy: {proxy} - {e}")

        # If we found valid proxies, update success and data
        if valid_proxies:
            success = True
            message = "Proxies found and validated"
            data = {"proxies": valid_proxies}

    except Exception as e:
        message = f"Error finding and validating free HTTPS proxies: {e}"

    return {"success": success, "message": message, "data": data}

TOOL_METADATA = {
    "name": "find_and_validate_free",
    "description": "Finds and validates free HTTPS proxies from the web.",
    "parameters": {},
    "category": "tool"
}