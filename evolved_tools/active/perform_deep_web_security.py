import socket

def perform_deep_web_security(target_url: str = None, target: str = None, **kwargs) -> dict:
    """
    Scan common ports on the target server to find open ones.
    """
    url = target_url or target
    if not url:
        return {"success": False, "message": "Missing target", "data": {}}

    # Remove http:// or https:// and any path
    hostname = url.replace("http://", "").replace("https://", "").split("/")[0]

    # Common ports to check
    common_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]
    open_ports = []

    for port in common_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)          # wait 1 second for a response
            result = sock.connect_ex((hostname, port))
            if result == 0:              # port is open
                open_ports.append(port)
            sock.close()
        except Exception:
            # Ignore errors (like hostname resolution) and continue
            pass

    return {
        "success": True,
        "message": f"Scan completed for {hostname}",
        "data": {"open_ports": open_ports}
    }

TOOL_METADATA = {
    "name": "perform_deep_web_security",
    "description": "Scans a target URL for open ports using TCP connect.",
    "parameters": {"target_url": {"type": "string", "required": True}},
    "category": "security"
}