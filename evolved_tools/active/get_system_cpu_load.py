import psutil

def get_system_cpu_load(**kwargs):
    """Get system CPU load.

    Args:
        **kwargs: Additional arguments passed to the function.

    Returns:
        dict: A dictionary with keys `{"success": bool, "message": str, "data": dict}`.
              If successful, `data` contains a dictionary with CPU usage details.
              If not successful, `message` contains an error message and `data` is None.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=kwargs.get('time_interval_seconds', 0.5), percpu=kwargs.get('per_core', False))
        return {"success": True, "message": "", "data": {"cpu_usage": cpu_percent}}
    except Exception as e:
        return {"success": False, "message": str(e), "data": None}

TOOL_METADATA = {
    "name": "get_system_cpu_load",
    "description": "Get the current CPU load of the system.",
    "parameters": {
        "time_interval_seconds": {"type": "number", "default": 0.5},
        "per_core": {"type": "boolean", "default": False}
    },
    "category": "System"
}