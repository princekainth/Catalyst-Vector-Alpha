# In tools.py
import logging
import random
import psutil # Make sure you have installed this: pip install psutil
from datetime import datetime
import time
from typing import Optional, Dict
logger = logging.getLogger("CatalystLogger")

def get_system_cpu_load_tool():
    """
    Safely gets the current system-wide CPU load.
    Returns the load as a percentage or an error message string.
    """
    try:
        # psutil is a library for system monitoring
        cpu_load = psutil.cpu_percent(interval=1)
        return f"Current system CPU load is {cpu_load}%."
    except Exception as e:
        # If psutil fails for any reason, return a clear error message
        # instead of crashing the agent.
        return f"Tool 'get_system_cpu_load' execution failed due to an exception: {str(e)}"

def initiate_network_scan_tool(target_ip: str, scan_type: str = "ping_sweep") -> str:
    """
    Simulates initiating a network scan on a target IP.
    Emits a concise, success-oriented result; failure rate effectively zero (configurable later).
    """
    logger.info(f"[TOOL EXEC] network_scan: type={scan_type} target={target_ip}")
    time.sleep(0.3)  # small simulated delay

    if scan_type == "full_port_scan":
        ports_pool = [21, 22, 23, 80, 443, 3389, 8080]
        # choose 1–3 ports deterministically-ish
        k = random.randint(1, 3)
        open_ports = random.sample(ports_pool, k=k)
        vulnerable_services = []
        if random.random() < 0.03:  # very small chance
            vulnerable_services = random.sample(
                ["SSH_vulnerable", "FTP_anonymous_access"], k=1
            )
        result = f"Port scan on {target_ip} completed. Open ports: {', '.join(map(str, open_ports))}."
        if vulnerable_services:
            result += f" Detected vulnerable services: {', '.join(vulnerable_services)}."
        logger.info(result)
        return result

    elif scan_type == "vulnerability_scan":
        vulns = []
        if random.random() < 0.02:
            vulns = random.sample(["CVE-2023-1234 (High)", "CVE-2022-5678 (Medium)"], k=1)
        result = (
            f"Vulnerability scan on {target_ip} completed. Found vulnerabilities: {', '.join(vulns)}."
            if vulns
            else f"Vulnerability scan on {target_ip} completed. No critical vulnerabilities found."
        )
        logger.info(result)
        return result

    elif scan_type == "ping_sweep":
        result = f"Successfully pinged {target_ip}. Host is up."
        logger.info(result)
        return result

    else:
        result = f"Unknown scan type '{scan_type}'. Scan on {target_ip} completed with default results."
        logger.warning(result)
        return result

def deploy_recovery_protocol_tool(protocol_name: str, target_system_id: str, urgency_level: str = "medium") -> str:
    """
    Simulates deploying a specific recovery protocol to a target system.
    Always succeeds here (keep the surface deterministic for agent logic).
    """
    logger.info(f"[TOOL EXEC] deploy_recovery: protocol={protocol_name} target={target_system_id} urgency={urgency_level}")
    time.sleep(0.5)
    result = f"Recovery protocol '{protocol_name}' successfully deployed to {target_system_id} (Urgency: {urgency_level})."
    logger.info(result)
    return result     

def update_resource_allocation_tool(resource_type: str, target_agent_name: str, new_allocation_percentage: float) -> str:
    """
    Simulates updating resource allocation for a specific agent.
    """
    logger.info(f"[TOOL EXEC] update_allocation: resource={resource_type} agent={target_agent_name} pct={new_allocation_percentage}")
    if not (0.0 <= new_allocation_percentage <= 1.0):
        msg = f"Update resource allocation FAILED: Invalid percentage {new_allocation_percentage}. Must be between 0.0 and 1.0."
        logger.error(msg)
        return msg
    pct_str = f"{new_allocation_percentage * 100:.1f}%"
    result = f"{resource_type} allocation for {target_agent_name} successfully updated to {pct_str}."
    logger.info(result)
    return result

def get_environmental_data_tool(agent_name: str, location: Optional[str] = None, data_type: str = 'all', **kwargs) -> Dict:
        print(f"[Tool Call] Agent '{agent_name}' is fetching {data_type} environmental data for {location if location else 'general area'}.")
        data = {
            "temperature_celsius": round(random.uniform(15.0, 30.0), 2),
            "humidity_percent": round(random.uniform(40.0, 60.0), 2),
            "air_quality_index": round(random.uniform(20.0, 50.0), 2),
            "water_level_m": round(random.uniform(5.0, 10.0), 2),
            "timestamp": datetime.now().isoformat()
        }
        if data_type != 'all' and data_type in data:
            return {"status": "completed", "result": {data_type: data[data_type], "timestamp": data["timestamp"]},
                    "tool_name": "get_environmental_data"}
        return {"status": "completed", "result": data, "tool_name": "get_environmental_data"}
