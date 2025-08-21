# In tool_registry.py
from typing import Optional, List, Dict
# This import will now work because tools.py is in the same folder
from tools import (
    get_system_cpu_load_tool,
    initiate_network_scan_tool,
    deploy_recovery_protocol_tool,
    update_resource_allocation_tool,
    get_environmental_data_tool,
    analyze_threat_signature_tool,
    isolate_network_segment_tool,
)

class Tool:
    """
    Represents an external function or API call that an agent can use.
    The schema adheres to common LLM function calling conventions.
    """
    def __init__(self, name: str, description: str, parameters: dict, func):
        self.name = name
        self.description = description
        self.parameters = parameters # JSON schema-like dictionary for parameters
        self.func = func # The actual Python function to call

    def get_function_spec(self) -> dict:
        """Returns the tool's specification in a format suitable for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __call__(self, *args, **kwargs):
        """Allows the Tool instance to be called directly, executing its wrapped function."""
        return self.func(*args, **kwargs)
GET_SYSTEM_CPU_LOAD_PARAMS = {
    "type": "object",
    "properties": {
        "time_interval_seconds": {
            "type": "integer",
            "description": "The time interval in seconds to average CPU load over. Defaults to 1 second.",
            "default": 1
        }
    },
    "required": []
}

INITIATE_NETWORK_SCAN_PARAMS = {
    "type": "object",
    "properties": {
        "target_ip": {
            "type": "string",
            "description": "The IP address or hostname to scan (e.g., '192.168.1.1', 'example.com')."
        },
        "scan_type": {
            "type": "string",
            "description": "The type of scan to perform ('full_port_scan', 'ping_sweep', 'vulnerability_scan'). Defaults to 'ping_sweep'.",
            "enum": ["full_port_scan", "ping_sweep", "vulnerability_scan"],
            "default": "ping_sweep"
        }
    },
    "required": ["target_ip"]
}

DEPLOY_RECOVERY_PROTOCOL_PARAMS = {
    "type": "object",
    "properties": {
        "protocol_name": {
            "type": "string",
            "description": "The name of the recovery protocol to deploy (e.g., 'network_isolation', 'data_rollback')."
        },
        "target_system_id": {
            "type": "string",
            "description": "The ID of the system to apply the recovery protocol to."
        },
        "urgency_level": {
            "type": "string",
            "description": "The urgency level of the deployment ('low', 'medium', 'high', 'critical'). Defaults to 'medium'.",
            "enum": ["low", "medium", "high", "critical"],
            "default": "medium"
        }
    },
    "required": ["protocol_name", "target_system_id"]
}

UPDATE_RESOURCE_ALLOCATION_PARAMS = {
    "type": "object",
    "properties": {
        "resource_type": {
            "type": "string",
            "description": "The type of resource to adjust ('CPU', 'Memory', 'NetworkBandwidth')."
        },
        "target_agent_name": {
            "type": "string",
            "description": "The name of the agent whose resources are being adjusted."
        },
        "new_allocation_percentage": {
            "type": "number",
            "description": "The new percentage of allocation (e.g., 0.5 for 50%)."
        }
    },
    "required": ["resource_type", "target_agent_name", "new_allocation_percentage"]
}

GET_ENVIRONMENTAL_DATA_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "Specific geographical location to fetch data for (e.g., 'Arctic_Ice_Sheet')."
        },
        "data_type": {
            "type": "string",
            "description": "Type of environmental data to retrieve (e.g., 'temperature', 'humidity', 'all'). Defaults to 'all'.",
            "enum": ["temperature", "humidity", "air_quality", "water_level", "all"],
            "default": "all"
        }
    },
    "required": []
}

class ToolRegistry:
    """
    Manages a collection of callable tools available to agents.
    """
    def __init__(self):
        self._tools = {} # Use _tools to avoid conflict with register_tool method name
        self._initialize_default_tools() # Call the initialization method

    def _initialize_default_tools(self):
        """Register default simulated tools."""
        self.register_tool(Tool(
            name="get_system_cpu_load",
            description="Retrieves the current average CPU load of the system. Useful for monitoring resource usage.",
            parameters=GET_SYSTEM_CPU_LOAD_PARAMS,
            func=get_system_cpu_load_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="initiate_network_scan",
            description="Initiates a network scan on a specified IP address to check connectivity, open ports, or vulnerabilities.",
            parameters=INITIATE_NETWORK_SCAN_PARAMS,
            func=initiate_network_scan_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="deploy_recovery_protocol",
            description="Deploys a pre-defined recovery or mitigation protocol to a target system. Use in emergency or recovery scenarios.",
            parameters=DEPLOY_RECOVERY_PROTOCOL_PARAMS,
            func=deploy_recovery_protocol_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="update_resource_allocation",
            description="Adjusts the allocation of a specific resource (CPU, Memory, NetworkBandwidth) for a given agent.",
            parameters=UPDATE_RESOURCE_ALLOCATION_PARAMS,
            func=update_resource_allocation_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="get_environmental_data",
            description="Fetches real-time environmental data from a sensor array (e.g., temperature, humidity, air quality, water level).",
            parameters=GET_ENVIRONMENTAL_DATA_PARAMS, # Ensure GET_ENVIRONMENTAL_DATA_PARAMS is defined
            func=get_environmental_data_tool
        ))
        self.register_tool(Tool(
                name="analyze_threat_signature",
                description="Analyzes a known threat signature (like a CVE ID or virus name) to determine its risk level.",
                parameters={"type": "object", "properties": {
                    "signature": {"type": "string", "description": "The threat signature to analyze."},
                    "source_ip": {"type": "string", "description": "The source IP associated with the threat."}
                }, "required": ["signature"]},
                func=analyze_threat_signature_tool
        ))
            
        self.register_tool(Tool(
                name="isolate_network_segment",
                description="Isolates a specific network segment to prevent a threat from spreading.",
                parameters={"type": "object", "properties": {
                    "segment_id": {"type": "string", "description": "The ID of the network segment to isolate (e.g., 'WebServer_A')."},
                    "reason": {"type": "string", "description": "The reason for the isolation."}
                }, "required": ["segment_id", "reason"]},
                func=isolate_network_segment_tool
        ))
        print("[ToolRegistry] Default tools initialized.")

    def register_tool(self, tool: Tool):
        """Adds a tool to the registry."""
        if tool.name in self._tools:
            print(f"Warning: Tool '{tool.name}' already registered. Overwriting.")
        self._tools[tool.name] = tool
        print(f"[ToolRegistry] Registered tool: '{tool.name}'")

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Retrieves a specific tool by name."""
        return self._tools.get(tool_name)

    def get_all_tool_specs(self) -> list[dict]:
        """Returns a list of all registered tool specifications for LLM context."""
        return [tool.get_function_spec() for tool in self._tools.values()]

    
            