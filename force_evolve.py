import time
import json
from evolution_agent import EvolutionAgent
from tool_registry import ToolRegistry
from shared_models import OllamaLLMIntegration

registry = ToolRegistry()
llm_engine = OllamaLLMIntegration()

evolver = EvolutionAgent(
    tool_registry=registry, 
    approval_mode="autonomous", 
    gap_threshold=1,
    llm=llm_engine
)

gap_desc = "Perform a deep web security vulnerability scan on example.com and identify all open ports using an advanced port scanning tool."
evolver.record_capability_gap(
    description=gap_desc,
    context="Task: User wants to scan a target IP/Web Server for vulnerabilities.",
    attempted_tool="nmap_scan",
    failure_reason="tool_not_found"
)

print(f"Recorded gap: {gap_desc}. Triggering manual evolution cycle...")
evolver._trigger_evolution_cycle()

print("Evolution cycle finished. Current status:")
print(json.dumps(evolver.get_status(), indent=2))
