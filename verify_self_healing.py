import os
import time
import sys
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

try:
    from tool_registry import tool_registry
    from evolution_agent import EvolutionAgent
    from core.self_healing_monitor import SelfHealingMonitor
    from shared_models import OllamaLLMIntegration
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# 1. Setup minimal environment
llm = OllamaLLMIntegration()
evolver = EvolutionAgent(tool_registry=tool_registry, llm=llm, approval_mode="autonomous")
monitor = SelfHealingMonitor(tool_registry=tool_registry, evolution_agent=evolver)

# Link evolver to registry
tool_registry.evolution_agent = evolver

print("\n--- [1] Triggering Failures for 'get_system_cpu_load' ---")
for i in range(5):
    res = tool_registry.safe_call("get_system_cpu_load", caller_agent="VerificationScript")
    print(f"Call {i+1} status: {res.get('status')} | Error: {res.get('error')}")

# 2. Check Circuit Breaker
with tool_registry._failure_lock:
    broken_until = tool_registry._broken_until.get("get_system_cpu_load", 0)
    
if broken_until > time.time():
    print(f"✅ Success: Circuit breaker tripped for 'get_system_cpu_load'.")
else:
    print(f"❌ Failure: Circuit breaker NOT tripped. Broken until: {broken_until}")

# 3. Proactive Health Scan
print("\n--- [2] Triggering Proactive Health Scan ---")
monitor.perform_health_scan()

# 4. Observe Evolution Agent Gaps
print("\n--- [3] Checking Evolution Agent Gaps ---")
gaps = evolver.capability_gaps
repair_gaps = [g for g in gaps if "REPAIR" in g['description']]
if repair_gaps:
    print(f"✅ Success: Found {len(repair_gaps)} REPAIR missions.")
    for g in repair_gaps:
        print(f"  - {g['description']}")
else:
    print("❌ Failure: No REPAIR missions found in Evolution Agent.")

# 5. Simulate Evolution Cycle (Manual trigger to avoid waiting for thread)
print("\n--- [4] Triggering Evolution Cycle for Repair ---")
evolver._trigger_evolution_cycle()

# 6. Verify Repair (Promotion from quarantine to active)
print("\n--- [5] Verifying Deployment ---")
# Check evolution history
history = evolver.evolution_history
if history and history[-1]['tool_name'] == 'get_system_cpu_load':
    print(f"✅ Success: 'get_system_cpu_load' was repaired and quarantined.")
    print(f"  - Status: {history[-1]['status']}")
    
    # Check if promoted (since we are in autonomous mode in the script's evolver)
    if history[-1]['status'] == 'promoted':
         print(f"✅ Success: Tool auto-promoted to active.")
    else:
         print(f"ℹ️ Tool is quarantined. (CVA_ALLOW_EVOLUTION_DEPLOY not set in shell)")
else:
    print("❌ Failure: Repair mission did not result in a tool evolution.")

print("\n--- Verification Script Complete ---")
