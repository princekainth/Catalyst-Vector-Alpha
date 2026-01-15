
import sys
import os
import time
import logging
import threading
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from catalyst_vector_alpha import CatalystVectorAlpha
from agents import ProtoAgent
from curiosity_loop import CuriosityLoop
import prompts

# Setup Logging to Console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FieldTest")

def monitor_agent_thoughts(cva):
    """Periodically peek into agent brains."""
    print("\n--- 🧠 CORTEX MONITORING ACTIVE ---")
    while cva.is_running:
        with cva._agents_lock:
            for name, agent in cva.agent_instances.items():
                if hasattr(agent, 'current_intent'):
                    # Check for persona
                    persona_snippet = agent.persona[:50].replace('\n', ' ') if hasattr(agent, 'persona') else "Unknown"
                    print(f"[{name}] Mode: {agent.operational_mode} | Intent: {agent.current_intent[:60]}... | Persona: {persona_snippet}...")
        time.sleep(2)

from shared_models import MessageBus, EventMonitor
from tool_registry import ToolRegistry
from curiosity_loop import CuriosityLoop

def main():
    print("🚀 INITIALIZING CATALYST VECTOR ALPHA (LIVE SIMULATION)...")
    
    # 1. Initialize System Dependencies
    bus = MessageBus()
    monitor = EventMonitor() # EventMonitor takes no args
    registry = ToolRegistry()
    
    # 2. Initialize CVA with dependencies
    cva = CatalystVectorAlpha(message_bus=bus, tool_registry=registry, event_monitor=monitor)
    
    # Restart curiosity loop with fast cycle
    if hasattr(cva, 'curiosity_loop') and cva.curiosity_loop:
        print("🔄 Restarting Curiosity Loop with accelerated time...")
        cva.curiosity_loop.stop()
        cva.curiosity_loop = CuriosityLoop(cycle_time=2, orchestrator=cva)
        cva.curiosity_loop.idle_cycles = 4 # Start close to dreaming
        cva.curiosity_loop.start()

    # Start CVA in a separate thread
    cva_thread = threading.Thread(target=cva.run_cognitive_loop, kwargs={'tick_sleep': 1}, daemon=True)
    cva_thread.start()
    
    # Allow boot time
    time.sleep(5)
    
    # 2. Monitor Initial State
    print("\n🧐 OBSERVATION 1: INITIAL STATE & PERSONALITY")
    with cva._agents_lock:
        for name, agent in cva.agent_instances.items():
            print(f"   - Found Agent: {name} ({agent.eidos_spec.get('role')})")
            if hasattr(agent, 'persona'):
                print(f"     🗣️  Voice: {agent.persona.splitlines()[1].strip() if len(agent.persona.splitlines()) > 1 else agent.persona[:50]}")

    # 3. Stimulate: Request Reproduction (Mitosis)
    print("\n🧪 STIMULUS: TRIGGERING MITOSIS")
    print("   Injecting directive: 'SPAWN_DYNAMIC_AGENT' for 'Specialized Data Analysis'")
    
    directive = {
        "type": "SPAWN_DYNAMIC_AGENT",
        "purpose": "Perform deep analysis of system logs for security anomalies",
        "context": {"priority": "high"},
        "requester_agent": "Field_Researcher",
        "timestamp": time.time()
    }
    cva.inject_directives([directive])
    
    # Wait for birth
    time.sleep(5)
    print("\n🧐 OBSERVATION 2: POPULATION DYNAMICS")
    found_child = False
    with cva._agents_lock:
        for name in cva.agent_instances.keys():
            if "Dynamic_" in name:
                print(f"   👶 IT'S ALIVE! Found new agent: {name}")
                found_child = True
    
    if not found_child:
        print("   ⚠️  No child agent found yet. It might be processing or mocking is overriding factory.")

    # 4. Stimulate: Induce Dreaming (Idle State)
    print("\n💤 STIMULUS: INDUCING SLEEP CYCLE")
    print("   Stopping inputs. Watching Curiosity Loop...")
    
    # We wait for the loop to tick
    start_wait = time.time()
    dream_detected = False
    
    # We monitor logs/output for "Entering Dream State"
    # (In a real run we'd check internal state, here we sleep and hope the sped-up loop triggers)
    for i in range(10):
        if cva.curiosity_loop.idle_cycles == 0 and i > 2:
            # It reset, meaning it probably triggered! (Or explored)
            pass
        print(f"   ...tick (Idle Count: {cva.curiosity_loop.idle_cycles})")
        time.sleep(2)

    # 5. Shutdown
    print("\n🛑 ENDING SIMULATION")
    cva.stop()
    cva_thread.join(timeout=5)
    print("✅ SIMULATION COMPLETE")

if __name__ == "__main__":
    main()
