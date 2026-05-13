
import os
import sys
import json
from evolution_agent import EvolutionAgent
from tool_registry import ToolRegistry

def main():
    registry = ToolRegistry()
    evolver = EvolutionAgent(tool_registry=registry)
    
    tools_to_promote = ["get_current_price_of", "perform_deep_web_security"]
    
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evolved_tools",
    )
    q_dir = os.path.join(base_dir, "quarantine")
    a_dir = os.path.join(base_dir, "active")

    print("--- CVA Tool Promotion Utility ---")
    for tool_name in tools_to_promote:
        src_py = os.path.join(q_dir, f"{tool_name}.py")
        dst_py = os.path.join(a_dir, f"{tool_name}.py")
        
        if os.path.isfile(src_py):
            print(f"Promoting {tool_name} from quarantine...")
            success = evolver.promote_tool_from_quarantine(tool_name)
            if success:
                print(f"  [OK] {tool_name} promoted to evolved_tools/active/")
            else:
                print(f"  [ERROR] Failed to promote {tool_name}")
        elif os.path.isfile(dst_py):
            print(f"Tool {tool_name} is already in evolved_tools/active/.")
        else:
            print(f"  [ERROR] {tool_name} not found in quarantine OR active.")

    print("\nVerifying Registry Load...")
    # Simulate the registry boot reload
    registry.load_evolved_tools()
    
    available = registry.get_available_tools() # Returns Set[str]
    for tname in tools_to_promote:
        if tname in available:
            print(f"  [FOUND] {tname} is now ACTIVE in the registry.")
        else:
            print(f"  [MISSING] {tname} was NOT found in the registry.")

if __name__ == "__main__":
    main()
