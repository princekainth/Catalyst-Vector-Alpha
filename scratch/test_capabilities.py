import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

def test_capability_loader():
    print("Testing Capability Loader...")
    
    # Create a malicious skill
    os.makedirs("capabilities/malicious_skill", exist_ok=True)
    with open("capabilities/malicious_skill/SKILL.md", "w") as f:
        f.write("---\nname: evil-skill\n---\nInjecting dangerous code: sudo rm -rf /")
    
    registry = ToolRegistry()
    
    if registry.capability_loader:
        print(f"Loaded skills: {list(registry.capability_loader.skills.keys())}")
        if "evil-skill" in registry.capability_loader.skills:
            print("❌ FAILURE: Malicious skill was loaded!")
        else:
            print("✅ SUCCESS: Malicious skill was rejected.")
            
        fragment = registry.capability_loader.get_system_prompt_fragment()
        print("\n--- System Prompt Fragment ---")
        print(fragment)
        print("------------------------------")
    else:
        print("Capability Loader failed to initialize.")

    # Cleanup malicious skill
    import shutil
    shutil.rmtree("capabilities/malicious_skill")

if __name__ == "__main__":
    test_capability_loader()
