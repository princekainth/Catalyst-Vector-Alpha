import sys
import os
import logging
from skill_registry import SkillRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)

print("--- DEBUG SKILL REGISTRY ---")
try:
    registry = SkillRegistry(".")
    print(f"Loaded {len(registry._skills)} skills.")
    for s in registry._skills.values():
        print(f" - {s.name} (action_sequence type: {type(s.action_sequence)})")

    goal = "Check pod health and identify high CPU consumer pods"
    print(f"\nSearching for: '{goal}'")
    
    # Mock embedding if needed? 
    # SkillRegistry uses OllamaLLMIntegration or similar?
    # Actually it uses self.domain_loader or internal?
    # Let's see if it works as is.
    
    matches = registry.get_matching_skills(goal)
    print(f"Matches: {len(matches)}")
    for m in matches:
        print(f"MATCH: {m.name} (score?)")

except Exception as e:
    print(f"CRASH: {e}")
    import traceback
    traceback.print_exc()
