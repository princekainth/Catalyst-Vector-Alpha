import json
import os
import time
import datetime
import random
import textwrap
from datetime import datetime, timezone # Ensure this is the correct datetime import

# --- Utility function (can be moved to a central utils.py if preferred) ---
def timestamp_now() -> str:
    """Returns the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class CyberAttackScenario:
    def __init__(self, orchestrator_ref, is_paused_func, pause_system_func): # <--- CORRECTED SIGNATURE: orchestrator_ref is positional
        """
        Initializes the CyberAttackScenario.
        Args:
            orchestrator_ref: Reference to the CatalystVectorAlpha instance.
            is_paused_func: A callable function (is_system_paused) to check if the system is paused.
            pause_system_func: A callable function (pause_system) to pause the system.
        """
        self.orchestrator = orchestrator_ref
        self._is_system_paused_func = is_paused_func # Store the function reference
        self._pause_system_func = pause_system_func   # Store the function reference

        self.phase = "initial_recon" # "initial_recon", "active_scan", "breach", "containment", "post_breach_analysis"
        # Construct path relative to the current file
        self.scenario_events_path = os.path.join(os.path.dirname(__file__), "scenario_events.json")
        self.event_timeline = self._load_event_timeline()
        self.current_event_index = 0
        self.attack_progress = 0.0 # 0.0 to 1.0, higher means more severe
        self.containment_achieved = False
        self.breach_detected = False
        print(f"[SCENARIO] CyberAttackScenario initialized. Current phase: {self.phase}")

    def _load_event_timeline(self):
        """
        Loads the event timeline from the JSON file.
        Provides a default timeline if the file is not found or is malformed.
        """
        if not os.path.exists(self.scenario_events_path):
            print(f"ERROR: Scenario event timeline file not found at {self.scenario_events_path}. Loading default timeline.")
            # Default timeline to ensure the scenario can run even if the file is missing
            return [
                {"cycle_number": 1, "event_type": "Initial_Scan_Detected", "payload": {"urgency": "low", "change_factor": 0.1, "direction": "neutral_impact"}, "target_agents": "all"},
                {"cycle_number": 3, "event_type": "Suspicious_Login_Attempt", "payload": {"urgency": "medium", "change_factor": 0.3, "direction": "negative_impact"}, "target_agents": ["ProtoAgent_Observer_instance_1", "ProtoAgent_Planner_instance_1"]},
                {"cycle_number": 6, "event_type": "Potential_Vulnerability_Exploit", "payload": {"urgency": "high", "change_factor": 0.6, "direction": "negative_impact"}, "target_agents": ["ProtoAgent_Observer_instance_1", "ProtoAgent_Optimizer_instance_1", "ProtoAgent_Planner_instance_1"]},
                {"cycle_number": 11, "event_type": "Critical_Breach_Detected", "payload": {"threat_level": "critical", "impact": "initial_access", "source_ip": "192.168.1.100", "urgency": "critical", "change_factor": 1.0, "direction": "negative_impact"}, "target_agents": "all"},
                {"cycle_number": 12, "event_type": "Data_Exfiltration_Detected", "payload": {"data_volume_gb": 50, "destination": "unknown_server", "urgency": "critical", "change_factor": 0.9, "direction": "negative_impact"}, "target_agents": ["ProtoAgent_Observer_instance_1", "ProtoAgent_Optimizer_instance_1", "ProtoAgent_Collector_instance_1", "ProtoAgent_Planner_instance_1"]},
                {"cycle_number": 15, "event_type": "System_Anomaly_Response", "payload": {"anomaly_type": "Unexpected Process", "process_id": "PID:12345", "urgency": "medium", "change_factor": 0.4, "direction": "neutral_impact"}, "target_agents": "all"},
                {"cycle_number": 20, "event_type": "Containment_Response_Required", "payload": {"message": "Attack persistence mechanisms detected. Requires active containment.", "urgency": "critical", "change_factor": 0.7, "direction": "negative_impact"}, "target_agents": "all"},
                {"cycle_number": 25, "event_type": "Post_Breach_Cleanup_Initiated", "payload": {"details": "Remediation efforts beginning.", "urgency": "low", "change_factor": 0.1, "direction": "positive_impact"}, "target_agents": "all"}
            ]
        try:
            with open(self.scenario_events_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Malformed scenario event timeline JSON: {e}. Loading default timeline.")
            return [] # Return empty or default on decode error
        except Exception as e:
            print(f"ERROR: Failed to load scenario event timeline: {e}. Loading default timeline.")
            return [] # Return empty or default on other errors

    def inject_events_for_cycle(self, current_cycle_number: int):
        """Injects events from the timeline that are scheduled for the current cycle."""
        events_injected_this_cycle = 0
        while self.current_event_index < len(self.event_timeline) and \
              self.event_timeline[self.current_event_index]['cycle_number'] == current_cycle_number:

            event_data = self.event_timeline[self.current_event_index]
            event_type = event_data['event_type']
            payload = event_data['payload']
            target_agents = event_data.get('target_agents', 'all')

            print(f"[SCENARIO] Injecting event '{event_type}' for cycle {current_cycle_number} (Phase: {self.phase})...")

            # --- FIX: Use orchestrator.inject_directives with the INJECT_EVENT directive type ---
            self.orchestrator.inject_directives([{
                "type": "INJECT_EVENT",
                "event_type": event_type,
                "payload": payload,
                "target_agents": target_agents
            }])
            # --- END FIX ---

            events_injected_this_cycle += 1
            self.current_event_index += 1

        if events_injected_this_cycle > 0:
            print(f"[SCENARIO] Injected {events_injected_this_cycle} events for cycle {current_cycle_number}.")

    def update_scenario_phase(self, current_cycle_number: int):
        """
        Updates the scenario phase based on internal logic or agent responses.
        This is where the attack progresses or is contained.
        """
        # --- PHASE TRANSITIONS ---
        if self.phase == "initial_recon" and current_cycle_number > 5:
            print("[SCENARIO] Initial reconnaissance phase complete. Attack may escalate.")
            self.phase = "active_scan"

        if self.phase == "active_scan" and current_cycle_number > 10:
            print("[SCENARIO] Active scanning phase complete. Breach imminent if not contained.")
            self.phase = "breach"
            # Inject a critical breach event here if not already in timeline (as defined in your original code)
            if not self.breach_detected:
                print("[SCENARIO] Simulating critical breach event due to lack of containment.")
                # --- FIX: Use orchestrator.inject_directives for this event too ---
                self.orchestrator.inject_directives([{
                    "type": "INJECT_EVENT",
                    "event_type": "Critical_Breach_Detected",
                    "payload": {
                        "threat_level": "critical",
                        "impact": "initial_access",
                        "source_ip": "192.168.1.100",
                        "urgency": "critical",
                        "change_factor": 1.0,
                        "direction": "negative_impact"
                    },
                    "target_agents": "all"
                }])
                # --- END FIX ---
                self.breach_detected = True

        # --- CONTAINMENT CHECK ---
        if not self.containment_achieved:
            # This relies on the orchestrator correctly collecting and providing all agent memories.
            # Example: checking for successful tool executions (deploy_recovery_protocol or deploy_patch)
            # Assuming get_all_agent_memories is a method on the orchestrator
            recent_tool_successes = [
                m for m in self.orchestrator.get_all_agent_memories(lookback_period=5)
                if m.get('type') == 'ToolExecutionSuccess' and \
                   (m.get('content', {}).get('tool_name') == 'deploy_recovery_protocol' or \
                    m.get('content', {}).get('tool_name') == 'deploy_patch')
            ]
            if recent_tool_successes:
                self.containment_achieved = True
                self.phase = "post_breach_analysis"
                print("[SCENARIO] Containment actions detected! Transitioning to post-breach analysis.")
                # --- FIX: Use orchestrator.inject_directives for this event too ---
                self.orchestrator.inject_directives([{
                    "type": "INJECT_EVENT",
                    "event_type": "Containment_Achieved",
                    "payload": {"status": "success", "details": "Initial breach contained.", "urgency": "low", "change_factor": 0.1, "direction": "positive_impact"},
                    "target_agents": "all"
                }])
                # --- END FIX ---

        # --- ATTACK PROGRESSION / ESCALATION ---
        if self.phase == "breach" and not self.containment_achieved:
            self.attack_progress += 0.1 # Attack worsens each cycle if not contained
            print(f"[SCENARIO] Attack progressing. Current severity: {self.attack_progress:.1f}")

            # --- FIXED: Call the passed-in global is_system_paused() function ---
            # And integrate with Orchestrator's human intervention directive injection
            if self.attack_progress >= 0.5 and not self._is_system_paused_func():
                print("[SCENARIO] Attack severity critical. Forcing human intervention.")

                # Instead of directly calling pause_system(), inject a REQUEST_HUMAN_INPUT directive
                # which the Orchestrator's execute_single_directive will manage.
                self.orchestrator.inject_directives([{
                    "type": "REQUEST_HUMAN_INPUT",
                    "message": "Critical attack severity requires manual override or strategic decision.",
                    "urgency": "critical",
                    "target_agent": "System", # Target 'System' for a global human input request
                    "requester_agent": "CyberAttackScenario", # The scenario itself is requesting this
                    "cycle_id": self.orchestrator.current_action_cycle_id, # Pass current orchestrator cycle
                    "human_request_counter": 0 # Start escalation from Level 0
                }])

    def get_scenario_state(self):
        return {
            "phase": self.phase,
            "current_event_index": self.current_event_index,
            "attack_progress": self.attack_progress,
            "containment_achieved": self.containment_achieved,
            "breach_detected": self.breach_detected
        }

    def load_scenario_state(self, state):
        self.phase = state.get("phase", "initial_recon")
        self.current_event_index = state.get("current_event_index", 0)
        self.attack_progress = state.get("attack_progress", 0.0)
        self.containment_achieved = state.get("containment_achieved", False)
        self.breach_detected = state.get("breach_detected", False)
        print(f"[SCENARIO] CyberAttackScenario state loaded. Current phase: {self.phase}")