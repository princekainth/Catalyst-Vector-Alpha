# ==================================================================
#                      Consolidated Imports
# ==================================================================

# --- Standard Library ---
from __future__ import annotations
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
import collections # Import the full module to resolve NameError
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone, timedelta
from typing import Optional, Union, Tuple, List, Dict

# --- Third-Party Libraries ---
import yaml
import ollama
import textwrap
import psutil
import chromadb
import jsonschema
import numpy as np

# --- Project-Specific (Local Application) ---
from core import (
    MessageBus, EventMonitor, MemeticKernel, ISLSchemaValidator,
    OllamaLLMIntegration, SovereignGradient,
    timestamp_now, mark_override_processed,
    _get_recent_log_entries as get_recent_log_entries
)
# CORRECTED: Import ToolRegistry from its own file
from tool_registry import ToolRegistry

# CORRECTED: Add the new ProtoAgent_Security
from agents import (
    ProtoAgent,
    ProtoAgent_Observer,
    ProtoAgent_Optimizer,
    ProtoAgent_Collector,
    ProtoAgent_Planner,
    ProtoAgent_Security
)
from scenarios.cyber_attack import CyberAttackScenario
import prompts
import llm_schemas
from ccn_monitor_mock import MockCCNMonitor
from tools import (
    get_system_cpu_load_tool,
    initiate_network_scan_tool,
    deploy_recovery_protocol_tool,
    update_resource_allocation_tool,
    get_environmental_data_tool,
)

# --- Logging Setup (Must be after imports) ---
# It's good practice to get a specific logger rather than configuring the root.
logger = logging.getLogger("CatalystLogger")
# The basicConfig should ideally be in your main execution block, but this works.
logging.basicConfig(level=logging.INFO, handlers=[], force=True)

# --- Communication Channel ---
class MessageBus:
    def __init__(self):
        self.messages = {}
        self.catalyst_vector_ref = None # Will be set by CatalystVectorAlpha

    # --- FIX: Updated the method signature and logic ---
    def send_message(self, sender: str, recipient: str, message_type: str, content: any, 
                     task_description: str = None, status: str = "pending", cycle_id: str = None):
        """Sends a structured message from one agent to another."""
        if recipient not in self.messages:
            self.messages[recipient] = []
        
        self.messages[recipient].append({
            "sender": sender,
            "message_type": message_type,
            "content": content,
            "task_description": task_description,
            "status": status,
            "cycle_id": cycle_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    # --- END FIX ---

    def send_directive(self, directive):
        if self.catalyst_vector_ref:
            self.catalyst_vector_ref.dynamic_directive_queue.append(directive)

    def get_messages_for_agent(self, agent_name):
        messages_for_agent = self.messages.get(agent_name, [])
        self.messages[agent_name] = [] # Clear inbox after retrieval
        return messages_for_agent
        
class MetaSystemMonitor:
    def __init__(self):
        self.system_patterns = {}

    def analyze_system_state(self, all_agents):
        """
        Detect system-wide stagnation patterns by observing individual agent states.
        Returns a recommended system-level intervention string if a pattern is found, else None.
        """
        stuck_patterns = collections.defaultdict(int)
        total_agents = len(all_agents)

        if total_agents == 0:
            return None

        stuck_agent_count = 0
        for agent in all_agents:
            if agent.stagnation_adaptation_attempts >= 2:
                stuck_agent_count += 1
                # Try to infer a category from the current intent for aggregation
                if "REPAIR_" in agent.current_intent:
                    category = agent.current_intent.split(":")[0].replace("REPAIR_", "").lower()
                elif "DIAGNOSE" in agent.current_intent.upper():
                    category = "diagnostic"
                elif "MONITOR" in agent.current_intent.upper():
                    category = "monitoring"
                else:
                    category = "unknown"
                stuck_patterns[category] += 1

        if stuck_agent_count > total_agents * 0.6 and total_agents > 0: # Ensure total_agents is not zero before multiplication
            return self._recommend_system_intervention(stuck_patterns)

        return None

    def _recommend_system_intervention(self, patterns):
        """
        Recommends a system-level intervention based on the dominant patterns
        of agent stagnation.
        """
        if not patterns:
            return "GENERAL_SYSTEM_REBOOT: No specific pattern, generalized system reset."

        sorted_patterns = sorted(patterns.items(), key=lambda item: item[1], reverse=True)
        dominant_pattern = sorted_patterns[0][0]

        if 'diagnostic' in dominant_pattern:
            return "REFRAME_GOALS: Switch system objectives or provide a new overarching directive to break diagnostic loops."
        elif 'monitoring' in dominant_pattern:
            return "INJECT_RESOURCES: Provide new tools, data sources, or external context to break monitoring loops."
        elif 'environmental_impact' in dominant_pattern:
             return "RECONFIGURE_INTERFACE: Evaluate and reconfigure interfaces for environmental interaction."
        elif 'tool_effectiveness' in dominant_pattern:
            return "AUDIT_TOOLS: System-wide audit and potential replacement/upgrade of core tools."
        elif 'creative_block' in dominant_pattern:
            return "INJECT_NOVELTY: Introduce novel stimuli, data, or unexpected challenges."
        else:
            return "SHUFFLE_ROLES: Temporarily reassign agent responsibilities and re-evaluate eidos definitions."


# --- Swarm Protocol ---
class SwarmProtocol:
    def __init__(self,
                 swarm_name: str,
                 initial_goal: str,
                 initial_members: list,
                 consensus_mechanism: str,
                 description: str,
                 catalyst_vector_ref: 'CatalystVectorAlpha', # Reference to the orchestrator instance
                 swarm_state_file_path: str, # Full path to this specific swarm's state file
                 loaded_state: Optional[dict] = None):

        self.name = swarm_name
        self.goal = initial_goal
        self.members = set(initial_members)
        self.consensus_mechanism = consensus_mechanism
        self.description = description
        self.sovereign_gradient = None # Initialize sovereign gradient for swarm (will be loaded or created)
        self.catalyst_vector_ref = catalyst_vector_ref # Store reference to the orchestrator

        # Store the provided state file path
        self.swarm_state_file_full_path_path = swarm_state_file_path

        # Get necessary components from the orchestrator for MemeticKernel initialization
        # Ensure CatalystVectorAlpha instance has these attributes available and correctly set
        orchestrator_log_sink = self.catalyst_vector_ref.external_log_sink
        orchestrator_chroma_db_path = self.catalyst_vector_ref.chroma_db_full_path
        orchestrator_persistence_dir = self.catalyst_vector_ref.persistence_dir

        # Load MemeticKernel state if available from loaded_state
        loaded_kernel_state = loaded_state.get('memetic_kernel', {}) if loaded_state else {}
        loaded_memories_for_kernel = loaded_kernel_state.get('memories', []) # Assuming 'memories' is the key for the deque list

        # --- CORRECTED: MemeticKernel instantiation for SwarmProtocol ---
        # Pass all required arguments to MemeticKernel.__init__
        self.memetic_kernel = MemeticKernel(
            agent_name=f"SwarmKernel_{swarm_name}",  # Unique name for the swarm's kernel
            external_log_sink=orchestrator_log_sink,      # <--- CRITICAL FIX: Pass external_log_sink
            chroma_db_path=orchestrator_chroma_db_path,  # <--- CRITICAL FIX: Pass chroma_db_path
            persistence_dir=orchestrator_persistence_dir, # <--- CRITICAL FIX: Pass persistence_dir

            config={'goal': self.goal, 'members': list(self.members), 'description': self.description}, # Pass initial config
            loaded_memories=loaded_memories_for_kernel, # Pass loaded_memories
            memetic_archive_path=loaded_kernel_state.get('memetic_archive_path') # Pass if saved, else MemeticKernel uses its default
        )
        # --- END CORRECTED ---

        # Load other SwarmProtocol attributes if loading from state
        if loaded_state:
            # Re-load attributes from loaded_state (some might have been set by arguments, so keep the .get() with default)
            self.goal = loaded_state.get('goal', self.goal)
            self.members = set(loaded_state.get('members', [])) # Convert back to set on load
            self.consensus_mechanism = loaded_state.get('consensus_mechanism', self.consensus_mechanism)
            self.description = loaded_state.get('description', self.description)
            
            if loaded_state.get('sovereign_gradient'):
                # Assuming SovereignGradient.from_state accepts a dict and returns a SovereignGradient object
                self.sovereign_gradient = SovereignGradient.from_state(loaded_state['sovereign_gradient'])
            else:
                # Initialize if not found in loaded state or malformed
                self.sovereign_gradient = SovereignGradient(target_entity_name=self.name, config={})

            # If MemeticKernel has a separate load_state method for its *internal* state, call it here.
            # (Its constructor above already took initial state, but this is good if it has more complex loading)
            # if loaded_kernel_state:
            #     self.memetic_kernel.load_state(loaded_kernel_state) # Call its load_state if it has one
            
            # Log successful reload via orchestrator's logger
            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_RELOADED",
                self.name,
                f"Swarm '{self.name}' reloaded from persistence.",
                {"goal": self.goal, "members_count": len(self.members), "file": self.swarm_state_file_full_path_path}, # Add file path to details
                level='info'
            )
                    
        else:
            # Initial logging for a newly formed swarm (not loaded from state)
            self.memetic_kernel.add_memory("SwarmFormation", f"Swarm '{self.name}' established. Goal: '{self.goal}'. Consensus: {self.consensus_mechanism}")
            print(f"[SwarmProtocol] Swarm '{self.name}' established. Goal: '{self.goal}'. Consensus: {self.consensus_mechanism}")
            
            # Log new swarm formation via orchestrator's logger
            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_FORMED",
                self.name,
                f"Swarm '{self.name}' established.",
                {"goal": self.goal, "consensus": self.consensus_mechanism, "initial_members_count": len(initial_members)},
                level='info'
            )

    def add_member(self, agent_name):
        if agent_name not in self.members:
            self.members.add(agent_name)
            self.memetic_kernel.add_memory("MemberAdded", f"Agent '{agent_name}' joined the swarm.")
            self.memetic_kernel.config['members'] = list(self.members) # Ensure config also uses list
            if self.catalyst_vector_ref: # Use the orchestrator's logger via _log_swarm_activity
                self.catalyst_vector_ref._log_swarm_activity(
                    "SWARM_MEMBER_ADDED", # event_type
                    self.name,            # source
                    f"Agent '{agent_name}' joined swarm '{self.name}'.", # description
                    {"agent": agent_name, "swarm": self.name}, # details
                    level='info'          # level
                )
                
    def set_goal(self, new_goal):
        old_goal = self.goal
        self.goal = new_goal
        self.memetic_kernel.add_memory("GoalUpdate", f"Swarm goal updated to: '{new_goal}'.")
        self.memetic_kernel.config['goal'] = new_goal
        if self.catalyst_vector_ref: # Use the orchestrator's logger via _log_swarm_activity
            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_GOAL_UPDATED", # event_type
                self.name,            # source
                f"Swarm goal updated.", # description
                {"old_goal": old_goal, "new_goal": new_goal}, # details
                level='info'          # level
            )
            
    def set_sovereign_gradient(self, new_gradient: 'SovereignGradient'):
        """Sets the sovereign gradient for this swarm."""
        old_gradient_state = self.sovereign_gradient.get_state() if self.sovereign_gradient else None
        self.sovereign_gradient = new_gradient
        self.memetic_kernel.config['gradient'] = new_gradient.get_state()
        self.memetic_kernel.add_memory("GradientUpdate", f"Sovereign gradient set for swarm: '{new_gradient.autonomy_vector}'.")
        # FIX: Corrected direct call to CatalystVectorAlpha's _log_swarm_activity
        if self.catalyst_vector_ref:
             self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_GRADIENT_SET", # event_type
                self.name,            # source
                f"Sovereign gradient set.", # description
                {"old_gradient": old_gradient_state, "new_gradient": new_gradient.get_state()}, # details
                level='info'          # level
            )
            
    def coordinate_task(self, task_description):
        final_task_description = task_description
        gradient_compliant = True
        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            gradient_compliant = compliant
            final_task_description = adjusted_task
            if not compliant:
                print(f"  [SovereignGradient] Swarm task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
        
        self.memetic_kernel.add_memory("TaskCoordination", f"Swarm '{self.name}' coordinating task: '{final_task_description}' (Compliant: {gradient_compliant}) among {len(self.members)} members (conceptual).")
        print(f"[SwarmProtocol] Swarm '{self.name}' coordinating task: '{final_task_description}' among {len(self.members)} members (conceptual).")
        if self.catalyst_vector_ref:
            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_TASK_COORDINATION", # event_type
                self.name,                 # source
                f"Coordinating task: '{final_task_description}'.", # description
                {"task": final_task_description, "members_count": len(self.members), "compliant": gradient_compliant}, # details
                level='info'               # level
            )

    def get_state(self):
        return {
            'name': self.name,
            'goal': self.goal,
            'members': list(self.members), # FIX: Convert set to list for JSON serialization
            'consensus_mechanism': self.consensus_mechanism,
            'description': self.description,
            'sovereign_gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None,
            'memetic_kernel': self.memetic_kernel.get_state()
        }

    def save_state(self):
        """Saves the swarm's current state to its designated JSON file."""
        try:
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(self.swarm_state_file_full_path_path), exist_ok=True)
            with open(self.swarm_state_file_full_path_path, 'w') as f:
                json.dump(self.get_state(), f, indent=2)
            self.external_log_sink.info(f"Swarm '{self.name}' state saved to {self.swarm_state_file_full_path_path}.")
        except Exception as e:
            self.external_log_sink.error(f"Failed to save state for swarm '{self.name}' to {self.swarm_state_file_full_path_path}: {e}")
            # Optionally re-raise or handle more specifically

# --- Catalyst Vector Alpha (Main Orchestrator) ---
class CatalystVectorAlpha:
    def __init__(self,
                 message_bus: 'MessageBus',
                 tool_registry: 'ToolRegistry',
                 event_monitor: 'EventMonitor',
                 external_log_sink: Optional[logging.Logger] = None,
                 persistence_dir: str = "persistence_data",
                 swarm_activity_log: str = "logs/swarm_activity.jsonl",
                 system_pause_file: str = "system_pause.flag",
                 swarm_state_file: str = "swarm_state.json",
                 paused_agents_file: str = "paused_agents.json",
                 isl_schema_path: str = "isl_schema.yaml",
                 chroma_db_path: str = "chroma_db",
                 intent_override_prefix: str = "intent_override_",
                 ccn_monitor_interface=None
                 ):
        # --- Core Dependencies ---
        self.message_bus = message_bus
        self.tool_registry = tool_registry
        self.event_monitor = event_monitor
        self.external_log_sink = external_log_sink if external_log_sink is not None else logging.getLogger(__name__)
        self.ccn_monitor_interface = ccn_monitor_interface
        self.swarm_state = {}

        # --- Base Persistence Directory (Must be set first) ---
        self.persistence_dir = persistence_dir
        os.makedirs(self.persistence_dir, exist_ok=True)

        # --- Construct and store full paths for all files/directories ---
        self.swarm_activity_log_full_path = os.path.join(self.persistence_dir, swarm_activity_log)
        self.system_pause_file_full_path = os.path.join(self.persistence_dir, system_pause_file)
        self.swarm_state_file_full_path = os.path.join(self.persistence_dir, swarm_state_file)
        self.paused_agents_file_full_path = os.path.join(self.persistence_dir, paused_agents_file)
        self.chroma_db_full_path = os.path.join(self.persistence_dir, chroma_db_path)
        
        # Ensure directories for logs and chromaDB also exist
        os.makedirs(os.path.dirname(self.swarm_activity_log_full_path), exist_ok=True)
        os.makedirs(self.chroma_db_full_path, exist_ok=True)

        # Initialize ISL Schema Validator
        self.isl_schema_validator = ISLSchemaValidator(isl_schema_path)

        # Store the intent override prefix
        self.intent_override_prefix = intent_override_prefix

        # --- Internal Registries and Queues ---
        self.eidos_registry = {}
        self.agent_instances = {}
        self.swarm_protocols = {}
        self.dynamic_directive_queue = deque()
        self.current_action_cycle_id = None
        
        # --- System State and Flags ---
        self.is_running = True
        self.is_paused = False
        self.pending_human_interventions = {}
        self._no_pattern_reports_count = defaultdict(int)

        # --- Orchestrator Self-Reference for MessageBus ---
        self.message_bus.catalyst_vector_ref = self

        # --- Scenario Initialization ---
        self.scenario = CyberAttackScenario(
            self,
            is_paused_func=self.is_system_paused,
            pause_system_func=self.pause_system
        )
        self.active_scenario = self.scenario # Assign the scenario to the active_scenario attribute

        # --- System-wide Thresholds for Self-Regulation ---
        self.SWARM_RESET_THRESHOLD = 3
        self.NO_PATTERN_AGENT_THRESHOLD = 0.6

        # --- NEW: Instantiate MetaSystemMonitor ---
        self.meta_monitor = MetaSystemMonitor()
        
        # --- Log initial setup completion ---
        self._log_swarm_activity("SYSTEM_INITIALIZATION", "CatalystVectorAlpha",
                                 "Catalyst Vector Alpha system is starting its core initialization.",
                                 {"persistence_dir": self.persistence_dir, "isl_schema": self.isl_schema_validator.schema_path},
                                 level='info')

        print(f"Successfully loaded ISL Schema: {self.isl_schema_validator.schema_path}")
        print("[IP-Integration] The Eidos Protocol System is initiating, demonstrating the Gemini™ wordmark in its functionality.")
        
        self._directive_handlers = self._initialize_directive_handlers()

        # The initial manifest content (kept for reference, usually loaded from a file/string)
        self.isl_manifest_content = textwrap.dedent("""
        directives:
          - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Observer
            eidos_spec:
              role: data_observer
              initial_intent: Continuously observe diverse data streams and report findings.
              location: Local_Alpha_Testbed_ZoneA

          - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Optimizer
            eidos_spec:
              role: resource_optimizer
              initial_intent: Optimize simulated resource allocation efficiency based on inputs.
              location: Local_Alpha_Testbed_Central
                            
          - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Collector
            eidos_spec: # <--- CORRECTED: ADDED eidos_spec
              role: data_collector
              initial_intent: Efficiently collect and process environmental data.
              location: Local_Alpha_Testbed_ZoneB

          - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Planner
            eidos_spec:
              role: strategic_planner
              initial_intent: Strategically plan and inject directives to achieve high-level goals.
              location: Central_Control_Node
                                                    
          - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Security
            eidos_spec:
              role: security_analyst
              initial_intent: Monitor all system events for security threats and anomalies.
              location: Security_Operations_Center                                          
                                                
          # 2. SPAWN_AGENT_INSTANCE: Create instances.
          - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Observer
            instance_name: ProtoAgent_Observer_instance_1
            initial_task: Prepare for data analysis

          - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Optimizer
            instance_name: ProtoAgent_Optimizer_instance_1
            initial_task: Monitor incoming reports

          - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Planner
            instance_name: ProtoAgent_Planner_instance_1
            initial_task: Initialize planning modules
                            
          - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Collector
            instance_name: ProtoAgent_Collector_instance_1
            initial_task: Prepare data collection modules
          
          - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Security
            instance_name: ProtoAgent_Security_instance_1
            initial_task: Perform initial system security baseline check.
                                    
          # 3. AGENT_PERFORM_TASK: The LLM summarization task (stays).
          - type: AGENT_PERFORM_TASK
            agent_name: ProtoAgent_Observer_instance_1
            task_description: Summarize recent environmental data report on polar ice melt.
            text_content: |
              A recent environmental report highlights alarming rates of polar ice melt, exceeding previous projections.
              Satellite data from the Arctic indicates a 15% reduction in multi-year ice thickness compared to the decade
              prior. In Antarctica, the Thwaites Glacier, often called the "Doomsday Glacier," shows accelerated retreat,
              contributing significantly to global sea-level rise. Ocean temperatures in polar regions are increasing,
              leading to feedback loops where warmer water eroding ice from below. The report emphasizes the urgency of
              reducing greenhouse gas emissions to mitigate irreversible impacts on global climate systems and coastal communities.
            cycle_id: llm_summary_task_001
            reporting_agents: ProtoAgent_Optimizer_instance_1
            on_success: log_llm_summary

          # 4. INITIATE_PLANNING_CYCLE: Give the Planner agent a high-level goal.
          - type: INITIATE_PLANNING_CYCLE
            planner_agent_name: ProtoAgent_Planner_instance_1
            high_level_goal: Ensure comprehensive environmental stability and optimize resource distribution.
            cycle_id: planner_cycle_001
            
         # 5. INJECT_EVENT: Simulate an external environmental alert. (DISABLED FOR TESTING)
          # - type: INJECT_EVENT
          #   event_type: "Environmental_Sensor_Alert"
          #   payload:
          #     sensor_id: "ENV-007"
          #     location: "Arctic_Ice_Sheet"
          #     data:
          #       temperature_anomaly: "+3.5C"
          #       ice_thickness_reduction: "2.1m"
          #     change_factor: 0.8
          #     urgency: "critical"
          #     direction: "negative_impact"
          #   target_agents: ["ProtoAgent_Observer_instance_1", "ProtoAgent_Optimizer_instance_1"]
        """)
        self._directive_handlers = self._initialize_directive_handlers()
        
    def _log_swarm_activity(self, event_type, source, description, details=None, level='info'):
        """Logs significant swarm activity to a central file and pushes to external sink."""
        # This method seems to be fine, but the log calls throughout the code need to be fixed
        # to match its signature and use the correct file path attributes.
        max_desc_len = 500
        truncated_description = (description[:max_desc_len] + "...") if len(description) > max_desc_len else description

        truncated_details = {}
        if details:
            for k, v in details.items():
                if isinstance(v, str) and len(v) > 500:
                    truncated_details[k] = (v[:500] + "...")
                elif isinstance(v, list) and k == 'patterns':
                    truncated_patterns = []
                    for pattern_item in v:
                        if isinstance(pattern_item, str) and len(pattern_item) > 200:
                            truncated_patterns.append(pattern_item[:200] + "...")
                        else:
                            truncated_patterns.append(str(pattern_item)[:200])
                    truncated_details[k] = truncated_patterns
                else:
                    truncated_details[k] = v

        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_type": event_type,
            "source": source,
            "description": truncated_description,
            "details": truncated_details if truncated_details else {}
        }
        
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            log_entry['details'].update({"cpu_usage": cpu_usage, "memory_usage": memory_usage})
            print(f"[System] Resource Usage: CPU {cpu_usage}%, Memory {memory_usage}%")
        except Exception as e:
            log_entry['details'].update({"cpu_usage": "N/A", "memory_usage": "N/A", "resource_error": str(e)})

        try:
            # FIX: Use the corrected full path attribute
            with open(self.swarm_activity_log_full_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"ERROR: Exception in _log_swarm_activity file writing for {source}: {e}")
            if self.external_log_sink:
                self.external_log_sink.error(f"Failed to write to swarm activity file for {source}: {e}", extra={"error_details":str(e)})

        if self.external_log_sink:
            try:
                if level == 'error':
                    self.external_log_sink.error(json.dumps(log_entry))
                elif level == 'warning':
                    self.external_log_sink.warning(json.dumps(log_entry))
                elif level == 'debug':
                    self.external_log_sink.debug(json.dumps(log_entry))
                else:
                    self.external_log_sink.info(json.dumps(log_entry))
            except Exception as e:
                print(f"ERROR: Failed to push swarm log to external sink for {source}: {e}")

    def load_or_create_swarm_state(self):
        """Loads the overall swarm state from a JSON file or initializes a new one."""
        file_path = self.swarm_state_file_full_path
        # ... (rest of method is fine, just needs to use the corrected file path)
        self._log_swarm_activity(
            "SYSTEM_STATE_LOADING",
            "CatalystVectorAlpha",
            f"Attempting to load previous system state from '{self.persistence_dir}'.",
            {"load_path": file_path},
            level='info'
        )
        print(f"--- Loading previous system state from '{self.persistence_dir}' ---")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.swarm_state = json.load(f)
                self._log_swarm_activity(
                    "SYSTEM_STATE_LOADED",
                    "CatalystVectorAlpha",
                    f"Successfully loaded previous swarm state from '{file_path}'.",
                    {"loaded_path": file_path},
                    level='info'
                )
                print(f"  Successfully loaded previous swarm state from '{file_path}'.")
            except (json.JSONDecodeError, Exception) as e:
                self.external_log_sink.error(
                    f"Failed to load swarm state from '{file_path}': {e}",
                    extra={"agent": "CatalystVectorAlpha"}
                )
                self._log_swarm_activity(
                    "SYSTEM_STATE_LOAD_FAILED",
                    "CatalystVectorAlpha",
                    f"Failed to load swarm state. Starting fresh.",
                    {"error": str(e)},
                    level='error'
                )
                print(f"  Error loading swarm state: {e}. Starting fresh.")
                self.swarm_state = self._initialize_default_swarm_state()
        else:
            self._log_swarm_activity(
                "SYSTEM_STATE_INFO",
                "CatalystVectorAlpha",
                "No previous swarm states found, starting fresh.",
                {},
                level='info'
            )
            print("  No previous swarm states found.")
            self.swarm_state = self._initialize_default_swarm_state()

        print("--- Finished loading previous system state ---")

    def _initialize_default_swarm_state(self) -> dict:
        """Initializes and returns a default empty swarm state structure."""
        return {
            "current_cycle_id": "initial",
            "agent_states": {},
            "global_state_variables": {},
            "last_saved_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "system_start_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }

    def save_swarm_state(self):
        """Saves the current overall swarm state to a JSON file."""
        file_path = self.swarm_state_file_full_path
        # ... (rest of method is fine, just needs to use the corrected file path)
        self.swarm_state["last_saved_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with open(file_path, 'w') as f:
                json.dump(self.swarm_state, f, indent=4)
            self._log_swarm_activity(
                "SYSTEM_SAVE_SUCCESS",
                "CatalystVectorAlpha",
                f"Overall swarm state saved to {file_path}.",
                {"file": file_path},
                level='info'
            )
            print(f"  Overall swarm state saved to {file_path}.")
        except Exception as e:
            self.external_log_sink.error(
                f"Failed to save swarm state to '{file_path}': {e}",
                extra={"agent": "CatalystVectorAlpha"}
            )
            self._log_swarm_activity(
                "SYSTEM_SAVE_FAILED",
                "CatalystVectorAlpha",
                f"Failed to save swarm state to '{file_path}'.",
                {"error": str(e)},
                level='error'
            )
            print(f"  Error saving swarm state: {e}")
        print("--- System state saved ---")

    def is_system_paused(self):
        return os.path.exists(self.system_pause_file_full_path)

    def get_pending_human_intervention_requests(self):
        """Returns a list of currently pending human intervention requests."""
        return list(self.pending_human_interventions.values())

    def handle_human_response(self, request_id, response_data):
        # ... (This method seems to have an indentation error and needs to be placed correctly)
        response_file_path = os.path.join(self.persistence_dir, f"control_human_response_{request_id}.json")
        original_request_details = self.pending_human_interventions.get(request_id)
        if not original_request_details:
            print(f"ERROR: Received response for unknown/already cleared request ID: {request_id}. Skipping global clear.")
            self._log_swarm_activity(
                "HUMAN_RESPONSE_NO_MATCH",
                "Dashboard",
                f"Received response for unknown/old request ID: {request_id}.",
                {"request_id": request_id, "response_data_preview": str(response_data)[:100]},
                level='warning'
            )
            return

        requester = original_request_details.get('requester_agent', 'System')
        
        try:
            os.makedirs(self.persistence_dir, exist_ok=True)
            with open(response_file_path, 'w') as f:
                json.dump(response_data, f, indent=2)
            print(f"[CatalystVectorAlpha] Human response for request ID '{request_id}' written to {response_file_path}")
            
            if os.path.exists(self.system_pause_file_full_path):
                os.remove(self.system_pause_file_full_path)
                print(f"[CatalystVectorAlpha] System pause flag removed: {self.system_pause_file_full_path}")
                self.is_paused = False

            requests_to_remove = []
            for pending_req_id, pending_req_details in list(self.pending_human_interventions.items()):
                if pending_req_id == request_id:
                    requests_to_remove.append(pending_req_id)
                    continue

                if pending_req_details.get('requester_agent') == requester and \
                   pending_req_details.get('human_request_counter', 0) >= 1:
                    requests_to_remove.append(pending_req_id)

            for req_id_to_remove in requests_to_remove:
                if req_id_to_remove in self.pending_human_interventions:
                    del self.pending_human_interventions[req_id_to_remove]
                    self._log_swarm_activity(
                        "HUMAN_INTERVENTION_REQUEST_CLEARED_BATCH",
                        "CatalystVectorAlpha",
                        f"Human intervention request {req_id_to_remove} cleared from pending list (part of batch clear for {request_id}).",
                        {"request_id": req_id_to_remove, "answered_request_id": request_id},
                        level='info'
                    )

            self._log_swarm_activity(
                "HUMAN_RESPONSE_FILE_WRITTEN_SUCCESS",
                "Dashboard",
                f"Human response file successfully written for request ID: {request_id}. System will process in next cycle.",
                {"request_id": request_id, "response_data_preview": str(response_data)[:100]},
                level='info'
            )

            if requester in self.agent_instances:
                req_agent = self.agent_instances[requester]
                if hasattr(req_agent, 'human_input_received') and callable(getattr(req_agent, 'human_input_received')):
                    response_for_agent = response_data.copy()
                    response_for_agent['request_id'] = request_id
                    response_for_agent['original_directive'] = original_request_details
                    response_for_agent['problem_id'] = original_request_details.get('problem_id')
                    req_agent.human_input_received(response_for_agent)
                else:
                    print(f"  Warning: Requesting agent '{requester}' does not have a 'human_input_received' method.")
            else:
                print(f"  Warning: Requesting agent '{requester}' not found to inform about human input.")
        except json.JSONDecodeError as e:
            print(f"ERROR: Malformed human response file: {response_file_path}")
            self._log_swarm_activity("HUMAN_INPUT_ERROR", "CatalystVectorAlpha",
                                     f"Malformed human response file '{os.path.basename(response_file_path)}'. Error: {e}",
                                     {"file": response_file_path, "error": str(e)},
                                     level='error')
        except Exception as e:
            print(f"ERROR: Failed to process human response file: {response_file_path} - {e}")
            self._log_swarm_activity("HUMAN_INPUT_ERROR", "CatalystVectorAlpha",
                                     f"Failed to process human response file '{os.path.basename(response_file_path)}'. Error: {e}",
                                     {"file": response_file_path, "error": str(e)},
                                     level='error')

    def _add_human_intervention_request(self, request_data: dict):
        """
        Adds a new human intervention request to the pending list.
        """
        if 'id' not in request_data:
            request_data['id'] = f"HUMANREQ-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
        self.pending_human_interventions[request_data['id']] = request_data
        
        self.external_log_sink.info(
            f"Human intervention request added: {request_data.get('id', 'N/A')}",
            extra={"event_type": "HUMAN_INTERVENTION_REQUEST_ADDED",
                   "source": request_data.get("requester_agent", "System"),
                   "description": request_data.get("message", "No message provided")[:100],
                   "request_id": request_data.get('id')}
        )

    def get_pending_human_intervention_requests(self):
        """Returns a list of all currently pending human intervention requests."""
        return list(self.pending_human_interventions.values())

    def _initialize_directive_handlers(self):
        """Maps directive types to their corresponding handler methods."""
        return {
            'ASSERT_AGENT_EIDOS': self._handle_assert_agent_eidos,
            'ESTABLISH_SWARM_EIDOS': self._handle_establish_swarm_eidos,
            'SPAWN_AGENT_INSTANCE': self._handle_spawn_agent_instance,
            'ADD_AGENT_TO_SWARM': self._handle_add_agent_to_swarm,
            'ASSERT_GRADIENT_TRAJECTORY': self._handle_assert_gradient_trajectory,
            'CATALYZE_TRANSFORMATION': self._handle_catalyze_transformation,
            'BROADCAST_SWARM_INTENT': self._handle_broadcast_swarm_intent,
            # This is the central task handler.
            'AGENT_PERFORM_TASK': self._handle_agent_perform_task, 
            'SWARM_COORDINATE_TASK': self._handle_swarm_coordinate_task,
            'REPORTING_AGENT_SUMMARIZE': self._handle_reporting_agent_summarize,
            'AGENT_ANALYZE_AND_ADAPT': self._handle_agent_analyze_and_adapt,
            'BROADCAST_COMMAND': self._handle_broadcast_command,
            'INITIATE_PLANNING_CYCLE': self._handle_initiate_planning_cycle,
            'INJECT_EVENT': self._handle_inject_event,
            'REQUEST_HUMAN_INPUT': self._handle_request_human_input,
        }

    def _execute_single_directive(self, directive: dict):
        directive_type = directive['type']
        handler = self._directive_handlers.get(directive_type)
        if handler:
            try:
                directive['cycle_id'] = directive.get('cycle_id', self.current_action_cycle_id)
                handler(directive)
            except ValueError as ve:
                print(f"  ERROR: {ve}")
                self._log_swarm_activity("DIRECTIVE_VALIDATION_FAILED", "CatalystVectorAlpha",
                                         f"Directive validation failed for {directive_type}: {ve}",
                                         {"error": str(ve), "directive": directive}, level='error')
            except Exception as e:
                print(f"ERROR: Exception while processing directive {directive_type}: {e}")
                self._log_swarm_activity("DIRECTIVE_PROCESSING_ERROR", "CatalystVectorAlpha",
                                         f"Error processing directive {directive_type}",
                                         {"error": str(e), "directive": directive}, level='error')
        else:
            print(f"  Unknown Directive: {directive_type}. (Alpha stage limitation)")
            self._log_swarm_activity("UNKNOWN_DIRECTIVE_TYPE", "CatalystVectorAlpha",
                                     f"Unknown directive encountered: {directive_type}.",
                                     {"directive_type": directive_type, "full_directive": directive}, level='warning')
        print("\n--- Directives Execution Complete ---")

    def _handle_assert_agent_eidos(self, directive: dict):
        eidos_name = directive['eidos_name']
        eidos_spec = directive['eidos_spec']
        eidos_spec['eidos_name'] = eidos_name
        if eidos_name not in self.eidos_registry:
            self.eidos_registry[eidos_name] = eidos_spec
            self._log_swarm_activity("EIDOS_ASSERTED", "CatalystVectorAlpha",
                                     f"Defined EIDOS for '{eidos_name}'.", {"eidos_name": eidos_name}, level='info')
            print(f"  ASSERT_AGENT_EIDOS: Defined EIDOS for '{eidos_name}'.")
        else:
            print(f"  ASSERT_AGENT_EIDOS: EIDOS '{eidos_name}' already exists. Reusing.")
            self._log_swarm_activity("EIDOS_REUSED", "CatalystVectorAlpha",
                                     f"EIDOS '{eidos_name}' already exists, reusing.", {"eidos_name": eidos_name}, level='info')
    
    def _handle_establish_swarm_eidos(self, directive: dict):
        swarm_name = directive['swarm_name']
        initial_goal = directive.get('initial_goal', 'No specified goal')
        initial_members = directive.get('initial_members', [])
        consensus_mechanism = directive.get('consensus_mechanism', 'SimpleMajorityVote')
        description = directive.get('description', 'A collective intelligence.')

        if swarm_name in self.swarm_protocols:
            print(f"  ESTABLISH_SWARM_EIDOS: Swarm '{swarm_name}' already active. Reusing.")
            swarm = self.swarm_protocols[swarm_name]
            self._log_swarm_activity("SWARM_REUSED", "CatalystVectorAlpha",
                                     f"Swarm '{swarm_name}' already active, reusing.", {"swarm_name": swarm_name}, level='info')
        else:
            print(f"  ESTABLISH_SWARM_EIDOS: Establishing new swarm '{swarm_name}'.")
            specific_swarm_state_file_path = os.path.join(self.persistence_dir, f"swarm_state_{swarm_name}.json")

            swarm = SwarmProtocol(
                swarm_name=swarm_name,
                initial_goal=initial_goal,
                initial_members=initial_members,
                consensus_mechanism=consensus_mechanism,
                description=description,
                catalyst_vector_ref=self,
                swarm_state_file_path=specific_swarm_state_file_path
            )
            self.swarm_protocols[swarm_name] = swarm
            self._log_swarm_activity("SWARM_ESTABLISHED", "CatalystVectorAlpha",
                                     f"Established Swarm '{swarm_name}' with goal: '{initial_goal}'.",
                                     {"swarm": swarm_name, "initial_goal": initial_goal, "file_path": specific_swarm_state_file_path}, level='info')
        swarm.coordinate_task(directive.get('task_description', 'Initial swarm formation and goal orientation'))

    def _handle_swarm_coordinate_task(self, directive: dict):
        """Handles the SWARM_COORDINATE_TASK directive."""
        swarm_name = directive['swarm_name']
        task_description = directive['task_description']
        
        if swarm_name not in self.swarm_protocols:
            raise ValueError(f"Swarm '{swarm_name}' not found for task coordination.")
        
        swarm = self.swarm_protocols[swarm_name]
        swarm.coordinate_task(task_description)
        self._log_swarm_activity("SWARM_COORDINATED_TASK", "CatalystVectorAlpha",
                                f"Swarm '{swarm_name}' coordinated task '{task_description}'.",
                                {"swarm": swarm_name, "task": task_description})

    def _handle_spawn_agent_instance(self, directive: dict):
        eidos_name = directive['eidos_name']
        instance_name = directive['instance_name']
        task_description = directive.get('initial_task', 'Run diagnostic checks')
        if eidos_name not in self.eidos_registry:
            raise ValueError(f"EIDOS '{eidos_name}' not asserted yet. Define it first using ASSERT_AGENT_EIDOS.")

        eidos_spec = self.eidos_registry[eidos_name]
        agent = None

        if instance_name in self.agent_instances:
            agent = self.agent_instances[instance_name]
            print(f"  SPAWN_AGENT_INSTANCE: Agent '{instance_name}' already exists. Reusing existing instance.")
            self._log_swarm_activity("AGENT_REUSED", "CatalystVectorAlpha",
                                     f"Agent '{instance_name}' already existed, reusing.",
                                     {"agent_name": instance_name, "eidos_name": eidos_name}, level='info')
        else:
            print(f"  SPAWN_AGENT_INSTANCE: Spawning new agent '{instance_name}'.")
            try:
                agent = self._create_agent_instance(
                    name=instance_name,
                    eidos_name=eidos_name,
                    eidos_spec=eidos_spec,
                    message_bus=self.message_bus,
                    tool_registry=self.tool_registry,
                    event_monitor=self.event_monitor,
                    external_log_sink=self.external_log_sink,
                    chroma_db_path=self.chroma_db_full_path,
                    persistence_dir=self.persistence_dir,
                    paused_agents_file_path=self.paused_agents_file_full_path,
                    loaded_state=self.swarm_state.get('agent_states', {}).get(instance_name)
                )

                self.agent_instances[instance_name] = agent
                self._log_swarm_activity("AGENT_SPAWNED", "CatalystVectorAlpha",
                                         f"New agent '{instance_name}' spawned.",
                                         {"agent_name": instance_name, "eidos_name": eidos_name, "context": eidos_spec.get('location', 'Unknown')}, level='info')
                
                # FIX: Assign the initial task as the agent's current intent
                agent.update_intent(task_description)

                # Now, perform the initial task and reflect
                outcome, _, _, _ = agent.perform_task(task_description, cycle_id=self.current_action_cycle_id)

                if hasattr(agent, 'memetic_kernel') and agent.memetic_kernel:
                    reflection = agent.memetic_kernel.reflect()
                    print(f"  [MemeticKernel] {agent.name} reflects: '{reflection}'")
                else:
                    print(f"  [MemeticKernel] {agent.name} has no MemeticKernel or it's not initialized for reflection.")

            except Exception as e:
                import traceback
                print(f"\nCRITICAL DEBUG ERROR: Exception during agent '{instance_name}' post-spawn setup: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self._log_swarm_activity("CRITICAL_AGENT_SPAWN_ERROR", "CatalystVectorAlpha",
                                         f"Agent '{instance_name}' failed post-spawn setup: {str(e)}",
                                         {"agent_name": instance_name, "error": str(e), "directive": directive}, level='error')
                raise
    
    def _handle_add_agent_to_swarm(self, directive: dict):
        """Handles the ADD_AGENT_TO_SWARM directive."""
        swarm_name = directive['swarm_name']
        agent_name_to_add = directive['agent_name']

        if swarm_name not in self.swarm_protocols:
            raise ValueError(f"Swarm '{swarm_name}' not established. Cannot add agent '{agent_name_to_add}'.")
        if agent_name_to_add not in self.agent_instances:
            raise ValueError(f"Agent instance '{agent_name_to_add}' not found. Cannot add to swarm.")
        
        self.swarm_protocols[swarm_name].add_member(agent_name_to_add)
        self.agent_instances[agent_name_to_add].join_swarm(swarm_name)
        self._log_swarm_activity("AGENT_ADDED_TO_SWARM", "CatalystVectorAlpha",
                                 f"Agent '{agent_name_to_add}' added to swarm '{swarm_name}'.",
                                 {"agent": agent_name_to_add, "swarm": swarm_name}, level='info')
    
    def _handle_assert_gradient_trajectory(self, directive: dict):
        """Handles the ASSERT_GRADIENT_TRAJECTORY directive."""
        target_type = directive['target_type']
        target_ref = directive['target_ref']
        
        target_obj = None
        if target_type == 'Agent' and target_ref in self.agent_instances:
            target_obj = self.agent_instances[target_ref]
        elif target_type == 'Swarm' and target_ref in self.swarm_protocols:
            target_obj = self.swarm_protocols[target_ref]
        else:
            raise ValueError(f"Target entity '{target_ref}' (type {target_type}) not found for ASSERT_GRADIENT_TRAJECTORY.")
        
        gradient_config = {
            'autonomy_vector': directive.get('autonomy_vector', 'General self-governance'),
            'ethical_constraints': directive.get('ethical_constraints', []),
            'self_correction_protocol': directive.get('self_correction_protocol', 'BasicCorrection'),
            'override_threshold': directive.get('override_threshold', 0.0)
        }
        new_gradient = SovereignGradient(target_ref, gradient_config)
        target_obj.set_sovereign_gradient(new_gradient)
        
        self._log_swarm_activity("GRADIENT_ASSERTED", "CatalystVectorAlpha",
                                 f"Sovereign Gradient asserted for {target_type} '{target_ref}'.",
                                 {"entity": target_ref, "type": target_type, "autonomy_vector": new_gradient.autonomy_vector}, level='info')
        print(f"  ASSERT_GRADIENT_TRAJECTORY: Set gradient for {target_type} '{target_ref}' to '{new_gradient.autonomy_vector}'.")

    def _handle_catalyze_transformation(self, directive: dict):
        """Handles the CATALYZE_TRANSFORMATION directive."""
        target_agent_instance_name = directive['target_agent_instance']
        new_initial_intent = directive.get('new_initial_intent')
        new_description = directive.get('new_description')
        new_memetic_kernel_config_updates = directive.get('new_memetic_kernel_config_updates')

        if target_agent_instance_name not in self.agent_instances:
            raise ValueError(f"Target agent instance '{target_agent_instance_name}' not found for CATALYZE_TRANSFORMATION.")
        
        target_agent = self.agent_instances[target_agent_instance_name]
        print(f"  CATALYZE_TRANSFORMATION: Initiating self-transformation for '{target_agent_instance_name}'.")
        target_agent.catalyze_transformation(
            new_initial_intent=new_initial_intent,
            new_description=new_description,
            new_memetic_kernel_config_updates=new_memetic_kernel_config_updates
        )
        self._log_swarm_activity("AGENT_TRANSFORMATION", "CatalystVectorAlpha",
                                 f"Agent '{target_agent_instance_name}' transformed.",
                                 {"agent": target_agent_instance_name, "updates": {"intent": new_initial_intent, "description": new_description, "mk_updates": new_memetic_kernel_config_updates}}, level='info')
        print(f"  CATALYZE_TRANSFORMATION: Transformation directive processed for '{target_agent_instance_name}'.")
    
    def _handle_broadcast_swarm_intent(self, directive: dict):
        """Handles the BROADCAST_SWARM_INTENT directive."""
        swarm_name = directive['swarm_name']
        broadcast_intent_content = directive['broadcast_intent']
        alignment_threshold = directive.get('alignment_threshold', 0.7)

        if swarm_name not in self.swarm_protocols:
            raise ValueError(f"Swarm '{swarm_name}' not found for BROADCAST_SWARM_INTENT.")
        
        swarm = self.swarm_protocols[swarm_name]
        print(f"  BROADCAST_SWARM_INTENT: Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.")
        self._log_swarm_activity("SWARM_INTENT_BROADCAST", "CatalystVectorAlpha",
                                 f"Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.",
                                 {"swarm": swarm_name, "intent": broadcast_intent_content, "threshold": alignment_threshold}, level='info')
        
        for agent_ref in swarm.members:
            if agent_ref in self.agent_instances:
                agent = self.agent_instances[agent_ref]
                agent.process_broadcast_intent(broadcast_intent_content, alignment_threshold)
            else:
                print(f"  Warning: Agent '{agent_ref}' not found in instance list, skipping broadcast.")
                self._log_swarm_activity("AGENT_NOT_FOUND_FOR_BROADCAST", "CatalystVectorAlpha",
                                         f"Agent '{agent_ref}' not found for intent broadcast to swarm '{swarm_name}'.",
                                         {"agent": agent_ref, "swarm": swarm_name}, level='warning')
        self._log_swarm_activity("BROADCAST_PROCESSED_COMPLETE", "CatalystVectorAlpha",
                                 f"Broadcast processed for swarm '{swarm_name}'.", {"swarm": swarm_name}, level='info')


    def _create_agent_instance(self,
                                name: str,
                                eidos_name: str,
                                eidos_spec: dict,
                                message_bus: 'MessageBus',
                                tool_registry: 'ToolRegistry',
                                event_monitor: 'EventMonitor',
                                external_log_sink: logging.Logger,
                                chroma_db_path: str,
                                persistence_dir: str,
                                paused_agents_file_path: str,
                                loaded_state: Optional[dict] = None
                                ) -> 'ProtoAgent':
        """
        Helper method to create and return a specific agent instance based on its EIDOS (role).
        Ensures all necessary arguments are passed to the agent's __init__ method.
        """
        # Fix: Ensure all agent classes are imported at the top of the file using absolute paths
        # For example: from agents.proto_agent_planner import ProtoAgent_Planner
        # This prevents the "relative import with no parent package" error.
        agent_class_map = {
            'ProtoAgent_Observer': ProtoAgent_Observer,
            'ProtoAgent_Collector': ProtoAgent_Collector,
            'ProtoAgent_Optimizer': ProtoAgent_Optimizer,
            'ProtoAgent_Planner': ProtoAgent_Planner
        }

        agent_class = agent_class_map.get(eidos_name)

        if not agent_class:
            raise ValueError(f"Unsupported EIDOS type: {eidos_name}. No corresponding agent class found for spawning.")

        # Prepare common arguments for all agent constructors
        agent_init_kwargs = {
            "name": name,
            "eidos_spec": eidos_spec,
            "message_bus": message_bus,
            "tool_registry": tool_registry,
            "event_monitor": event_monitor,
            "external_log_sink": external_log_sink,
            "chroma_db_path": chroma_db_path,
            "persistence_dir": persistence_dir,
            "paused_agents_file_path": paused_agents_file_path,
            "loaded_state": loaded_state,
        }

        # --- Conditional injection of ccn_monitor_interface for ProtoAgent_Planner ---
        if eidos_name == "ProtoAgent_Planner":
            if self.ccn_monitor_interface:
                agent_init_kwargs["ccn_monitor_interface"] = self.ccn_monitor_interface
                self.external_log_sink.debug(f"Injecting ccn_monitor_interface into {name} (Planner).", extra={"agent": "CatalystVectorAlpha"})
            else:
                self.external_log_sink.warning(f"No ccn_monitor_interface provided to CatalystVectorAlpha. Planner agent '{name}' will default to MockCCNMonitor.", extra={"agent": "CatalystVectorAlpha"})

        # Pass arguments to the correct agent class based on the eidos_name
        new_agent = agent_class(**agent_init_kwargs)
        return new_agent
    
    def get_state(self):
        """Returns the current state of the CatalystVectorAlpha system for persistence."""
        
        # Helper to convert non-serializable types for JSON (can be defined outside class for reuse)
        def convert_to_serializable(obj):
            if isinstance(obj, deque):
                return list(obj)
            if isinstance(obj, defaultdict):
                return dict(obj) # Convert defaultdict to regular dict
            # Add other non-serializable types if you encounter them (e.g., custom objects with no .get_state())
            return obj

        state = {
            'current_action_cycle_id': self.current_action_cycle_id,
            'eidos_registry': self.eidos_registry, # Assuming dict of serializable data
            'dynamic_directive_queue': list(self.dynamic_directive_queue), # Correctly converted
            'scenario_state': self.scenario.get_scenario_state() if self.scenario else None, # Assuming CyberAttackScenario.get_scenario_state is serializable
            'pending_human_interventions': self.pending_human_interventions, # Assuming this dict is serializable
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            '_no_pattern_reports_count': self._no_pattern_reports_count, # Assuming this dict is serializable
            'active_scenario': self.active_scenario, # Include if it's part of the state (assuming serializable)
            'SWARM_RESET_THRESHOLD': self.SWARM_RESET_THRESHOLD, # Assuming serializable constant
            'NO_PATTERN_AGENT_THRESHOLD': self.NO_PATTERN_AGENT_THRESHOLD, # Assuming serializable constant
            # FIX: Include EventMonitor's state
            'event_monitor_state': self.event_monitor.get_state(), 
            # FIX: Ensure swarm_protocols' values are also processed for non-serializable types
            'swarm_protocols': {name: convert_to_serializable(swarm.get_state()) for name, swarm in self.swarm_protocols.items()} if self.swarm_protocols else {},
            'agent_instances': {}
        }

        # Handle agent instances
        for agent_name, agent_instance in self.agent_instances.items():
            if hasattr(agent_instance, 'get_state'): # Prefer agent's own get_state if available
                agent_state = agent_instance.get_state()
            else: # Fallback to __dict__ but filter and convert non-serializable objects
                agent_state = agent_instance.__dict__.copy()
                # Apply conversion to values within agent's __dict__
                for key, value in agent_state.items():
                    agent_state[key] = convert_to_serializable(value)
                # Filter out objects that are inherently non-serializable or problematic
                agent_state = {
                    k: v for k, v in agent_state.items() 
                    if not callable(v) and not isinstance(v, (logging.Logger, MessageBus, ToolRegistry, EventMonitor, type(self), type(agent_instance.ollama_inference_model))) # Add other non-serializable types/references
                }
            state['agent_instances'][agent_name] = agent_state
        
        return state

    def _handle_agent_perform_task(self, directive: dict):
        """Handles the AGENT_PERFORM_TASK directive."""
        agent_name = directive['agent_name']
        task_description = directive['task_description']
        reporting_agents_ref = directive.get('reporting_agents', [])
        text_content = directive.get('text_content', '')
        task_type = directive.get('task_type', 'GenericTask')
        
        if agent_name not in self.agent_instances:
            self._log_swarm_activity("AGENT_NOT_FOUND_FOR_TASK", "CatalystVectorAlpha",
                                    f"Agent '{agent_name}' not found for AGENT_PERFORM_TASK.",
                                    {"agent": agent_name, "task": task_description}, level='error')
            raise ValueError(f"Agent '{agent_name}' not found for AGENT_PERFORM_TASK.")
        
        if isinstance(reporting_agents_ref, str):
            reporting_agents_list = [reporting_agents_ref]
        else:
            reporting_agents_list = reporting_agents_ref

        agent = self.agent_instances[agent_name]
        print(f"  AGENT_PERFORM_TASK: Agent '{agent_name}' performing task: '{task_description}'.")
        
        # THE FIX: Unpack all four values from agent.perform_task to match the new signature.
        outcome, failure_reason, report_content, _ = agent.perform_task(
            task_description,
            cycle_id=self.current_action_cycle_id,
            reporting_agents=reporting_agents_list,
            context_info=directive.get('context_info'),
            text_content=text_content,
            task_type=task_type
        )
        
        # After execution, reflect and log.
        if hasattr(agent, 'memetic_kernel') and agent.memetic_kernel:
            reflection = agent.memetic_kernel.reflect()
            print(f"  [MemeticKernel] {agent.name} reflects: '{reflection}'")
        else:
            print(f"  [MemeticKernel] {agent.name} has no MemeticKernel or it's not initialized for reflection (post-task).")

        log_level = 'info' if outcome == 'completed' else 'warning'
        self._log_swarm_activity("AGENT_TASK_PERFORMED", agent_name,
                                f"Agent '{agent_name}' completed task '{task_description}' with outcome: {outcome}.",
                                {"agent": agent_name, "task": task_description, "outcome": outcome, "task_type": task_type, "report_content": report_content},
                                level=log_level)

        def _handle_swarm_coordinate_task(self, directive: dict):
            """Handles the SWARM_COORDINATE_TASK directive."""
            # [Original location: _execute_single_directive, lines ~3951-3968]
            swarm_name = directive['swarm_name']
            task_description = directive['task_description']
            
            if swarm_name not in self.swarm_protocols:
                raise ValueError(f"Swarm '{swarm_name}' not found for task coordination.")
            
            swarm = self.swarm_protocols[swarm_name]
            swarm.coordinate_task(task_description)
            self._log_swarm_activity("SWARM_COORDINATED_TASK", "CatalystVectorAlpha",
                                    f"Swarm '{swarm_name}' coordinated task '{task_description}'.",
                                    {"swarm": swarm_name, "task": task_description})

    def _handle_reporting_agent_summarize(self, directive: dict):
        """Handles the REPORTING_AGENT_SUMMARIZE directive."""
        # [Original location: _execute_single_directive, lines ~3971-3995]
        reporting_agent_name_from_manifest = directive['reporting_agent_name']
        cycle_id_to_summarize = directive.get('cycle_id', None)
        
        if reporting_agent_name_from_manifest not in self.agent_instances:
            raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' not found for REPORTING_AGENT_SUMMARIZE.")
        
        agent = self.agent_instances[reporting_agent_name_from_manifest]
        if not isinstance(agent, ProtoAgent_Observer):
            raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' is not an Observer. Only Observer agents can summarize reports.")
        
        print(f"  REPORTING_AGENT_SUMMARIZE: Agent '{reporting_agent_name_from_manifest}' summarizing reports for cycle '{cycle_id_to_summarize}'.")
        agent.summarize_received_reports(cycle_id=cycle_id_to_summarize)
        self._log_swarm_activity("AGENT_REPORT_SUMMARIZED", "CatalystVectorAlpha",
                                 f"Agent '{reporting_agent_name_from_manifest}' summarized reports.",
                                 {"agent": reporting_agent_name_from_manifest, "cycle_id": cycle_id_to_summarize})

    def _handle_agent_analyze_and_adapt(self, directive: dict):
        """Handles the AGENT_ANALYZE_AND_ADAPT directive."""
        # [Original location: _execute_single_directive, lines ~3998-4011]
        agent_name = directive['agent_name']
        if agent_name not in self.agent_instances:
            raise ValueError(f"Agent '{agent_name}' not found for AGENT_ANALYZE_AND_ADAPT.")
        
        agent = self.agent_instances[agent_name]
        print(f"  AGENT_ANALYZE_AND_ADAPT: Agent '{agent_name}' performing reflexive analysis and adaptation.")
        agent.analyze_and_adapt()
        self._log_swarm_activity("AGENT_ANALYZE_ADAPT", "CatalystVectorAlpha",
                                 f"Agent '{agent_name}' performed analysis and adaptation.",
                                 {"agent": agent_name})

    def _handle_broadcast_command(self, directive: dict):
        """Handles the BROADCAST_COMMAND directive."""
        # [Original location: _execute_single_directive, lines ~4014-4045]
        target_agent = directive['target_agent']
        command_type = directive['command_type']
        command_params = directive.get('command_params', {})
        
        if target_agent not in self.agent_instances:
            raise ValueError(f"Target agent '{target_agent}' not found for BROADCAST_COMMAND.")
        
        agent = self.agent_instances[target_agent]
        print(f"  BROADCAST_COMMAND: Agent '{target_agent}' received command '{command_type}' with params: {command_params}.")
        if hasattr(agent, 'process_command') and callable(getattr(agent, 'process_command')):
            agent.process_command(command_type, command_params)
        else:
            print(f"  Warning: Agent '{target_agent}' does not support 'process_command' method. Command skipped.")
            self._log_swarm_activity("AGENT_COMMAND_SKIPPED", "CatalystVectorAlpha",
                                     f"Agent '{target_agent}' does not support command '{command_type}'.",
                                     {"agent": target_agent, "command_type": command_type, "params": command_params})
        
        self._log_swarm_activity("AGENT_COMMANDED", "CatalystVectorAlpha",
                                 f"Agent '{target_agent}' received command '{command_type}'.",
                                 {"agent": target_agent, "command_type": command_type, "params": command_params})
    
    def _handle_initiate_planning_cycle(self, directive: dict):
        planner_agent_name = directive.get('planner_agent_name')
        high_level_goal = directive.get('high_level_goal')
        cycle_id = directive.get('cycle_id', self.current_action_cycle_id)

        if not planner_agent_name or not high_level_goal:
            raise ValueError("INITIATE_PLANNING_CYCLE directive missing 'planner_agent_name' or 'high_level_goal'.")
        planner_agent = self.agent_instances.get(planner_agent_name)
        if not planner_agent:
            raise ValueError(f"Planner agent '{planner_agent_name}' not found.")

        print(f"  INITIATE_PLANNING_CYCLE: Assigning high-level goal to Planner '{planner_agent_name}'.")
        planner_agent.perform_task(
            task_description=high_level_goal,
            cycle_id=cycle_id,
            reporting_agents=[planner_agent_name],
            task_type="StrategicPlanning"
        )
        self._log_swarm_activity("PLANNING_CYCLE_INITIATED", "CatalystVectorAlpha",
                                 f"Planner '{planner_agent_name}' assigned new high-level goal: '{high_level_goal}'.",
                                 {"planner": planner_agent_name, "goal": high_level_goal, "cycle_id": cycle_id}, level='info')



    def _handle_inject_event(self, directive: dict):
        """
        Processes a directive to inject a new event into the system,
        logging it and broadcasting it to all active agents.
        """
        # Extract all necessary details from the directive dictionary
        event_type = directive.get('event_type', 'UnknownEvent')
        payload = directive.get('payload', {})
        # Generate a unique event ID if one isn't provided in the directive
        event_id = directive.get('event_id', f"EVT-{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}")

        # --- THIS IS THE FIX ---
        # The original code was likely missing the 'event_id' and 'payload' arguments.
        # This corrected call now passes all three required arguments to the log_event function.
        self.event_monitor.log_event(
            event_type=event_type,
            event_id=event_id,
            payload=payload
        )
        # --- END FIX ---

        # Log the injection to the main swarm activity log for system-wide tracking
        self._log_swarm_activity("EVENT_INJECTED", "CatalystVectorAlpha",
                                f"Injected event '{event_type}' (ID: {event_id}) to all agents.",
                                {"event_type": event_type, "event_id": event_id, "payload": payload, "targets": "all"})

        # Prepare the full event object to broadcast to agents
        event_to_broadcast = {
            'event_id': event_id,
            'type': event_type,
            'payload': payload,
            'timestamp': timestamp_now()
        }

        # Send the event to every active agent
        for agent in self.agent_instances.values():
            if hasattr(agent, 'receive_event'):
                agent.receive_event(event_to_broadcast)

    def pause_system(self, reason: str = "System initiating self-pause due to critical condition."):
        """
        Creates a flag file to signal the system to pause its cognitive loop.
        This is a direct pause mechanism, typically for critical situations.
        """
        try:
            # CORRECTED: Use the full path attribute here.
            os.makedirs(os.path.dirname(self.system_pause_file_full_path), exist_ok=True)
            with open(self.system_pause_file_full_path, 'w') as f:
                f.write(reason) # Write the reason into the file
            
            print(f"\n!!! SYSTEM PAUSED: {reason} !!!")
            self._log_swarm_activity(
                "SYSTEM_PAUSED_CRITICAL",
                "CatalystVectorAlpha",
                f"System has been paused. Reason: {reason}",
                {"reason": reason},
                level='critical'
            )
            self.is_running = False # Stop the main cognitive loop
        except Exception as e:
            print(f"ERROR: Failed to create system pause file: {e}")
            self._log_swarm_activity("SYSTEM_PAUSE_ERROR", "CatalystVectorAlpha",
                                    f"Failed to initiate system pause. Error: {e}",
                                    {"error": str(e)}, level='error')
            
    def unpause_system(self, reason: str = "System unpaused by explicit command or human input."):
        """
        Removes the system pause flag file, allowing the cognitive loop to resume.
        Includes a reason for logging.
        """
        if os.path.exists(self.system_pause_file_full_path):
            try:
                os.remove(self.system_pause_file_full_path)
                print(f"\n--- SYSTEM UNPAUSED ({reason}) ---") # Added reason to print
                self._log_swarm_activity(
                    "SYSTEM_UNPAUSED",
                    "CatalystVectorAlpha",
                    f"System has been unpaused. Reason: {reason}", # Using the reason here
                    {"reason": reason}, # Passing reason in details
                    level='info'
                )
                self.is_running = True # Allow the main cognitive loop to continue
            except Exception as e:
                print(f"ERROR: Failed to remove system pause file: {e}")
                self._log_swarm_activity("SYSTEM_UNPAUSE_ERROR", "CatalystVectorAlpha",
                                          f"Failed to unpause system. Error: {e}",
                                          {"error": str(e)}, level='error')
        else:
            print(f"System not paused. No flag file found: {self.system_pause_file_full_path}")
            self._log_swarm_activity("SYSTEM_ALREADY_UNPAUSED", "CatalystVectorAlpha",
                                      "Attempted to unpause system, but no pause flag was found.",
                                      {}, level='warning')

    def _handle_request_human_input(self, directive: dict):
        message = directive['message']
        urgency = directive.get('urgency', 'medium')
        target_agent = directive.get('target_agent', 'System')
        request_cycle_id = directive.get('cycle_id', self.current_action_cycle_id)
        human_request_counter = directive.get('human_request_counter', 0)
        requester = directive.get('requester_agent', target_agent)

        request_id = directive.get('request_id')
        if not request_id:
            request_id = f"HUMANREQ-{timestamp_now().replace(':', '-').replace('Z', '')}_{human_request_counter}_{requester}_{uuid.uuid4().hex[:4]}"
            self.external_log_sink.warning(f"REQUEST_HUMAN_INPUT directive missing request_id. Generated: {request_id}",
                                           extra={"event_type": "MISSING_REQUEST_ID", "directive": directive})
            directive['request_id'] = request_id

        self.pending_human_interventions[request_id] = {
            "id": request_id,
            "message": message,
            "urgency": urgency,
            "target_agent": target_agent,
            "requester_agent": requester,
            "cycle_id": request_cycle_id,
            "human_request_counter": human_request_counter,
            "timestamp": timestamp_now(),
            "status": "pending"
        }

        if human_request_counter == 0: # Level 1: Initial request / Peer Review
            print(f"\n!!! HUMAN INTERVENTION REQUESTED (Initial Request, Urgency: {urgency.upper()}) !!!")
            print(f"!!! From: {requester} (Target: {target_agent}) !!!")
            print(f"!!! Message: {message} !!!")
            print(f"!!! Please review logs for cycle {request_cycle_id} (Request ID: {request_id}) !!!")

            self._log_swarm_activity(
                "HUMAN_INPUT_REQUESTED_LEVEL1",
                "CatalystVectorAlpha",
                f"Level 1: Human input requested from {requester}. Initiating peer review.",
                {"message": message, "urgency": urgency, "requester": requester, "target_agent": target_agent, "cycle_id": request_cycle_id, "request_id": request_id},
                level='info'
            )

            peer_review_message = f"Urgent: {requester} is requesting human input due to '{message}'. Please analyze situation related to cycle {request_cycle_id}."

            # Send peer review messages (if agents exist)
            if 'ProtoAgent_Planner_instance_1' in self.agent_instances:
                self.agent_instances['ProtoAgent_Planner_instance_1'].send_message(
                    'ProtoAgent_Planner_instance_1',
                    'PeerReviewRequest',
                    peer_review_message,
                    None,
                    "pending",
                    request_cycle_id
                )
            if 'ProtoAgent_Observer_instance_1' in self.agent_instances and \
            target_agent != 'ProtoAgent_Observer_instance_1':
                self.agent_instances['ProtoAgent_Observer_instance_1'].send_message(
                    'ProtoAgent_Observer_instance_1',
                    'PeerReviewRequest',
                    peer_review_message,
                    None,
                    "pending",
                    request_cycle_id
                )
            # Re-inject directive for next check (Level 2)
            next_directive = directive.copy()
            next_directive["human_request_counter"] = 1 # Increment counter for next level
            self.inject_directives([next_directive])
            print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 2) for next cycle check.")

        elif human_request_counter == 1: # Level 2: Human Response Check
            # Correctly define response_file_path using self.persistence_dir
            response_file_path = os.path.join(self.persistence_dir, f"control_human_response_{request_id}.json")

            if os.path.exists(response_file_path) and os.path.getsize(response_file_path) > 0: # Check if human response file exists and is not empty
                try:
                    with open(response_file_path, 'r') as f:
                        human_response_from_file = json.load(f)

                    print(f"\n--- HUMAN RESPONSE RECEIVED (from {requester}, Urgency: {urgency.upper()}) ---")
                    print(f"Message: {message}")
                    print(f"Human Input: {human_response_from_file.get('response', 'No specific response.')}")

                    self._log_swarm_activity(
                        "HUMAN_INPUT_RECEIVED",
                        "CatalystVectorAlpha",
                        f"Level 2: Human input received for request from {requester}.",
                        {"message": message, "response": human_response_from_file.get('response', 'N/A'), "cycle_id": request_cycle_id, "request_id": request_id},
                        level='info'
                    )

                    # CRITICAL FIX: Pass the request_id and original directive info to the agent
                    # Create a comprehensive response object for the agent
                    response_for_agent = human_response_from_file.copy()
                    response_for_agent['request_id'] = request_id
                    response_for_agent['original_directive'] = directive # Pass the full original directive for context
                    # Add problem_id if it was part of the original directive (good for agent context)
                    response_for_agent['problem_id'] = directive.get('problem_id')

                    # Delete the file after it has been read
                    os.remove(response_file_path)
                    self._log_swarm_activity(
                        "HUMAN_RESPONSE_FILE_CLEANUP",
                        "CatalystVectorAlpha",
                        f"Cleaned up human response file: {response_file_path}",
                        {"file": response_file_path},
                        level='debug'
                    )

                    # Remove the request from the pending_human_interventions dictionary
                    if request_id in self.pending_human_interventions:
                        del self.pending_human_interventions[request_id]
                        self._log_swarm_activity(
                            "HUMAN_INTERVENTION_REQUEST_CLEARED",
                            "CatalystVectorAlpha",
                            f"Human intervention request {request_id} cleared from pending list after processing response.",
                            {"request_id": request_id},
                            level='info'
                        )

                    if self.is_system_paused():
                        self.unpause_system(reason=f"Human input received for {request_id}.")

                    # Inform the requesting agent about the human response
                    if requester in self.agent_instances:
                        req_agent = self.agent_instances[requester]
                        if hasattr(req_agent, 'human_input_received') and callable(getattr(req_agent, 'human_input_received')):
                            # Pass the enriched response_for_agent
                            req_agent.human_input_received(response_for_agent)
                        else:
                            print(f"  Warning: Requesting agent '{requester}' does not have a 'human_input_received' method.")
                    else:
                        print(f"  Warning: Requesting agent '{requester}' not found to inform about human input.")

                    # Important: If a response was received and processed, this directive should NOT be re-injected.
                    # It has been successfully handled.

                except json.JSONDecodeError as e:
                    print(f"ERROR: Malformed human response file: {response_file_path}")
                    self._log_swarm_activity("HUMAN_INPUT_ERROR", "CatalystVectorAlpha",
                                            f"Level 2: Malformed human response file '{os.path.basename(response_file_path)}'. Error: {e}",
                                            {"file": response_file_path, "error": str(e)},
                                            level='error')
                    # If malformed, still escalate to next level
                    next_directive = directive.copy()
                    next_directive["human_request_counter"] = 2 # Increment to Level 3
                    self.inject_directives([next_directive])
                    print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 3) due to malformed response.")
                except Exception as e:
                    print(f"ERROR: Failed to process human response file: {response_file_path} - {e}")
                    self._log_swarm_activity("HUMAN_INPUT_ERROR", "CatalystVectorAlpha",
                                            f"Level 2: Failed to process human response file '{os.path.basename(response_file_path)}'. Error: {e}",
                                            {"file": response_file_path, "error": str(e)},
                                            level='error')
                    # If other processing error, also escalate
                    next_directive = directive.copy()
                    next_directive["human_request_counter"] = 2 # Increment to Level 3
                    self.inject_directives([next_directive])
                    print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 3) due to processing error.")
            else: # No human response file found, escalate to Level 3
                print(f"\n!!! HUMAN INTERVENTION PENDING (Level 2, Urgency: {urgency.upper()}) !!!")
                print(f"!!! From: {requester} (Target: {target_agent}) !!!")
                print(f"!!! Message: {message} !!!")
                # This line needs the fix for PERSISTENCE_DIR -> self.persistence_dir (and use response_file_path)
                print(f"!!! No human response in '{response_file_path}'. Escalating. !!!") # Used response_file_path here

                self._log_swarm_activity("HUMAN_INPUT_PENDING_LEVEL2", "CatalystVectorAlpha",
                                        f"Level 2: No human input received for {request_id}. Escalating to Level 3. Expected file: {response_file_path}", # Added expected file path to log
                                        {"message": message, "cycle_id": request_cycle_id, "request_id": request_id, "expected_file": response_file_path},
                                        level='warning')
                next_directive = directive.copy()
                next_directive["human_request_counter"] = 2 # Increment to Level 3
                self.inject_directives([next_directive])
                print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 3) for next cycle check.")

        elif human_request_counter >= 2: # Level 3: Full System Pause (Critical)
            print(f"\n!!! CRITICAL: HUMAN INTERVENTION FAILED (Level 3, Urgency: {urgency.upper()}) !!!")
            print(f"!!! From: {requester} (Target: {target_agent}) !!!")
            print(f"!!! Message: {message} !!!")
            print("!!! No human response. Initiating full system pause. !!!")
            
            # Removed the direct file write for system_pause.flag here.
            # It's now handled by self.pause_system() below.
            # Removed the redundant print statement about system_pause.flag created here.

            pause_reason = f"Human intervention failed for request ID {request_id} from {requester}: '{message}'."
            self.pause_system(reason=pause_reason) # This method will log the SYSTEM_PAUSED_CRITICAL event

            # This log entry is useful for marking the *failure* of the human input process at Level 3,
            # distinct from the system pause itself (which self.pause_system logs).
            self._log_swarm_activity("HUMAN_INPUT_FAILED_LEVEL3_CRITICAL", "CatalystVectorAlpha",
                                    f"Level 3 Critical: Human input failed to materialize for {request_id}. System initiating full pause.",
                                    {"message": message, "urgency": urgency, "requester": requester, "cycle_id": request_cycle_id, "request_id": request_id},
                                    level='error')

            # Remove the request from the pending_human_interventions dictionary
            if request_id in self.pending_human_interventions:
                del self.pending_human_interventions[request_id]
                self._log_swarm_activity(
                    "HUMAN_INTERVENTION_REQUEST_FINAL_FAILURE_CLEARED",
                    "CatalystVectorAlpha",
                    f"Human intervention request {request_id} removed from pending list after critical failure and system pause.",
                    {"request_id": request_id},
                    level='info'
                )

    def inject_directives(self, new_directives_list: list):
        """
        Allows other components (e.g., Planner agents) to inject new directives
        into the CatalystVectorAlpha's processing queue, with basic validation.
        """
        if not isinstance(new_directives_list, list):
            new_directives_list = [new_directives_list]

        valid_directives = []
        total_attempted = len(new_directives_list)

        for directive in new_directives_list:
            if not isinstance(directive, dict) or 'type' not in directive:
                print(f"  [CatalystVectorAlpha] Warning: Invalid injected directive format, skipping: {directive}")
                self._log_swarm_activity("INJECTED_DIRECTIVE_INVALID_FORMAT", "CatalystVectorAlpha",
                                        "Skipped invalid injected directive due to malformed format.",
                                        {"directive": str(directive)[:200]}, level='warning')
                continue

            directive_type = directive['type']
            is_valid = True
            validation_reason = ""
            
            # Check if directive type is in our schema (this is basic validation)
            if directive_type not in self._directive_handlers:
                is_valid = False
                validation_reason = f"Unknown directive type: {directive_type}."
            elif directive_type in ['AGENT_PERFORM_TASK', 'SPAWN_AGENT_INSTANCE', 'BROADCAST_COMMAND', 'INITIATE_PLANNING_CYCLE', 'CATALYZE_TRANSFORMATION']:
                agent_field = None
                if directive_type == 'AGENT_PERFORM_TASK': agent_field = 'agent_name'
                elif directive_type == 'SPAWN_AGENT_INSTANCE': agent_field = 'instance_name'
                elif directive_type == 'BROADCAST_COMMAND': agent_field = 'target_agent'
                elif directive_type == 'INITIATE_PLANNING_CYCLE': agent_field = 'planner_agent_name'
                elif directive_type == 'CATALYZE_TRANSFORMATION': agent_field = 'target_agent_instance'

                if agent_field and agent_field in directive:
                    if directive[agent_field] not in self.agent_instances:
                        is_valid = False
                        validation_reason = f"Target agent '{directive[agent_field]}' not found for directive type '{directive_type}'."
            elif directive_type == 'INJECT_EVENT' and ('payload' not in directive or 'change_factor' not in directive['payload'] or 'urgency' not in directive['payload'] or 'direction' not in directive['payload']):
                is_valid = False
                validation_reason = "INJECT_EVENT directive missing 'payload' or required payload fields (change_factor, urgency, direction)."
            elif directive_type == 'REQUEST_HUMAN_INPUT' and 'message' not in directive:
                is_valid = False
                validation_reason = "REQUEST_HUMAN_INPUT directive missing 'message' field."

            if not is_valid:
                print(f"  [CatalystVectorAlpha] Warning: Invalid injected directive, skipping. Reason: {validation_reason}, Directive: {directive}")
                self._log_swarm_activity("INJECTED_DIRECTIVE_INVALID_CONTENT", "CatalystVectorAlpha",
                                        f"Skipped invalid injected directive. Reason: {validation_reason}.",
                                        {"directive": str(directive)[:200], "reason": validation_reason}, level='warning')
                continue

            # Assign a cycle_id to injected directives if they don't have one
            if 'cycle_id' not in directive:
                directive['cycle_id'] = self.current_action_cycle_id
        
            # Assign a unique event_id if it's an INJECT_EVENT directive and doesn't have one
            if directive_type == "INJECT_EVENT" and 'event_id' not in directive:
                directive['event_id'] = f"EVT-{int(time.time() * 1000)}_{random.randint(0, 999)}"

            # --- CORRECTED LOGIC: Append the directive only ONCE if it's valid ---
            valid_directives.append(directive)

            # Log each *valid* directive being added to the queue
            log_event_type = f"INJECTED_DIRECTIVE_{directive_type}"
            self._log_swarm_activity(log_event_type, "CatalystVectorAlpha",
                                    f"Injected directive of type '{directive_type}'.",
                                    {"directive_type": directive_type, "details": str(directive)[:200]}, level='info')
            # --- END OF CORRECTED LOGIC ---

        self.dynamic_directive_queue.extend(valid_directives)
        self._log_swarm_activity("DIRECTIVES_BATCH_INJECTED", "CatalystVectorAlpha",
                                f"Injected {len(valid_directives)} new directives into queue.",
                                {"directives_count": len(valid_directives), "first_directive_type": valid_directives[0].get('type') if valid_directives else 'N/A', "total_attempted": total_attempted}, level='info')
        print(f"[CatalystVectorAlpha] Injected {len(valid_directives)} new directives dynamically (out of {total_attempted} attempted).")

    def _load_system_state(self):
        """
        Loads the overall system state from persistence, including agents and swarms.
        """
        self._log_swarm_activity("SYSTEM_STARTUP", "CatalystVectorAlpha",
                                 f"Attempting to load previous system state from '{self.persistence_dir}'.",
                                 level='info')
        print(f"\n--- Loading previous system state from '{self.persistence_dir}' ---")

        temp_agent_states_to_instantiate = {}

        # --- Load Agent States ---
        # Use self.persistence_dir for os.listdir
        for filename in os.listdir(self.persistence_dir):
            if filename.startswith('agent_state_') and filename.endswith('.json'):
                agent_name = filename.replace('agent_state_', '').replace('.json', '')
                file_path = os.path.join(self.persistence_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        loaded_state = json.load(f)

                    if 'eidos_spec' in loaded_state:
                        eidos_name_from_state = loaded_state['eidos_spec'].get('eidos_name')
                        if eidos_name_from_state:
                            # Register EIDOS if not already known (important for re-instantiation)
                            self.eidos_registry[eidos_name_from_state] = loaded_state['eidos_spec']
                        temp_agent_states_to_instantiate[agent_name] = loaded_state
                    else:
                        print(f"Error loading agent state from {filename}: Missing key 'eidos_spec'. Skipping this old state file.")
                        self.external_log_sink.warning(f"Skipping agent state '{filename}': Missing 'eidos_spec'.")
                except json.JSONDecodeError:
                    print(f"Error loading agent state from {filename}: Invalid JSON format. Skipping.")
                    self.external_log_sink.error(f"Skipping agent state '{filename}': Invalid JSON format.")
                except Exception as e:
                    print(f"Unexpected error loading agent state from {filename}: {e}. Skipping.")
                    self.external_log_sink.error(f"Unexpected error loading agent state '{filename}': {e}.")

        for agent_name, loaded_state in temp_agent_states_to_instantiate.items():
            eidos_name = loaded_state['eidos_spec'].get('eidos_name')
            if eidos_name and eidos_name in self.eidos_registry:
                agent_class_map = {
                    'ProtoAgent_Observer': ProtoAgent_Observer,
                    'ProtoAgent_Collector': ProtoAgent_Collector,
                    'ProtoAgent_Optimizer': ProtoAgent_Optimizer,
                    'ProtoAgent_Planner': ProtoAgent_Planner
                }
                agent_class = agent_class_map.get(eidos_name)
                if agent_class:
                    try:
                        # CRITICAL FIX: Pass all required arguments to agent_class.__init__
                        # These arguments should now be correctly defined as self.xxx_full_path in CatalystVectorAlpha.__init__
                        self.agent_instances[agent_name] = agent_class(
                            name=agent_name,
                            eidos_spec=loaded_state['eidos_spec'],
                            message_bus=self.message_bus,
                            tool_registry=self.tool_registry,
                            event_monitor=self.event_monitor,
                            external_log_sink=self.external_log_sink,
                            chroma_db_path=self.chroma_db_full_path, # <--- Pass full ChromaDB path
                            persistence_dir=self.persistence_dir,     # <--- Pass base persistence dir
                            paused_agents_file_path=self.paused_agents_file_full_path_full_path, # <--- Pass full paused agents file path
                            sovereign_gradient=loaded_state.get('sovereign_gradient'), # Pass loaded gradient config/state
                            loaded_state=loaded_state # Pass the entire loaded state for agent's _load_or_initialize_state
                        )
                        self.external_log_sink.info(f"Agent '{agent_name}' re-instantiated from loaded state.")
                    except Exception as e:
                        print(f"Error re-instantiating agent '{agent_name}' from loaded state: {e}. Skipping.")
                        self.external_log_sink.error(f"Error re-instantiating agent '{agent_name}': {e}.")
                else:
                    print(f"Warning: Unknown agent EIDOS '{eidos_name}' found in state file for '{agent_name}'. Skipping instantiation.")
                    self.external_log_sink.warning(f"Unknown EIDOS '{eidos_name}' for agent '{agent_name}'. Skipping.")
            else:
                print(f"Warning: EIDOS for agent '{agent_name}' (type '{eidos_name}') not found in registry. Skipping instantiation.")
                self.external_log_sink.warning(f"EIDOS for agent '{agent_name}' not found in registry. Skipping.")

        # --- Load Swarm States ---
        # Iterate through potential swarm state files (assuming swarm_state_<name>.json format)
        swarm_files_found = False
        for filename in os.listdir(self.persistence_dir):
            if filename.startswith('swarm_state_') and filename.endswith('.json'):
                swarm_files_found = True
                current_swarm_name = filename.replace('swarm_state_', '').replace('.json', '')
                file_path = os.path.join(self.persistence_dir, filename) # Full path to this specific swarm file
                try:
                    with open(file_path, 'r') as f:
                        swarm_state = json.load(f)
                    
                    # CRITICAL FIX: Pass the full path for this specific swarm's state file to SwarmProtocol
                    swarm_protocol_class = SwarmProtocol
                    self.swarm_protocols[current_swarm_name] = swarm_protocol_class(
                        swarm_name=swarm_state['name'],
                        initial_goal=swarm_state['goal'],
                        initial_members=swarm_state['members'],
                        consensus_mechanism=swarm_state['consensus_mechanism'],
                        description=swarm_state.get('description', 'A collective intelligence.'),
                        loaded_state=swarm_state, # Pass the full loaded state
                        catalyst_vector_ref=self, # Pass reference to orchestrator
                        swarm_state_file_path=file_path # <--- Pass the full path to its own state file
                    )
                    self.external_log_sink.info(f"Successfully reloaded swarm '{current_swarm_name}' state from {file_path}.")
                except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                    print(f"Error loading swarm state from {file_path}: {e}. Skipping.")
                    self.external_log_sink.error(f"Error loading swarm state from {file_path}: {e}.")
                except Exception as e:
                    print(f"Unexpected error loading swarm state from {file_path}: {e}. Skipping.")
                    self.external_log_sink.error(f"Unexpected error loading swarm state from {file_path}: {e}.")
        
        if not swarm_files_found:
            print("  No previous swarm states found.")
            self._log_swarm_activity("SYSTEM_STATE_INFO", "CatalystVectorAlpha",
                                     "No previous swarm states found, starting fresh.", level='info')
        else:
            print(f"  Successfully loaded {len(self.swarm_protocols)} swarm states.")


        print("--- Finished loading previous system state ---\n")

    def _save_system_state(self):
        print("\n--- Saving current system state ---")
        
        # Helper to convert non-serializable types for JSON (defined locally for clarity)
        def convert_to_serializable_recursive(obj):
            if isinstance(obj, deque):
                return list(obj)
            if isinstance(obj, defaultdict):
                return dict(obj)
            if isinstance(obj, dict):
                # Recursively apply to dictionary values
                return {k: convert_to_serializable_recursive(v) for k, v in obj.items()}
            if isinstance(obj, list):
                # Recursively apply to list elements
                return [convert_to_serializable_recursive(elem) for elem in obj]
            # Handle specific non-serializable object types (add more as needed if issues arise)
            # This is a general filter for common problematic object references
            if isinstance(obj, (logging.Logger, type(self.message_bus), type(self.tool_registry), type(self.event_monitor), type(self.scenario))):
                return None # Or a string representation like obj.__class__.__name__
            if callable(obj):
                return None # Functions/methods are not serializable

            return obj

        # --- Collect all state data into a single dictionary ---
        system_state_data = {
            'current_action_cycle_id': self.current_action_cycle_id,
            'eidos_registry': convert_to_serializable_recursive(self.eidos_registry),
            'dynamic_directive_queue': list(self.dynamic_directive_queue), # Explicitly converted
            'scenario_state': convert_to_serializable_recursive(self.active_scenario.get_scenario_state()) if self.active_scenario else None,
            'pending_human_interventions': convert_to_serializable_recursive(self.pending_human_interventions),
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            '_no_pattern_reports_count': convert_to_serializable_recursive(self._no_pattern_reports_count),
            # 'active_scenario_ref' saves a key piece of identity for the active scenario
            'active_scenario_ref': self.active_scenario.phase if self.active_scenario else None,
            'SWARM_RESET_THRESHOLD': self.SWARM_RESET_THRESHOLD,
            'scenario_state': convert_to_serializable_recursive(self.active_scenario.get_scenario_state()) if self.active_scenario else None,
            'NO_PATTERN_AGENT_THRESHOLD': self.NO_PATTERN_AGENT_THRESHOLD,
            'event_monitor_state': convert_to_serializable_recursive(self.event_monitor.get_state()),
            'swarm_protocols': {name: convert_to_serializable_recursive(swarm.get_state()) for name, swarm in self.swarm_protocols.items()} if self.swarm_protocols else {},
            'agent_instances': {} # Populated in the loop below
            
        }

        # Handle agent instances and ensure their states are serializable
        for agent_name, agent_instance in self.agent_instances.items():
            if hasattr(agent_instance, 'get_state'): # Prefer agent's own get_state if available
                agent_state = convert_to_serializable_recursive(agent_instance.get_state()) # Apply recursive conversion
            else: # Fallback to __dict__ but apply recursive conversion and filter
                agent_state = agent_instance.__dict__.copy()
                for key, value in agent_state.items():
                    agent_state[key] = convert_to_serializable_recursive(value)
                # Filter out objects that are inherently non-serializable or problematic references
                agent_state = {
                    k: v for k, v in agent_state.items() 
                    if not callable(v) and not isinstance(v, (logging.Logger, MessageBus, ToolRegistry, EventMonitor, type(self), type(agent_instance.ollama_inference_model)))
                }
            system_state_data['agent_instances'][agent_name] = agent_state # Store in system_state_data
        
        # Save to a main system state file
        try:
            # CORRECTED: Use the correct attribute name
            os.makedirs(os.path.dirname(self.swarm_state_file_full_path), exist_ok=True)
            with open(self.swarm_state_file_full_path, 'w') as f:
                json.dump(system_state_data, f, indent=2)
            print(f"  Overall swarm state saved to {self.swarm_state_file_full_path}.")
            self._log_swarm_activity("SYSTEM_SAVE_SUCCESS", "CatalystVectorAlpha", f"Overall swarm state saved to {self.swarm_state_file_full_path}.", {"file": self.swarm_state_file_full_path}, level='info')
        except Exception as e:
            print(f"ERROR: Could not save overall swarm state: {e}")
            self._log_swarm_activity("SYSTEM_SAVE_ERROR", "CatalystVectorAlpha", f"Error saving swarm state: {e}", {"error": str(e)}, level='error')

        print("--- System state saved ---")
    
    def is_swarm_stagnant(self) -> bool:
        """
        Checks if a significant portion of the swarm is considered stagnant.
        This is the central logic for the Planner to decide when to intervene.
        """
        if not self.agent_instances:
            return False # No agents, no stagnation

        # An agent is considered stagnant if its loop counter is high
        stagnant_threshold = 3 # e.g., stagnant if counter > 3
        stagnant_agents = 0
        
        for agent in self.agent_instances.values():
            if agent.intent_loop_count >= stagnant_threshold:
                stagnant_agents += 1
                
        # If more than 60% of agents are stagnant, the swarm is stagnant
        stagnation_ratio = stagnant_agents / len(self.agent_instances)
        
        if stagnation_ratio >= 0.6:
            self._log_swarm_activity(
                "SWARM_STAGNATION_DETECTED", "CatalystVectorAlpha",
                f"Swarm stagnation confirmed: {stagnant_agents}/{len(self.agent_instances)} agents are stagnant.",
                {"ratio": stagnation_ratio}
            )
            return True
            
        return False

    def _process_intent_overrides(self):
        """Scans for and processes intent override files from the console."""
        override_files = [f for f in os.listdir(self.persistence_dir) if f.startswith(self.intent_override_prefix) and f.endswith('.json')]
        
        if override_files:
            print("\n--- Processing Intent Overrides ---")
        
        for filename in override_files:
            filepath = os.path.join(self.persistence_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    override_data = json.load(f)
                
                target_name = override_data.get('target')
                new_intent = override_data.get('new_intent')

                if target_name and new_intent:
                    if target_name in self.agent_instances:
                        target_entity = self.agent_instances[target_name]
                        print(f"  Override: Applying new intent '{new_intent}' to Agent '{target_name}'.")
                        target_entity.update_intent(new_intent)
                        self._log_swarm_activity("INTENT_OVERRIDDEN", "CatalystVectorAlpha",
                                                 f"Agent '{target_name}' intent overridden by console.",
                                                 {"agent": target_name, "new_intent": new_intent})
                    elif target_name in self.swarm_protocols:
                        target_entity = self.swarm_protocols[target_name]
                        print(f"  Override: Applying new goal '{new_intent}' to Swarm '{target_name}'.")
                        target_entity.set_goal(new_intent) # Swarms have 'goal' instead of 'intent'
                        self._log_swarm_activity("SWARM_GOAL_OVERRIDDEN", "CatalystVectorAlpha",
                                                 f"Swarm '{target_name}' goal overridden by console.",
                                                 {"swarm": target_name, "new_goal": new_intent})
                    else:
                        print(f"  Override Error: Target '{target_name}' not found for intent override.")
                        self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                                 f"Intent override target '{target_name}' not found.",
                                                 {"target": target_name, "new_intent": new_intent})
                else:
                    print(f"  Override Error: Malformed override data in '{filename}'.")
                    self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                             f"Malformed intent override data in '{filename}'.",
                                             {"filepath": filepath, "data": override_data})

                # Mark the override file as processed (rename it)
                mark_override_processed(filepath)

            except Exception as e:
                print(f"  Error processing override file '{filename}': {e}")
                self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                         f"Error processing intent override file '{filename}': {e}",
                                         {"filepath": filepath, "error": str(e)})    
    
    def perform_causal_failure_analysis(self, num_recent_log_entries=30) -> list[dict]:
        """
        Performs a conceptual causal analysis based on recent failure logs from swarm_activity.jsonl.
        For this prototype, it uses rule-based inference on log event types.
        """
        print(f"\n--- Initiating Causal Failure Analysis (Scanning last {num_recent_log_entries} log entries) ---")
        self._log_swarm_activity("CAUSAL_ANALYSIS_INITIATED", "CatalystVectorAlpha",
                                f"Starting causal analysis on last {num_recent_log_entries} log entries.")

        analysis_results = []

        # CRITICAL FIX: The function call now uses the public alias 'get_recent_log_entries'
        recent_log_entries = get_recent_log_entries(self.swarm_activity_log_full_path, num_recent_log_entries)

        # Filter for events that typically indicate problems or lead to failures
        critical_events_to_analyze = [
            entry for entry in recent_log_entries
            if entry.get('event_type') in [
                "DIRECTIVE_ERROR",
                "AGENT_ADAPTATION_HALTED", "AGENT_ADAPTATION_HALTED_PREDICTIVE", "RECURSION_LIMIT_EXCEEDED",
                "PLANNING_CYCLE_FAILED", "PLANNING_FALLBACK",
                "MEMORY_COMPRESSION_FAILED", "SHARED_MEMORY_ADD_FAILED",
                "HUMAN_INPUT_PENDING_LEVEL2", "HUMAN_INPUT_FAILED_LEVEL3_CRITICAL",
                "LLM_CALL_FAILED"
            ]
        ]

        if not critical_events_to_analyze:
            print("  No recent critical failure events found for analysis in the scanned window.")
            self._log_swarm_activity("CAUSAL_ANALYSIS_COMPLETE", "CatalystVectorAlpha",
                                    "Causal analysis finished: No critical events found.", {"analysis_summary": {}})
            return []

        # Simple rule-based causal inference (for prototype)
        failure_patterns_summary = {}

        for event in critical_events_to_analyze:
            event_type = event.get('event_type')
            source = event.get('source')
            description = event.get('description', '')
            details = event.get('details', {})
            timestamp = event.get('timestamp')

            analysis = {
                "timestamp": timestamp,
                "event_type": event_type,
                "source": source,
                "description": description,
                "inferred_cause": "Uncategorized/Requiresdeeper_LLM_analysis",
                "suggested_next_steps": []
            }

            if event_type in ["RECURSION_LIMIT_EXCEEDED", "AGENT_ADAPTATION_HALTED", "AGENT_ADAPTATION_HALTED_PREDICTIVE"]:
                analysis["inferred_cause"] = "Agent stuck in adaptive loop or persistent task failure."
                analysis["suggested_next_steps"].append(f"Review {source}'s recent TaskOutcome memories for specific persistent failures.")
                analysis["suggested_next_steps"].append(f"Consider manual intent override or re-initialization for agent {source}.")
                if source in self.agent_instances and hasattr(self.agent_instances[source], 'planning_failure_count'):
                    analysis["suggested_next_steps"].append(f"Check {source}'s planning_failure_count for persistent planning issues.")

            elif event_type in ["PLANNING_CYCLE_FAILED", "PLANNING_FALLBACK"]:
                analysis["inferred_cause"] = "Planner failed to decompose a high-level goal or triggered fallback."
                analysis["suggested_next_steps"].append("Examine Planner's last high-level goal for complexity or ambiguity.")
                analysis["suggested_next_steps"].append("Review LLM prompts/outputs for Planner if LLM planning was attempted.")
                analysis["suggested_next_steps"].append("Query shared memory for similar problems and their solutions.")

            elif event_type == "HUMAN_INPUT_FAILED_LEVEL3_CRITICAL":
                analysis["inferred_cause"] = "Critical human intervention request went unaddressed, leading to system pause."
                analysis["suggested_next_steps"].append("Immediate human attention required to unpause system or provide guidance.")
                analysis["suggested_next_steps"].append("Investigate why human response was not provided within the timeout.")

            elif event_type == "LLM_CALL_FAILED":
                analysis["inferred_cause"] = "Local LLM inference (summary or embedding) failed."
                analysis["suggested_next_steps"].append("Verify Ollama server status and required models (llama3, nomic-embed-text) are running.")
                analysis["suggested_next_steps"].append("Check LLM-related configuration and prompts for syntax errors.")

            elif event_type == "DIRECTIVE_ERROR":
                error_details = details.get('error', 'N/A')
                directive_type_errored = details.get('directive', {}).get('type', 'N/A')
                analysis["inferred_cause"] = f"System failed to process directive '{directive_type_errored}'. Error: {error_details}."
                analysis["suggested_next_steps"].append(f"Validate the syntax and content of '{directive_type_errored}' directive against ISL schema.")
                analysis["suggested_next_steps"].append(f"Trace the origin of the directive (manifest, planner, human input).")

            elif event_type in ["MEMORY_COMPRESSION_FAILED", "SHARED_MEMORY_ADD_FAILED"]:
                analysis["inferred_cause"] = "Memory system encountered an issue (compression or shared DB write)."
                analysis["suggested_next_steps"].append(f"Check agent's local memory log ({source}).")
                if "SHARED_MEMORY" in event_type:
                    analysis["suggested_next_steps"].append("Verify ChromaDB instance is running and accessible.")
                    analysis["suggested_next_steps"].append("Inspect embeddings generation process for errors.")

            analysis_results.append(analysis)

            cause_category = analysis["inferred_cause"].split(' ')[0]
            failure_patterns_summary[cause_category] = failure_patterns_summary.get(cause_category, 0) + 1

        print(f"\n  Identified Failure Patterns Summary:")
        for pattern, count in failure_patterns_summary.items():
            print(f"    - '{pattern}': {count} occurrences.")

        self._log_swarm_activity("CAUSAL_ANALYSIS_COMPLETE", "CatalystVectorAlpha",
                                f"Causal analysis finished. Identified {len(failure_patterns_summary)} patterns.",
                                {"analysis_patterns": failure_patterns_summary, "detailed_results_count": len(analysis_results)})
        print("--- Causal Failure Analysis Complete ---\n")
        return analysis_results

    def _get_recent_log_entries(self, log_file, num_entries):
        """Helper to get recent log entries."""
        entries = []
        try:
            with open(log_file, 'r') as f:
                # Read all lines and take the last 'num_entries'
                lines = f.readlines()
                for line in lines[-num_entries:]:
                    entries.append(json.loads(line))
        except FileNotFoundError:
            print(f"Log file not found: {log_file}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from log file {log_file}: {e}")
        return entries

    def pause_system_for_human_input(self, reason: str, urgency: str = "critical", source_agent: str = "System"):
        """
        Injects a REQUEST_HUMAN_INPUT directive into the system,
        triggering the multi-phase human escalation protocol.
        This method is called by scenarios or other high-level system components.
        """
        print(f"\n!!! ORCHESTRATOR INITIATING HUMAN INTERVENTION (Reason: {reason}, Urgency: {urgency.upper()}) !!!")
        
        # Inject a REQUEST_HUMAN_INPUT directive
        human_input_directive = {
            "type": "REQUEST_HUMAN_INPUT",
            "message": reason,
            "urgency": urgency,
            "target_agent": "System", # Direct to 'System' for a global human input request
            "requester_agent": source_agent, # Who is asking for it (e.g., CyberAttackScenario)
            "cycle_id": self.current_action_cycle_id,
            "human_request_counter": 0 # Start the escalation from Level 0
        }
        
        self.inject_directives([human_input_directive]) # Inject as a list
        self._log_swarm_activity("ORCHESTRATOR_REQUESTED_HUMAN_INPUT", "CatalystVectorAlpha",
                                 f"Orchestrator requested human input: {reason}",
                                 {"reason": reason, "urgency": urgency, "source": source_agent})

    def execute_single_directive(self, directive: dict):
        """
        Executes a single ISL directive. This method is called by execute_manifest
        and also by run_cognitive_loop for injected directives.
        """
        directive_type = directive['type']

        # Initialize variables at the top of the try block to ensure they are defined
        # for logging or other uses, even if specific branches don't define them.
        # These are common variables that might be used across different directive types' logging.
        target_agents_list = []
        event_type_for_log = directive.get('event_type', 'N/A') # For INJECT_EVENT logging
        payload_for_log = directive.get('payload', {}) # For INJECT_EVENT logging
        request_id_for_log = directive.get('request_id', 'N/A') # For REQUEST_HUMAN_INPUT logging
        requester_for_log = directive.get('requester_agent', 'N/A') # For REQUEST_HUMAN_INPUT logging
        message_for_log = directive.get('message', 'N/A') # For REQUEST_HUMAN_INPUT logging


        try:
            if directive_type == 'ASSERT_AGENT_EIDOS':
                eidos_name = directive['eidos_name']
                eidos_spec = directive['eidos_spec']
                eidos_spec['eidos_name'] = eidos_name
                if eidos_name not in self.eidos_registry:
                    self.eidos_registry[eidos_name] = eidos_spec
                    self._log_swarm_activity("EIDOS_ASSERTED", "CatalystVectorAlpha",
                        f"Defined EIDOS for '{eidos_name}'.", {"eidos_name": eidos_name})
                    print(f"  ASSERT_AGENT_EIDOS: Defined EIDOS for '{eidos_name}'.")
                else:
                    print(f"  ASSERT_AGENT_EIDOS: EIDOS '{eidos_name}' already exists. Reusing.")
                    self._log_swarm_activity("EIDOS_REUSED", "CatalystVectorAlpha",
                        f"EIDOS '{eidos_name}' already exists, reusing.", {"eidos_name": eidos_name})


            elif directive_type == 'ESTABLISH_SWARM_EIDOS':
                swarm_name = directive['swarm_name']
                initial_goal = directive.get('initial_goal', 'No specified goal')
                initial_members = directive.get('initial_members', [])
                consensus_mechanism = directive.get('consensus_mechanism', 'SimpleMajorityVote')
                description = directive.get('description', 'A collective intelligence.')

                if swarm_name in self.swarm_protocols:
                    print(f"  ESTABLISH_SWARM_EIDOS: Swarm '{swarm_name}' already active. Reusing.")
                    swarm = self.swarm_protocols[swarm_name]
                    self._log_swarm_activity("SWARM_REUSED", "CatalystVectorAlpha",
                        f"Swarm '{swarm_name}' already active, reusing.", {"swarm_name": swarm_name})
                else:
                    swarm = SwarmProtocol(swarm_name, initial_goal, initial_members, consensus_mechanism, description, catalyst_vector_ref=self)
                    self.swarm_protocols[swarm_name] = swarm
                    self._log_swarm_activity("SWARM_ESTABLISHED", "CatalystVectorAlpha",
                        f"Established Swarm '{swarm_name}' with goal: '{initial_goal}'.",
                        {"swarm": swarm_name, "initial_goal": initial_goal})
                swarm.coordinate_task(directive.get('task_description', 'Initial swarm formation and goal orientation'))

            elif directive_type == 'SPAWN_AGENT_INSTANCE':
                eidos_name = directive['eidos_name']
                instance_name = directive['instance_name']
                initial_task = directive.get('initial_task', 'Run diagnostic checks') # Capture initial_task here

                if eidos_name not in self.eidos_registry: # Changed from eidos_definitions based on your code
                    raise ValueError(f"EIDOS '{eidos_name}' not asserted yet. Define it first using ASSERT_AGENT_EIDOS.")

                eidos_spec = self.eidos_registry[eidos_name] # Changed from eidos_definitions

                if instance_name in self.agent_instances:
                    agent = self.agent_instances[instance_name]
                    print(f"  SPAWN_AGENT_INSTANCE: Agent '{instance_name}' already exists. Reusing existing instance.")
                    self._log_swarm_activity("AGENT_REUSED", "CatalystVectorAlpha",
                                            f"Agent '{instance_name}' already existed, reusing.",
                                            {"agent_name": instance_name, "eidos_name": eidos_name})
                else:
                    print(f"  SPAWN_AGENT_INSTANCE: Spawning new agent '{instance_name}'.")
                    # --- FIXED: Use the helper method here ---
                    # Ensure _create_agent_instance exists and correctly passes external_log_sink
                    agent = self._create_agent_instance(
                        name=instance_name,
                        eidos_spec=eidos_spec,
                        message_bus=self.message_bus, # Pass message_bus
                        tool_registry=self.tool_registry,
                        event_monitor=self.event_monitor,
                        external_log_sink=self.external_log_sink # <--- Pass external_log_sink to agent
                    )
                    # --- END OF FIXED BLOCK ---
                    self.agent_instances[instance_name] = agent
                    self._log_swarm_activity("AGENT_SPAWNED", "CatalystVectorAlpha",
                                            f"New agent '{instance_name}' spawned.",
                                            {"agent_name": instance_name, "eidos_name": eidos_name, "context": eidos_spec.get('location', 'Unknown')})

                # This part should be after agent is created/reused:
                # Ensure agent.perform_task exists and takes cycle_id
                outcome = agent.perform_task(initial_task, cycle_id=self.current_action_cycle_id)
                # Ensure agent.memetic_kernel.reflect() exists
                print(f"  [MemeticKernel] {agent.name} reflects: '{agent.memetic_kernel.reflect()}'")

            elif directive_type == 'ADD_AGENT_TO_SWARM':
                swarm_name = directive['swarm_name']
                agent_name_to_add = directive['agent_name']

                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not established. Cannot add agent '{agent_name_to_add}'.")
                if agent_name_to_add not in self.agent_instances:
                    raise ValueError(f"Agent instance '{agent_name_to_add}' not found. Cannot add to swarm.")

                self.swarm_protocols[swarm_name].add_member(agent_name_to_add)
                self.agent_instances[agent_name_to_add].join_swarm(swarm_name)
                self._log_swarm_activity("AGENT_ADDED_TO_SWARM", "CatalystVectorAlpha",
                                         f"Agent '{agent_name_to_add}' added to swarm '{swarm_name}'.",
                                         {"agent": agent_name_to_add, "swarm": swarm_name})

            elif directive_type == 'ASSERT_GRADIENT_TRAJECTORY':
                target_type = directive['target_type']
                target_ref = directive['target_ref']

                target_obj = None
                if target_type == 'Agent' and target_ref in self.agent_instances:
                    target_obj = self.agent_instances[target_ref]
                elif target_type == 'Swarm' and target_ref in self.swarm_protocols:
                    target_obj = self.swarm_protocols[target_ref]
                else:
                    raise ValueError(f"Target entity '{target_ref}' (type {target_type}) not found for ASSERT_GRADIENT_TRAJECTORY.")

                # Convert directive properties to a config dict for SovereignGradient
                gradient_config = {
                    'autonomy_vector': directive.get('autonomy_vector', 'General self-governance'),
                    'ethical_constraints': directive.get('ethical_constraints', []),
                    'self_correction_protocol': directive.get('self_correction_protocol', 'BasicCorrection'),
                    'override_threshold': directive.get('override_threshold', 0.0)
                }
                new_gradient = SovereignGradient(target_ref, gradient_config)
                target_obj.set_sovereign_gradient(new_gradient)

                self._log_swarm_activity("GRADIENT_ASSERTED", "CatalystVectorAlpha",
                                         f"Sovereign Gradient asserted for {target_type} '{target_ref}'.",
                                         {"entity": target_ref, "type": target_type, "autonomy_vector": new_gradient.autonomy_vector})
                print(f"  ASSERT_GRADIENT_TRAJECTORY: Set gradient for {target_type} '{target_ref}' to '{new_gradient.autonomy_vector}'.")

            elif directive_type == 'CATALYZE_TRANSFORMATION':
                target_agent_instance_name = directive['target_agent_instance']
                new_initial_intent = directive.get('new_initial_intent')
                new_description = directive.get('new_description')
                new_memetic_kernel_config_updates = directive.get('new_memetic_kernel_config_updates')

                if target_agent_instance_name not in self.agent_instances:
                    raise ValueError(f"Target agent instance '{target_agent_instance_name}' not found for CATALYZE_TRANSFORMATION.")

                target_agent = self.agent_instances[target_agent_instance_name]
                print(f"  CATALYZE_TRANSFORMATION: Initiating self-transformation for '{target_agent_instance_name}'.")
                target_agent.catalyze_transformation(
                    new_initial_intent=new_initial_intent,
                    new_description=new_description,
                    new_memetic_kernel_config_updates=new_memetic_kernel_config_updates
                )
                self._log_swarm_activity("AGENT_TRANSFORMED", "CatalystVectorAlpha",
                                         f"Agent '{target_agent_instance_name}' transformed.",
                                         {"agent": target_agent_instance_name, "updates": {"intent": new_initial_intent, "description": new_description, "mk_updates": new_memetic_kernel_config_updates}})
                print(f"  CATALYZE_TRANSFORMATION: Transformation directive processed for '{target_agent_instance_name}'.")

            elif directive_type == 'BROADCAST_SWARM_INTENT':
                swarm_name = directive['swarm_name']
                broadcast_intent_content = directive['broadcast_intent']
                alignment_threshold = directive.get('alignment_threshold', 0.7)

                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not found for BROADCAST_SWARM_INTENT.")

                swarm = self.swarm_protocols[swarm_name]
                print(f"  BROADCAST_SWARM_INTENT: Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.")
                self._log_swarm_activity("SWARM_INTENT_BROADCAST", "CatalystVectorAlpha",
                                         f"Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.",
                                         {"swarm": swarm_name, "intent": broadcast_intent_content, "threshold": alignment_threshold})

                for agent_ref in swarm.members:
                    if agent_ref in self.agent_instances:
                        agent = self.agent_instances[agent_ref]
                        agent.process_broadcast_intent(broadcast_intent_content, alignment_threshold)
                    else:
                        print(f"  Warning: Agent '{agent_ref}' not found in instance list, skipping broadcast.")
                        self._log_swarm_activity("WARNING", "CatalystVectorAlpha",
                            f"Agent '{agent_ref}' not found for intent broadcast.", {"agent": agent_ref, "swarm": swarm_name})
                self._log_swarm_activity("BROADCAST_PROCESSED", "CatalystVectorAlpha",
                                         f"Broadcast processed for '{swarm_name}'.", {"swarm": swarm_name})


            elif directive_type == 'AGENT_PERFORM_TASK':
                agent_name = directive['agent_name']
                task_description = directive['task_description']
                reporting_agents_ref = directive.get('reporting_agents', [])
                text_content = directive.get('text_content', '')

                if agent_name not in self.agent_instances:
                    raise ValueError(f"Agent '{agent_name}' not found for AGENT_PERFORM_TASK.")

                # Ensure reporting_agents_ref is a list for consistency
                if isinstance(reporting_agents_ref, str):
                    reporting_agents_list = [reporting_agents_ref]
                else:
                    reporting_agents_list = reporting_agents_ref

                agent = self.agent_instances[agent_name]
                print(f"  AGENT_PERFORM_TASK: Agent '{agent_name}' performing task: '{task_description}'.")

                outcome = agent.perform_task(task_description,
                                             cycle_id=self.current_action_cycle_id,
                                             reporting_agents=reporting_agents_list,
                                             text_content=text_content)

                print(f"  [MemeticKernel] {agent.name} reflects: '{agent.memetic_kernel.reflect()}'")

            elif directive_type == 'SWARM_COORDINATE_TASK':
                swarm_name = directive['swarm_name']
                task_description = directive['task_description']

                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not found for task coordination.")

                swarm = self.swarm_protocols[swarm_name]
                swarm.coordinate_task(task_description)
                self._log_swarm_activity("SWARM_COORDINATED_TASK", "CatalystVectorAlpha",
                                         f"Swarm '{swarm_name}' coordinated task '{task_description}'.",
                                         {"swarm": swarm_name, "task": task_description})

            elif directive_type == 'REPORTING_AGENT_SUMMARIZE':
                reporting_agent_name_from_manifest = directive['reporting_agent_name']
                cycle_id_to_summarize = directive.get('cycle_id', None)

                if reporting_agent_name_from_manifest not in self.agent_instances:
                    raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' not found for REPORTING_AGENT_SUMMARIZE.")

                agent = self.agent_instances[reporting_agent_name_from_manifest]
                # Assuming ProtoAgent_Observer is defined and imported or available in this scope
                if not isinstance(agent, ProtoAgent_Observer):
                    raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' is not an Observer. Only Observer agents can summarize reports.")

                print(f"  REPORTING_AGENT_SUMMARIZE: Agent '{reporting_agent_name_from_manifest}' summarizing reports for cycle '{cycle_id_to_summarize}'.")
                agent.summarize_received_reports(cycle_id=cycle_id_to_summarize)
                self._log_swarm_activity("AGENT_REPORT_SUMMARIZED", "CatalystVectorAlpha",
                                         f"Agent '{reporting_agent_name_from_manifest}' summarized reports.",
                                         {"agent": reporting_agent_name_from_manifest, "cycle_id": cycle_id_to_summarize})

            elif directive_type == 'AGENT_ANALYZE_AND_ADAPT':
                agent_name = directive['agent_name']
                if agent_name not in self.agent_instances:
                    raise ValueError(f"Agent '{agent_name}' not found for AGENT_ANALYZE_AND_ADAPT.")

                agent = self.agent_instances[agent_name]
                print(f"  AGENT_ANALYZE_AND_ADAPT: Agent '{agent_name}' performing reflexive analysis and adaptation.")
                agent.analyze_and_adapt()
                self._log_swarm_activity("AGENT_ANALYZE_ADAPT", "CatalystVectorAlpha",
                                         f"Agent '{agent_name}' performed analysis and adaptation.",
                                         {"agent": agent_name})

            elif directive_type == 'BROADCAST_COMMAND':
                target_agent = directive['target_agent']
                command_type = directive['command_type']
                command_params = directive.get('command_params', {})

                if target_agent not in self.agent_instances:
                    raise ValueError(f"Target agent '{target_agent}' not found for BROADCAST_COMMAND.")

                agent = self.agent_instances[target_agent]
                print(f"  BROADCAST_COMMAND: Agent '{target_agent}' received command '{command_type}' with params: {command_params}.")
                # Check if the agent has a process_command method
                if hasattr(agent, 'process_command') and callable(getattr(agent, 'process_command')):
                    agent.process_command(command_type, command_params)
                else:
                    print(f"  Warning: Agent '{target_agent}' does not support 'process_command' method. Command skipped.")
                    self._log_swarm_activity("AGENT_COMMAND_SKIPPED", "CatalystVectorAlpha",
                                             f"Agent '{target_agent}' does not support command '{command_type}'.",
                                             {"agent": target_agent, "command_type": command_type, "params": command_params})

                self._log_swarm_activity("AGENT_COMMANDED", "CatalystVectorAlpha",
                                         f"Agent '{target_agent}' received command '{command_type}'.",
                                         {"agent": target_agent, "command_type": command_type, "params": command_params})

            elif directive_type == 'INITIATE_PLANNING_CYCLE': # <<< NEW DIRECTIVE HANDLING >>>
                planner_agent_name = directive['planner_agent_name']
                high_level_goal = directive['high_level_goal']

                if planner_agent_name not in self.agent_instances:
                    raise ValueError(f"Planner agent '{planner_agent_name}' not found for INITIATE_PLANNING_CYCLE.")

                planner_agent = self.agent_instances[planner_agent_name]
                # Assuming ProtoAgent_Planner is defined and imported or available in this scope
                if not isinstance(planner_agent, ProtoAgent_Planner):
                    raise ValueError(f"Agent '{planner_agent_name}' is not a Planner agent. Only Planners can initiate planning cycles.")

                print(f"  INITIATE_PLANNING_CYCLE: Planner '{planner_agent_name}' initiating planning for goal: '{high_level_goal}'.")

                planner_agent.perform_task(
                    task_description=f"Initiate planning for goal: {high_level_goal}", # General task
                    high_level_goal=high_level_goal, # Pass the high-level goal
                    cycle_id=self.current_action_cycle_id # Pass current cycle ID
                )
                self._log_swarm_activity("PLANNING_CYCLE_INITIATED_BY_MANIFEST", "CatalystVectorAlpha",
                                         f"Manifest initiated planning cycle for '{planner_agent_name}'.",
                                         {"planner": planner_agent_name, "goal": high_level_goal})
            elif directive_type == 'INJECT_EVENT':
                event_type = directive['event_type']
                payload = directive['payload']
                target_agents_ref = directive.get('target_agents', list(self.agent_instances.keys()))

                # --- FIX: Ensure target_agents_list is defined here ---
                if isinstance(target_agents_ref, str):
                    target_agents_list = [target_agents_ref]
                else:
                    target_agents_list = target_agents_ref
                # --- END FIX ---

                # Generate a unique event_id and timestamp at injection
                event_id = directive.get('event_id', f"EVT-{int(time.time() * 1000)}_{random.randint(0, 999)}")
                event_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                full_event = {
                    'type': event_type,
                    'payload': payload,
                    'event_id': event_id,
                    'timestamp': event_timestamp,
                    'cycle_id': self.current_action_cycle_id
                }

                self.event_monitor.log_event(full_event.copy())
                print(f"  INJECT_EVENT: Injecting event '{event_type}' (ID: {event_id[:8]}) to {len(target_agents_list)} agents.")
                self._log_swarm_activity("EVENT_INJECTION_INITIATED", "CatalystVectorAlpha",
                                        f"Injecting event '{event_type}'.",
                                        {"event_type": event_type, "payload_preview": str(payload)[:100], "target_count": len(target_agents_list), "event_id": event_id})
                for agent_name in target_agents_list:
                    if agent_name in self.agent_instances:
                        agent = self.agent_instances[agent_name]
                        agent.receive_event(full_event)
                        self.event_monitor.log_agent_response(agent_name, event_id, 'perceived_event')
                    else:
                        print(f"  Warning: Target agent '{agent_name}' for INJECT_EVENT not found. Skipping.")
                        self._log_swarm_activity("WARNING", "CatalystVectorAlpha",
                                                f"Target agent '{agent_name}' not found for event injection.",
                                                {"event_type": event_type, "target_agent": agent_name, "event_id": event_id})

            elif directive_type == 'REQUEST_HUMAN_INPUT':
                message = directive['message']
                urgency = directive.get('urgency', 'medium')
                target_agent = directive.get('target_agent', 'System')
                request_cycle_id = directive.get('cycle_id', self.current_action_cycle_id)

                # Retrieve the human_request_counter from directive, default to 0 for first request
                human_request_counter = directive.get('human_request_counter', 0)

                # Track original requester for logging clarity
                requester = directive.get('requester_agent', target_agent)

                # Generate a unique request ID for this intervention.
                # This ID will be used to identify the response file and the pending request.
                # Use the provided cycle_id and requester for a more specific ID.
                # Ensure it's consistent for re-injected directives.
                request_id = directive.get('request_id', f"{requester.replace('ProtoAgent_', '').replace('_instance_1', '').lower()}_request_{request_cycle_id}")

                # --- START NEW/MODIFIED CODE FOR HUMAN INTERVENTION TRACKING ---
                if human_request_counter == 0: # Level 1: Initial request / Peer Review
                    # Store the request details in the new attribute only for the initial request
                    self.pending_human_interventions[request_id] = {
                        "request_id": request_id,
                        "message": message,
                        "urgency": urgency,
                        "target_agent": target_agent,
                        "requester_agent": requester,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status": "pending"
                    }
                    # Also, ensure the initial print statement includes the request_id
                    print(f"\n!!! HUMAN INTERVENTION REQUESTED (Initial Request, Urgency: {urgency.upper()}) !!!")
                    print(f"!!! From: {requester} (Target: {target_agent}) !!!")
                    print(f"!!! Message: {message} !!!")
                    print(f"!!! Please review logs for cycle {request_cycle_id} (Request ID: {request_id}) !!!")

                    self._log_swarm_activity("HUMAN_INPUT_REQUESTED_LEVEL1", "CatalystVectorAlpha",
                                             f"Level 1: Human input requested from {requester}. Initiating peer review.",
                                             {"message": message, "urgency": urgency, "requester": requester, "target_agent": target_agent, "cycle_id": request_cycle_id, "request_id": request_id})

                    # Send messages to other relevant agents for peer review (Level 1)
                    peer_review_message = f"Urgent: {requester} is requesting human input due to '{message}'. Please analyze situation related to cycle {request_cycle_id}."

                    if 'ProtoAgent_Planner_instance_1' in self.agent_instances:
                        self.agent_instances['ProtoAgent_Planner_instance_1'].send_message(
                            'ProtoAgent_Planner_instance_1', # recipient_agent_name
                            'PeerReviewRequest',             # message_type
                            peer_review_message,             # content
                            None,                            # task_description
                            "pending",                       # status
                            request_cycle_id                 # cycle_id
                        )
                    if 'ProtoAgent_Observer_instance_1' in self.agent_instances:
                        self.agent_instances['ProtoAgent_Observer_instance_1'].send_message(
                            'ProtoAgent_Observer_instance_1', # recipient_agent_name
                            'PeerReviewRequest',              # message_type
                            peer_review_message,              # content
                            None,                             # task_description
                            "pending",                        # status
                            request_cycle_id                  # cycle_id
                        )

                    # Re-inject the directive for the next escalation check (Level 2)
                    self.inject_directives([{
                        "type": "REQUEST_HUMAN_INPUT",
                        "message": message,
                        "urgency": urgency,
                        "target_agent": target_agent,
                        "cycle_id": request_cycle_id,
                        "human_request_counter": 1, # Increment counter for next phase
                        "requester_agent": requester,
                        "request_id": request_id # Pass the generated request_id
                    }])
                    print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 2) for next cycle check.")

                elif human_request_counter == 1: # Level 2: Pending
                    print(f"\n!!! HUMAN INTERVENTION PENDING (Level 2, Urgency: {urgency.upper()}) !!!")
                    print(f"!!! From: {requester} (Target: {target_agent}) !!!")
                    print(f"!!! Message: {message} !!!")
                    response_file_name = os.path.join(self.persistence_dir, f'control_human_response_{request_id}.json')
                    if not os.path.exists(response_file_name) or os.path.getsize(response_file_name) == 0:
                        print(f"!!! No human response in '{response_file_name}'. Escalating. !!!")
                        # Re-inject for Level 3 escalation
                        self.inject_directives([{
                            "type": "REQUEST_HUMAN_INPUT",
                            "message": message,
                            "urgency": urgency,
                            "target_agent": target_agent,
                            "cycle_id": request_cycle_id,
                            "human_request_counter": 2, # Increment counter for next phase
                            "requester_agent": requester,
                            "request_id": request_id
                        }])
                        print(f"  [Escalation] Re-injected REQUEST_HUMAN_INPUT (Level 3) for next cycle check.")
                    else:
                        print(f"!!! Human response found in '{response_file_name}'. Processing. !!!")
                        # Response found, so remove from pending and do NOT re-inject
                        if request_id in self.pending_human_interventions:
                            del self.pending_human_interventions[request_id]
                        # The actual processing of the response data will happen in handle_human_response
                        # when the dashboard's API endpoint is called.
                        return # Stop processing this directive further in this cycle

                elif human_request_counter == 2: # Level 3: Critical / Failed
                    print(f"\n!!! CRITICAL: HUMAN INTERVENTION FAILED (Level 3, Urgency: {urgency.upper()}) !!!")
                    print(f"!!! From: {requester} (Target: {target_agent}) !!!")
                    print(f"!!! Message: {message} !!!")
                    print("!!! No human response. Initiating full system pause. !!!")
                    # Create the system pause flag
                    with open(self.system_pause_file_full_path, 'w') as f:
                        f.write(f"PAUSED: Human intervention failed for request ID: {request_id}\n")
                    print(f"!!! SYSTEM PAUSED: '{self.system_pause_file_full_path}' created. !!!")
                    self._log_swarm_activity("SYSTEM_PAUSED_HUMAN_FAILURE", "CatalystVectorAlpha",
                                             f"System paused due to human intervention failure for request ID: {request_id}.",
                                             {"request_id": request_id, "message_preview": message[:100]})
                    self.pause_system(f"Human intervention failed for request ID: {request_id}")
                    # Remove from pending interventions as it's now "resolved" by pausing
                    if request_id in self.pending_human_interventions:
                        del self.pending_human_interventions[request_id]
                    return # Stop re-injecting, system is paused.
                # --- END NEW/MODIFIED CODE FOR HUMAN INTERVENTION TRACKING ---

            else:
                print(f"  [CatalystVectorAlpha] Warning: Unhandled directive type '{directive_type}'. Skipping.")
                self._log_swarm_activity("UNHANDLED_DIRECTIVE", "CatalystVectorAlpha",
                                         f"Skipped unhandled directive type '{directive_type}'.",
                                         {"directive_type": directive_type, "directive_content": str(directive)[:200]})

        except Exception as e:
            error_message = f"Exception while processing directive {directive_type}: {e}"
            print(f"ERROR: {error_message}")
            self._log_swarm_activity("DIRECTIVE_PROCESSING_ERROR", "CatalystVectorAlpha",
                                     error_message, {"directive_type": directive_type, "error": str(e), "directive": str(directive)[:200]})
            # Optionally, re-raise the exception if it's critical and should stop the loop
            # raise # Uncomment to re-raise for debugging purposes

    def _inject_random_system_stimulus(self, current_cycle_id):
        """
        Randomly injects a system-level stimulus (event or command) to promote dynamism
        and potentially break behavioral deadlocks.
        """
        print(f"\n--- Injecting Random System Stimulus (Cycle {current_cycle_id}) ---")

        stimulus_types = [
            "INJECT_EVENT",
            "BROADCAST_COMMAND_TO_AGENT",
            "INITIATE_PLANNING_CYCLE_NEW_GOAL"
        ]
        chosen_stimulus = random.choice(stimulus_types)
        injected_directives = []

        # Get all active agent names for targeting
        all_active_agent_names = list(self.agent_instances.keys())

        # If no agents are active, skip stimulus injection that requires targets
        if not all_active_agent_names and chosen_stimulus in ["INJECT_EVENT", "BROADCAST_COMMAND_TO_AGENT", "INITIATE_PLANNING_CYCLE_NEW_GOAL"]:
            print(f"  [Stimulus] Skipped {chosen_stimulus}: No active agents to target.")
            self._log_swarm_activity("RANDOM_STIMULUS_SKIPPED", "CatalystVectorAlpha",
                                     f"Random stimulus '{chosen_stimulus}' skipped: No active agents.",
                                     {"stimulus_type": chosen_stimulus, "cycle_id": current_cycle_id})
            return # Exit if no agents to target for these stimulus types

        if chosen_stimulus == "INJECT_EVENT":
            event_type = random.choice(["ResourceFluctuation", "DataAnomaly", "EnvironmentalShift", "SystemLoadIncrease", "ComponentDegradation", "NewOpportunity"])
            raw_change_factor = random.uniform(-0.6, 0.6)

            normalized_change_factor = round(abs(raw_change_factor), 2)

            if raw_change_factor < -0.1:
                direction = "negative_impact"
            elif raw_change_factor > 0.1:
                direction = "positive_impact"
            else:
                direction = "neutral_impact"

            if normalized_change_factor >= 0.5:
                urgency = "critical"
            elif normalized_change_factor >= 0.3:
                urgency = "high"
            elif normalized_change_factor >= 0.1:
                urgency = "medium"
            else:
                urgency = "low"

            payload = {'change_factor': normalized_change_factor, 'urgency': urgency, 'direction': direction}
            
            target_agents_for_directive = all_active_agent_names 
            print(f"  [Stimulus] Injected {event_type} event targeting ALL active agents (Urgency: {urgency}, Change: {normalized_change_factor}, Direction: {direction}).")

            injected_directives.append({
                "type": "INJECT_EVENT",
                "event_type": event_type,
                "payload": payload,
                "target_agents": target_agents_for_directive,
                "cycle_id": current_cycle_id
            })

        elif chosen_stimulus == "BROADCAST_COMMAND_TO_AGENT":
            target_agent_name = random.choice(all_active_agent_names)
            command_type = random.choice(["REPORT_STATUS", "PERFORM_DIAGNOSTICS", "ADJUST_PARAMETERS", "REQUEST_DATA_REFRESH"])
            command_params = {"reason": "Random system stimulus"}

            injected_directives.append({
                "type": "BROADCAST_COMMAND",
                "target_agent": target_agent_name,
                "command_type": command_type,
                "command_params": command_params,
                "cycle_id": current_cycle_id
            })
            print(f"  [Stimulus] Commanded agent {target_agent_name} to {command_type}.")

        elif chosen_stimulus == "INITIATE_PLANNING_CYCLE_NEW_GOAL":
            if 'ProtoAgent_Planner_instance_1' in self.agent_instances:
                new_goals = [
                    "Investigate unexpected resource fluctuations.",
                    "Propose new efficiency metrics for data processing.",
                    "Analyze recent environmental anomalies and their impact.",
                    "Develop contingency plan for unexpected system behavior.",
                    "Review security vulnerabilities in data transfer protocols."
                ]
                chosen_goal = random.choice(new_goals)

                injected_directives.append({
                    "type": "INITIATE_PLANNING_CYCLE",
                    "planner_agent_name": 'ProtoAgent_Planner_instance_1',
                    "high_level_goal": chosen_goal,
                    "cycle_id": current_cycle_id
                })
                print(f"  [Stimulus] Assigned new planning goal to Planner: '{chosen_goal}'.")
            else:
                print(f"  [Stimulus] Skipped INITIATE_PLANNING_CYCLE_NEW_GOAL: Planner not active.")
                return

        if injected_directives:
            self.inject_directives(injected_directives)
            self._log_swarm_activity("RANDOM_STIMULUS_INJECTED", "CatalystVectorAlpha",
                                     f"Injected random stimulus of type '{chosen_stimulus}'.",
                                     {"stimulus_type": chosen_stimulus, "directive_count": len(injected_directives), "cycle_id": current_cycle_id})
            
    def monitor_pattern_detection(self, agent_id: str, patterns_found: bool):
        """
        Monitors agent reports on pattern detection to identify stagnation.
        Trigger swarm-wide reset if multiple agents report no patterns.
        """
        # These are now initialized in __init__
        # if not hasattr(self, '_no_pattern_reports_count'):
        #     self._no_pattern_reports_count = defaultdict(int) 
        #     self.SWARM_RESET_THRESHOLD = 3 # Number of cycles without new patterns from majority
        #     self.NO_PATTERN_AGENT_THRESHOLD = 0.6 # Percentage of agents reporting no patterns

        if not patterns_found:
            self._no_pattern_reports_count[agent_id] += 1
            print(f"  Orchestrator: {agent_id} reported no new patterns for {self._no_pattern_reports_count[agent_id]} cycles.")
        else:
            self._no_pattern_reports_count[agent_id] = 0 # Reset count if patterns are found

        # Check for swarm-wide stagnation
        # Only check if there are active agents to avoid division by zero
        if self.agent_instances:
            stagnant_agents_count = sum(1 for agent_name, count in self._no_pattern_reports_count.items()
                                        if agent_name in self.agent_instances and count >= self.SWARM_RESET_THRESHOLD) # Only count active agents

            if stagnant_agents_count / len(self.agent_instances) >= self.NO_PATTERN_AGENT_THRESHOLD:
                print("\n!!! Orchestrator: Detected swarm-wide cognitive stagnation. Initiating swarm reset. !!!")
                self._log_swarm_activity("SWARM_STAGNATION_DETECTED", "CatalystVectorAlpha",
                                         "Swarm-wide cognitive stagnation detected. Initiating reset.",
                                         {"stagnant_agents_count": stagnant_agents_count, "total_agents": len(self.agent_instances)})
                # self.swarm_reset() # <<< CRITICAL FIX: COMMENT THIS LINE OUT TEMPORARILY
                # After reset, clear counts to prevent immediate re-trigger
                self._no_pattern_reports_count.clear()

    def swarm_reset(self):
        """
        Performs a swarm-wide reset to break stagnation.
        """
        print("  Swarm Reset: Resetting memory contexts for all agents...")
        for agent in self.agent_instances.values():
            agent.memetic_kernel.clear_working_memory() # Each agent's kernel needs this method
            print(f"    {agent.name} memory context reset.")
        
        # You might also want to explicitly reset planner's current goal/state if it's stuck
        if 'ProtoAgent_Planner_instance_1' in self.agent_instances:
            planner = self.agent_instances['ProtoAgent_Planner_instance_1']
            if isinstance(planner, ProtoAgent_Planner):
                planner.current_high_level_goal = None # Clear its current goal
                planner.planning_failure_count = 0 # Reset planning failures
                print(f"    Planner's current high-level goal and planning failures reset.")


        print("  Swarm Reset: Injecting diversification events...")
        self.inject_diversification_events()
        self._log_swarm_activity("SWARM_RESET_EXECUTED", "CatalystVectorAlpha",
                                 "Swarm reset executed. Diversification events injected.")

    def inject_diversification_events(self):
        """
        Injects a series of events/directives as per the Emergency Recovery Protocol.
        """
        print("\n--- EMERGENCY PROTOCOL: Injecting Specific Recovery Directives ---")

        # Your specified new_directives, translated into ISL:
        recovery_directives = [
            # 'PLANNER: Switch to predefined task lists'
            # Assuming 'Switch to predefined task lists' means setting a new high-level goal for the Planner
            {
                'type': 'INITIATE_PLANNING_CYCLE',
                'planner_agent_name': 'ProtoAgent_Planner_instance_1',
                'high_level_goal': 'Decompose and execute predefined recovery task lists for system stabilization.',
                'cycle_id': self.current_action_cycle_id # Use current cycle ID
            },
            # 'OBSERVER: Focus on external data streams only'
            {
                'type': 'AGENT_PERFORM_TASK',
                'agent_name': 'ProtoAgent_Observer_instance_1',
                'task_description': 'Strictly monitor external data streams for anomalies and critical changes. Limit internal observations.',
                'cycle_id': self.current_action_cycle_id,
                'task_type': 'ExternalMonitoring'
            },
            # 'COLLECTOR: Use last known good configuration'
            # This would ideally be a command to the Collector. Assuming 'last_known_good_config' is an internal state it can revert to.
            {
                'type': 'BROADCAST_COMMAND',
                'target_agent': 'ProtoAgent_Collector_instance_1',
                'command_type': 'REVERT_CONFIGURATION',
                'command_params': {'config_version': 'last_known_good'},
                'cycle_id': self.current_action_cycle_id
            },
            # 'OPTIMIZER: Freeze adaptation algorithms'
            # This would be a command to the Optimizer. Assuming it has a method to control its adaptation.
            {
                'type': 'BROADCAST_COMMAND',
                'target_agent': 'ProtoAgent_Optimizer_instance_1',
                'command_type': 'FREEZE_ADAPTATION',
                'command_params': {'reason': 'Emergency protocol'},
                'cycle_id': self.current_action_cycle_id
            },
            # Also, inject an event to signal the emergency itself
            {
                'type': 'INJECT_EVENT',
                'event_type': 'EmergencyRecoveryInitiated',
                'payload': {'change_factor': 1.0, 'urgency': 'critical', 'direction': 'neutral_impact', 'protocol': 'HardReset'},
                'target_agents': ['all'],
                'cycle_id': self.current_action_cycle_id
            }
        ]

        for event_directive in recovery_directives:
            self.inject_directives([event_directive])
        self._log_swarm_activity("EMERGENCY_RECOVERY_INJECTED", "CatalystVectorAlpha",
                                "Emergency Recovery Protocol directives injected.",
                                {"protocol_directives_count": len(recovery_directives)})

    def distill_self_narrative(self) -> str:
        """
        Synthesizes a self-narrative from the agent's memories,
        prioritizing recent patterns and compressed insights.
        This method will generate the '[Narrative]' output.
        """
        narrative_parts = []
        
        # 1. Look for recent PatternInsights (found by reflect_and_find_patterns)
        # Assuming PatternInsight memories have content structured as {'agent': ..., 'patterns': [...]}
        pattern_insights = [m for m in self.memetic_kernel.retrieve_recent_memories(lookback_period=10) if m['type'] == 'PatternInsight']
        
        if pattern_insights:
            # Sort by timestamp to get the most recent insights first
            pattern_insights.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Incorporate the latest few patterns
            narrative_parts.append(f"My recent cognitive scan revealed {len(pattern_insights)} pattern insights: ")
            for i, insight_mem in enumerate(pattern_insights[:3]): # Show up to 3 recent insights
                patterns_list = insight_mem['content'].get('patterns', [])
                if patterns_list:
                    # Summarize the LLM insight if it's too long, or take first few
                    pattern_summary = patterns_list[0] # Take the first pattern string
                    if isinstance(pattern_summary, str) and len(pattern_summary) > 100:
                        pattern_summary = pattern_summary[:100] + "..."
                    narrative_parts.append(f"- Pattern {i+1}: {pattern_summary}")

        # 2. Add context from current intent
        narrative_parts.append(f"My current primary intent is: '{self.current_intent}'.")

        # 3. Add a general summary from recent activities if no specific patterns were found, or as additional context
        if not pattern_insights: # If no patterns were found, summarize general activities
            recent_activities = [f"{m['type']}: {str(m['content'])[:50]}..." for m in self.memetic_kernel.retrieve_recent_memories(lookback_period=5) if m['type'] != 'PatternInsight']
            if recent_activities:
                narrative_parts.append(f"Recent activities include: {'; '.join(recent_activities)}.")
            else:
                narrative_parts.append("No significant recent activities or patterns observed.")

        # Final narrative construction and print
        final_narrative = f"My journey includes: {' '.join(narrative_parts)}"
        print(f"  [Narrative] {self.name} distilled self-narrative: {final_narrative}")
        return final_narrative

    def inject_event_to_agents(self, event_type: str, payload: dict, target_agents: Union[str, list]):
        """
        Allows the scenario to inject an event to specific agents or all agents.
        """
        event_id = f"SCN-EVT-{int(time.time() * 1000)}_{random.randint(100, 999)}"
        print(f"[SCENARIO] Injecting event '{event_type}' (ID: {event_id}) to {target_agents} agents.")
        
        agents_to_target = []
        if target_agents == 'all':
            agents_to_target = list(self.agent_instances.values())
        elif isinstance(target_agents, list):
            for agent_name in target_agents:
                if agent_name in self.agent_instances:
                    agents_to_target.append(self.agent_instances[agent_name])
                else:
                    print(f"Warning: Target agent '{agent_name}' not found for event injection.")
        
        for agent_instance in agents_to_target:
            agent_instance.receive_event({
                "event_id": event_id,
                "type": event_type,
                "payload": payload
            })
        self._log_swarm_activity("SCENARIO_EVENT_INJECTED", "ScenarioManager", f"Event '{event_type}' injected.", {"event_id": event_id, "payload": payload, "targets": target_agents})

    def get_all_agent_memories(self, lookback_period: int = 20) -> list:
        """
        Aggregates recent memories from all active agents for scenario monitoring.
        """
        all_recent_memories = []
        for agent_instance in self.agent_instances.values():
            all_recent_memories.extend(agent_instance.memetic_kernel.retrieve_recent_memories(lookback_period=lookback_period))
        # Sort by timestamp to get a chronological view
        all_recent_memories.sort(key=lambda m: m.get('timestamp', ''))
        return all_recent_memories

    def _process_dynamic_directives(self):
        """
        Pulls directives from the dynamic queue and executes them.
        This is the new, centralized loop for injected tasks.
        """
        if self.dynamic_directive_queue:
            print(f"\n--- Processing {len(self.dynamic_directive_queue)} Injected Directives ---")
            
            # Process all directives currently in the queue
            directives_to_process_now = list(self.dynamic_directive_queue)
            self.dynamic_directive_queue.clear()
            
            for i, directive in enumerate(directives_to_process_now):
                print(f"[Injected Directive {i+1}] Processing: {directive.get('type', 'N/A')}")
                try:
                    self._execute_single_directive(directive)
                except Exception as e:
                    print(f"ERROR: Failed to process injected directive {directive.get('type')}: {e}")
                    self._log_swarm_activity("INJECTED_DIRECTIVE_ERROR", "CatalystVectorAlpha",
                                             f"Failed to process injected directive {directive.get('type')}: {e}", 
                                             {"directive": directive, "error": str(e)})

            print("--- Injected Directives Processing Complete ---")

    def run_cognitive_loop(self):
        """
        The main cognitive loop of the Catalyst Vector Alpha system.
        This loop orchestrates all agents, processes directives, and handles system state.
        """
        print("Catalyst Vector Alpha (Phase 11 - Reflexive Behavioral Adaptation & Continuous Loop) Initiated...\n")
        self._log_swarm_activity("SYSTEM_STARTUP", "CatalystVectorAlpha", "System initiated, starting initial manifest processing.")

        # CRITICAL FIX: Load state before checking for agents to re-instantiate them.
        self.load_or_create_swarm_state()
        
        # --- Conditional Initial Manifest Processing / Agent Spawning ---
        if not self.agent_instances:
            # If no agents were loaded from state, process the manifest to create them.
            try:
                initial_manifest_data = yaml.safe_load(self.isl_manifest_content)
                self.isl_schema_validator.validate_manifest(initial_manifest_data)
                print("\n--- Initial Manifest Processing ---")
                
                # Use a separate queue for initial manifest directives to keep it clean.
                manifest_directives = deque(initial_manifest_data['directives'])
                while manifest_directives:
                    directive = manifest_directives.popleft()
                    self.current_action_cycle_id = f"initial_manifest_cycle_{timestamp_now().replace(':', '-').replace('Z', '')}"
                    self.message_bus.current_cycle_id = self.current_action_cycle_id
                    self.event_monitor.set_current_cycle(self.current_action_cycle_id)

                    print(f"\n[Initial Manifest Directive] Processing Directive: {directive.get('type', 'N/A')}")
                    self._execute_single_directive(directive)
                
            except Exception as e:
                print(f"CRITICAL ERROR during initial manifest processing: {e}")
                self._log_swarm_activity("CRITICAL_STARTUP_ERROR", "CatalystVectorAlpha",
                                         f"Initial manifest processing failed: {e}",
                                         {"error": str(e), "manifest_source": "embedded_manifest"}, level='error')
                return
        else:
            print("Agents loaded from previous state. Skipping initial manifest processing.")
        
        # Save state after initial setup is complete (either by loading or from manifest)
        self._save_system_state()
        self._log_swarm_activity("SYSTEM_INITIAL_SETUP_COMPLETE", "CatalystVectorAlpha", "Initial manifest processed/loaded and state saved.")

        print("\n--- Initial Setup Complete. Entering Continuous Cognitive Loop ---")
        self.message_bus.catalyst_vector_ref = self 
        self._log_swarm_activity("COGNITIVE_LOOP_START", "CatalystVectorAlpha", "Entering continuous cognitive loop.")
        
        loop_cycle_count = 0
        
        # Ensure active_scenario is properly initialized and has a reference to self
        self.active_scenario = CyberAttackScenario(
            self, is_paused_func=self.is_system_paused, pause_system_func=self.pause_system
        )

        while self.is_running:
            if self.is_system_paused():
                print("\n--- SYSTEM PAUSED: Awaiting external intervention. Check persistence_data/system_pause.flag ---")
                self._log_swarm_activity("SYSTEM_STATE", "CatalystVectorAlpha", "System is currently paused.", {"status": "paused"})
                time.sleep(10)
                continue
            
            loop_cycle_count += 1
            print(f"\n--- Cognitive Loop Cycle {loop_cycle_count} ---")
            self.current_action_cycle_id = f"loop_cycle_{timestamp_now().replace(':', '-').replace('Z', '')}_{loop_cycle_count}"
            self.message_bus.current_cycle_id = self.current_action_cycle_id
            self.event_monitor.set_current_cycle(self.current_action_cycle_id)
            self._log_swarm_activity("COGNITIVE_LOOP_CYCLE", "CatalystVectorAlpha",
                                     f"Starting Cognitive Loop Cycle {loop_cycle_count}.", {"cycle_id": self.current_action_cycle_id})
            
            # Phase 1: Ingest external stimuli (scenario events, overrides)
            self._process_intent_overrides()
            if self.active_scenario:
                print(f"DEBUG (Cycle {loop_cycle_count}): Injecting scenario events for active agents: {list(self.agent_instances.keys())}")
                self.active_scenario.inject_events_for_cycle(loop_cycle_count)
                self.active_scenario.update_scenario_phase(loop_cycle_count)
            
            # Phase 2: Process dynamically injected directives
            self._process_dynamic_directives()
            
            # Phase 3: Agents autonomously reflect and perform tasks based on their intent
            for agent_name, agent in list(self.agent_instances.items()):
                print(f"\nProcessing Agent: {agent_name}")
                if agent.is_paused():
                    print(f"  Agent {agent_name} is paused. Skipping tasks and reflection.")
                    self._log_swarm_activity("AGENT_PAUSED", agent_name,
                                             f"Agent is paused, skipping cycle.", {"agent": agent_name})
                    continue

                # Agents perform their current intent's task and then reflect on their state.
                task_outcome = agent.perform_task(agent.current_intent, context_info={"cycle_id": self.current_action_cycle_id})
                agent.analyze_and_adapt()
                
                # Check for recursion limit to prevent infinite loops
                if agent.intent_loop_count > agent.max_allowed_recursion:
                    print(f"  [Recursion Limit Exceeded] {agent_name} exceeded intent adaptation loop limit ({agent.max_allowed_recursion}). Forcing fallback and restarting.")
                    self._log_swarm_activity("RECURSION_LIMIT_EXCEEDED", agent_name,
                                             "Agent exceeded intent recursion limit. Forcing fallback intent and resetting.", {
                                                 "agent": agent_name, "current_intent": agent.current_intent, "loop_count": agent.intent_loop_count
                                             }, level='error')
                    agent.force_fallback_intent()
                    agent.reset_intent_loop_counter()
                
                # Periodic memory maintenance
                if loop_cycle_count % 5 == 0:
                    print(f"  [Agent] {agent_name} triggered for memory compression and pattern detection.")  
                    self._log_swarm_activity("MEMORY_COMPRESSION_TRIGGER", agent_name,
                                             f"Initiating memory compression process.", {"agent": agent_name, "cycle": loop_cycle_count})
                    agent.trigger_memory_compression()
                    agent.reflect_and_find_patterns() 
                    agent.distill_self_narrative()

            # Phase 4: Monitor system health
            all_active_agents = list(self.agent_instances.values())
            if all_active_agents:
                system_intervention_recommendation = self.meta_monitor.analyze_system_state(all_active_agents)
                if system_intervention_recommendation:
                    print(f"  [Meta-Monitor] System-wide issue detected: {system_intervention_recommendation}")
                    # Inject a new high-level goal for the planner to address the issue.
                    if 'ProtoAgent_Planner_instance_1' in self.agent_instances:
                        self.inject_directives([{
                            "type": "INITIATE_PLANNING_CYCLE",
                            "planner_agent_name": "ProtoAgent_Planner_instance_1",
                            "high_level_goal": system_intervention_recommendation,
                            "cycle_id": self.current_action_cycle_id
                        }])
                        self._log_swarm_activity("META_INTERVENTION_DIRECTIVE_INJECTED", "CatalystVectorAlpha",
                                                 f"Injected directive based on MetaSystemMonitor recommendation: {system_intervention_recommendation}",
                                                 {"recommendation": system_intervention_recommendation})
            
            # Phase 5: Persistence and Loop Control
            self._save_system_state()
            self._log_swarm_activity("SYSTEM_CHECKPOINT", "CatalystVectorAlpha", "System state checkpoint saved.")
            time.sleep(5)
            
# --- Main Execution --
class GlobalToolRegistry(ToolRegistry): # Assuming ToolRegistry is the base
    def __init__(self):
        super().__init__()
        # self._initialize_default_tools() # Call this if it auto-registers tools
GLOBAL_TOOL_REGISTRY = GlobalToolRegistry() # Instantiate it once

if __name__ == "__main__":
    # --- Define initial configuration parameters here ---
    PERSISTENCE_DIR = "persistence_data"
    ISL_SCHEMA_PATH = "isl_schema.yaml"
    # Other default paths that CatalystVectorAlpha's __init__ expects could also be defined here
    # E.g., SWARM_ACTIVITY_LOG = "logs/swarm_activity.jsonl" etc.

    # Ensure persistence directory exists before anything else uses it
    if not os.path.exists(PERSISTENCE_DIR):
        os.makedirs(PERSISTENCE_DIR, exist_ok=True) # Add exist_ok=True for robustness

    # Define the ISL schema content and write it to the file
    isl_schema_content = textwrap.dedent("""
    directives:
      ASSERT_AGENT_EIDOS:
        required:
          - eidos_name
          - eidos_spec
        eidos_spec_required:
          - role
          - initial_intent
          - location
      ESTABLISH_SWARM_EIDOS:
        required:
          - swarm_name
        optional:
          - initial_goal
          - initial_members
          - consensus_mechanism
          - task_description
          - description
        properties:
          swarm_name: { type: "string" }
          initial_goal: { type: "string" }
          initial_members: { type: "array", items: { type: "string" } }
          consensus_mechanism: { type: "string" }
          task_description: { type: "string" }
          description: { type: "string" }
      SPAWN_AGENT_INSTANCE:
        required:
          - eidos_name
          - instance_name
        optional:
          - initial_task
        properties:
          eidos_name: { type: "string" }
          instance_name: { type: "string" }
          initial_task: { type: "string" }
      ADD_AGENT_TO_SWARM:
        required:
          - agent_name
          - swarm_name
        properties:
          agent_name: { type: "string" }
          swarm_name: { type: "string" }
      ASSERT_GRADIENT_TRAJECTORY:
        required:
          - target_type
          - target_ref
          - autonomy_vector
          - ethical_constraints
          - self_correction_protocol
          - override_threshold
        properties:
          target_type: { type: "string", enum: ["Agent", "Swarm"] }
          target_ref: { type: "string" }
          autonomy_vector: { type: "string" }
          ethical_constraints: { type: "array", items: { type: "string" } }
          self_correction_protocol: { type: "string" }
          override_threshold: { type: "number", minimum: 0.0, maximum: 1.0 }
      CATALYZE_TRANSFORMATION:
        required:
          - target_agent_instance
        anyOf:
          - required: ["new_initial_intent"]
          - required: ["new_description"]
          - required: ["new_memetic_kernel_config_updates"]
        properties:
          target_agent_instance: { type: "string" }
          new_initial_intent: { type: "string" }
          new_description: { type: "string" }
          new_memetic_kernel_config_updates: { type: "object" }
      BROADCAST_SWARM_INTENT:
        required:
          - swarm_name
          - broadcast_intent
        optional:
          - alignment_threshold
        properties:
          swarm_name: { type: "string" }
          broadcast_intent: { type: "string" }
          alignment_threshold: { type: "number", minimum: 0.0, maximum: 1.0 }
      AGENT_PERFORM_TASK:
        required:
          - agent_name
          - task_description
        optional:
          - cycle_id
          - reporting_agents
          - on_success
          - on_failure
          - text_content
        properties:
          agent_name: { type: "string" }
          task_description: { type: "string" }
          cycle_id: { type: "string" }
          reporting_agents: { type: ["string", "array"], items: { type: "string" } }
          on_success: { type: "string" }
          on_failure: { type: "string" }
          text_content: { type: "string" }
      SWARM_COORDINATE_TASK:
        required:
          - swarm_name
          - task_description
        properties:
          swarm_name: { type: "string" }
          task_description: { type: "string" }
      REPORTING_AGENT_SUMMARIZE:
        required:
          - reporting_agent_name
        optional:
          - cycle_id
        properties:
          reporting_agent_name: { type: "string" }
          cycle_id: { type: "string" }
      AGENT_ANALYZE_AND_ADAPT:
        required:
          - agent_name
        properties:
          agent_name: { type: "string" }
      BROADCAST_COMMAND:
        required:
          - target_agent
          - command_type
        optional:
          - command_params
        properties:
          target_agent: { type: "string" }
          command_type: { type: "string" }
          command_params: { type: "object" }
      INITIATE_PLANNING_CYCLE:
        required:
          - planner_agent_name
          - high_level_goal
        properties:
          planner_agent_name: { type: "string" }
          high_level_goal: { type: "string" }
      INJECT_EVENT:
        required:
          - event_type
          - payload
        optional:
          - target_agents
        properties:
          event_type: { type: "string" }
          payload: { type: "object", required: ["change_factor", "urgency", "direction"] }
          target_agents: { type: ["string", "array"], items: { type: "string" } }
      REQUEST_HUMAN_INPUT:
        required:
            - message
        optional:
            - cycle_id
            - urgency
            - target_agent
            - human_request_counter
            - requester_agent
        properties:
            message: { type: "string" }
            cycle_id: { type: "string" }
            urgency: { type: "string", enum: ["low", "medium", "high", "critical"] }
            target_agent: { type: "string" }
            human_request_counter: { type: "integer", minimum: 0 }
            requester_agent: { type: "string" }
    """)
    with open(ISL_SCHEMA_PATH, 'w') as f:
        f.write(isl_schema_content)
    print(f"Successfully loaded ISL Schema: {ISL_SCHEMA_PATH}")
    print("[IP-Integration] The Eidos Protocol System is initiating, demonstrating the Gemini™ wordmark in its functionality.")

    # Initialize MessageBus and EventMonitor instances
    # These should be created once at the top level and passed to CatalystVectorAlpha
    message_bus_instance = MessageBus()
    event_monitor_instance = EventMonitor()

    # Setup the central logger that CatalystVectorAlpha will use and pass down
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    central_logger = logging.getLogger('CatalystLogger')

    # Instantiate CatalystVectorAlpha
    # Pass all required dependencies and configuration paths
    catalyst_alpha = CatalystVectorAlpha(
    message_bus=message_bus_instance,
    tool_registry=GLOBAL_TOOL_REGISTRY,
    event_monitor=event_monitor_instance,
    external_log_sink=central_logger,

    persistence_dir="persistence_data", # This remains the same
    swarm_activity_log_basename="swarm_activity.jsonl", # <--- New parameter name
    swarm_activity_log_subdir="logs", # <--- New parameter name
    system_pause_file_basename="system_pause.flag", # <--- New parameter name
    swarm_state_file_basename="swarm_state.json", # <--- New parameter name
    paused_agents_file_basename="paused_agents.json", # <--- New parameter name
    isl_schema_path="isl_schema.yaml",
    chroma_db_subdir="chroma_db", # <--- New parameter name
    intent_override_prefix="intent_override_"
)
    
    # --- CRITICAL: System Startup Sequence (Moved from __init__) ---
    # Log initial object initialization
    # This initial log can now use the fully configured catalyst_alpha instance
    catalyst_alpha._log_swarm_activity("SYSTEM_INITIALIZED", "CatalystVectorAlpha",
                                        "CatalystVectorAlpha object initialized and ready to start. Demonstrating Gemini™ wordmark use.",
                                        {"trademark_use": "Gemini"})
    
    # Load previous system state (this will call _load_system_state within CatalystVectorAlpha)
    catalyst_alpha._load_system_state()
    # --- END CRITICAL ---

    # Start the continuous cognitive loop
    catalyst_alpha.run_cognitive_loop()
    print("\nCatalyst Vector Alpha (Phase 11) Execution Finished.")