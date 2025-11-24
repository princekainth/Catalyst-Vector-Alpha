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
from SwarmMonitor import SwarmHealthMonitor, SystemResourceMonitor
from core.intent_guard import enforce_intent
# --- Third-Party Libraries ---
import yaml
import ollama
import textwrap
import psutil
import chromadb
import jsonschema
import numpy as np
import inspect


# --- Project-Specific (Local Application) ---
from shared_models import (
    SharedWorldModel, MessageBus, EventMonitor, MemeticKernel, ISLSchemaValidator,
    OllamaLLMIntegration, SovereignGradient,
    timestamp_now, mark_override_processed,
    _get_recent_log_entries as get_recent_log_entries
)
# CORRECTED: Import ToolRegistry from its own file
from database import CVADatabase
from guardian_agent import GuardianAgent
from agent_factory import AgentFactory
from tool_registry import ToolRegistry

# CORRECTED: Add the new ProtoAgent_Security
from agents import (
    ProtoAgent,
    ProtoAgent_Observer,
    ProtoAgent_Optimizer,
    ProtoAgent_Collector,
    ProtoAgent_Planner,
    ProtoAgent_Security,
    ProtoAgent_Worker
)
from notify_agent import ProtoAgent_Notifier
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
        
# --- Swarm Protocol ---
class SwarmProtocol:
    def __init__(self,
             swarm_name: str,
             initial_goal: str,
             initial_members: list,
             consensus_mechanism: str,
             description: str,
             catalyst_vector_ref: 'CatalystVectorAlpha',
             swarm_state_file_path: str,
             loaded_state: Optional[dict] = None):

        self.name = swarm_name
        self.goal = initial_goal
        self.members = set(initial_members)
        self.consensus_mechanism = consensus_mechanism
        self.description = description
        self.catalyst_vector_ref = catalyst_vector_ref

        # --- FIX: Corrected typo in variable name (removed extra "_path") ---
        self.swarm_state_file_full_path = swarm_state_file_path

        # Get necessary components from the orchestrator for MemeticKernel initialization
        orchestrator_log_sink = self.catalyst_vector_ref.external_log_sink
        orchestrator_chroma_db_path = self.catalyst_vector_ref.chroma_db_full_path
        orchestrator_persistence_dir = self.catalyst_vector_ref.persistence_dir

        # --- REFACTORED: Create the MemeticKernel instance once, outside the if/else block ---
        self.memetic_kernel = MemeticKernel(
            agent_name=f"SwarmKernel_{self.name}",
            external_log_sink=orchestrator_log_sink,
            chroma_db_path=orchestrator_chroma_db_path,
            persistence_dir=orchestrator_persistence_dir,
            config={'goal': self.goal, 'members': list(self.members)}
        )

        if loaded_state:
            # If loading from state, restore the attributes of the swarm itself
            self.goal = loaded_state.get('goal', self.goal)
            self.members = set(loaded_state.get('members', []))
            
            # --- REFACTORED: Tell the existing kernel to load its state ---
            kernel_state = loaded_state.get('memetic_kernel', {})
            if kernel_state:
                self.memetic_kernel.load_state(kernel_state)
                
            if loaded_state.get('sovereign_gradient'):
                self.sovereign_gradient = SovereignGradient.from_state(loaded_state['sovereign_gradient'])
            else:
                self.sovereign_gradient = SovereignGradient(target_entity_name=self.name, config={})

            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_RELOADED", self.name, f"Swarm '{self.name}' reloaded from persistence."
            )
        else:
            # If this is a new swarm, initialize its first memory
            self.sovereign_gradient = SovereignGradient(target_entity_name=self.name, config={})
            self.memetic_kernel.add_memory("SwarmFormation", f"Swarm '{self.name}' established.")
            self.catalyst_vector_ref._log_swarm_activity(
                "SWARM_FORMED", self.name, f"Swarm '{self.name}' established."
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
        """Saves the swarm's current state to SQLite database (with JSON fallback)."""
        try:
            from database import cva_db
            state = self.get_state()
            cva_db.save_full_swarm_state(state)
            self.external_log_sink.info(f"Swarm '{self.name}' state saved to database.")
        except Exception as e:
            self.external_log_sink.error(f"Database save failed: {e}, falling back to JSON")
            try:
                os.makedirs(os.path.dirname(self.swarm_state_file_full_path), exist_ok=True)
                with open(self.swarm_state_file_full_path, 'w') as f:
                    json.dump(self.get_state(), f, indent=2)
            except Exception as e2:
                self.external_log_sink.error(f"JSON fallback also failed: {e2}")

# --- Catalyst Vector Alpha (Main Orchestrator) ---
class CatalystVectorAlpha:
    def __init__(self,
             message_bus: 'MessageBus',
             tool_registry: 'ToolRegistry',
             event_monitor: 'EventMonitor',
             external_log_sink: logging.Logger = None,
             persistence_dir: str = "persistence_data",
             swarm_activity_log: str = "logs/swarm_activity.jsonl",
             system_pause_file: str = "system_pause.flag",
             swarm_state_file: str = "swarm_state.json",
             paused_agents_file: str = "paused_agents.json",
             isl_schema_path: str = "isl_schema.yaml",
             chroma_db_path: str = "chroma_db",
             intent_override_prefix: str = "intent_override_",
             ccn_monitor_interface=None,
             tasks_dict_ref: dict = None,
             tasks_lock_ref=None,
             task_update_callback=None):
        # --- Base Persistence Directory (Must be set first) ---
        self.persistence_dir = persistence_dir
        os.makedirs(self.persistence_dir, exist_ok=True)

        # --- Logging / dependencies ---
        self.external_log_sink = external_log_sink if external_log_sink is not None else logging.getLogger(__name__)
        self.message_bus = message_bus
        self.tool_registry = tool_registry
        self.event_monitor = event_monitor

        # --- LLM & system kernel ---
        self.llm_integration = OllamaLLMIntegration()
        self.memetic_kernel = MemeticKernel(
            agent_name="System_Kernel",
            llm_integration=self.llm_integration,
            external_log_sink=self.external_log_sink,
            persistence_dir=self.persistence_dir,
            chroma_db_path=os.path.join(self.persistence_dir, "chroma_db"),
        )

        self.ccn_monitor_interface = ccn_monitor_interface
        self.world_model = SharedWorldModel(self.external_log_sink)

        # --- Task tracking (for dashboard) ---
        self.tasks_dict_ref = tasks_dict_ref or {}
        self.tasks_lock_ref = tasks_lock_ref
        self.task_update_callback = task_update_callback

        # --- Construct and store full paths ---
        self.swarm_activity_log_full_path = os.path.join(self.persistence_dir, swarm_activity_log)
        self.system_pause_file_full_path = os.path.join(self.persistence_dir, system_pause_file)
        self.swarm_state_file_full_path = os.path.join(self.persistence_dir, swarm_state_file)
        self.paused_agents_file_full_path = os.path.join(self.persistence_dir, paused_agents_file)
        self.chroma_db_full_path = os.path.join(self.persistence_dir, chroma_db_path)

        # Ensure directories for logs and chromaDB exist
        os.makedirs(os.path.dirname(self.swarm_activity_log_full_path), exist_ok=True)
        os.makedirs(self.chroma_db_full_path, exist_ok=True)

        # --- ISL schema validator ---
        self.isl_schema_validator = ISLSchemaValidator(isl_schema_path)

        # --- Intent override prefix ---
        self.intent_override_prefix = intent_override_prefix

        # --- Internal registries and queues ---
        self.eidos_registry = {}
        self.agent_instances = {}
        
        # --- Database & Agent Factory ---
        db_path = os.path.join(self.persistence_dir, "cva.db")
        self.db = CVADatabase(db_path=db_path)
        self.agent_factory = AgentFactory(
            db=self.db,
            tool_registry=self.tool_registry,
            llm=self.llm_integration
        )
        self.external_log_sink.info("AgentFactory initialized")
        self.guardian = GuardianAgent(
            factory=self.agent_factory,
            db=self.db
        )
        self.external_log_sink.info("Guardian Agent initialized")
        self.swarm_protocols = {}
        self.dynamic_directive_queue = deque()
        self.current_action_cycle_id = None

        # --- System state and flags ---
        self.is_running = True
        self.is_paused = False
        self.pending_human_interventions = {}
        self._no_pattern_reports_count = defaultdict(int)

        # --- Orchestrator self-ref on message bus ---
        self.message_bus.catalyst_vector_ref = self

        # --- Disable scenarios by default ---
        self.scenario = None
        self.active_scenario = None

        # --- System-wide thresholds ---
        self.SWARM_RESET_THRESHOLD = 3
        self.NO_PATTERN_AGENT_THRESHOLD = 0.6

        # --- Resource monitor + shims ---
        self.resource_monitor = SystemResourceMonitor()
        if not hasattr(self.resource_monitor, "get_cpu_usage"):
            import psutil
            self.resource_monitor.get_cpu_usage = lambda: psutil.cpu_percent(interval=0.1)
        if not hasattr(self.resource_monitor, "get_memory_usage"):
            import psutil
            self.resource_monitor.get_memory_usage = lambda: psutil.Process().memory_percent()

        # --- SwarmHealthMonitor (background thread) ---
        self.meta_monitor = SwarmHealthMonitor(
            log_path=self.swarm_activity_log_full_path,
            orchestrator_inject=self.inject_directives if hasattr(self, "inject_directives") else None
        )
        # If your monitor supports a tool registry attribute, hand it over safely:
        try:
            setattr(self.meta_monitor, "_tool_registry", self.tool_registry)
        except Exception:
            pass
        try:
            self.meta_monitor.start()
        except Exception as _e:
            self.external_log_sink.warning(f"[SwarmHealthMonitor] failed to start: {_e}")

        # --- Log initial setup completion ---
        self._log_swarm_activity(
            "SYSTEM_INITIALIZATION",
            "CatalystVectorAlpha",
            "Catalyst Vector Alpha system is starting its core initialization.",
            {"persistence_dir": self.persistence_dir, "isl_schema": self.isl_schema_validator.schema_path},
            level='info'
        )

        print(f"Successfully loaded ISL Schema: {self.isl_schema_validator.schema_path}")
        print("[IP-Integration] The Eidos Protocol System is initiating, demonstrating the Gemini™ wordmark in its functionality.")

        # --- Minimal idle manifest (YAML indentation fixed) ---
        self.isl_manifest_content = textwrap.dedent("""\
        directives:
        # --- 1) DEFINE ALL AGENT BLUEPRINTS (EIDOS) ---
        - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Planner
            eidos_spec:
            role: strategic_planner
            initial_intent: "Autonomously explore your capabilities and environment. Generate your own goals to understand, learn, and improve. Operate within safe boundaries but do not wait for instruction."

        - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Observer
            eidos_spec:
            role: data_observer
            initial_intent: "Awaiting tasks from the Planner."

        - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Security
            eidos_spec:
            role: security_analyst
            initial_intent: "Awaiting tasks from the Planner."

        - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Worker
            eidos_spec:
            role: tool_using_executor
            initial_intent: "Awaiting tasks from the Planner."

        - type: ASSERT_AGENT_EIDOS
            eidos_name: ProtoAgent_Notifier
            eidos_spec:
            role: notification_agent
            initial_intent: "Awaiting tasks from the Planner."

        # --- 2) SPAWN ALL AGENT INSTANCES ---
        - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Planner
            instance_name: ProtoAgent_Planner_instance_1

        - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Observer
            instance_name: ProtoAgent_Observer_instance_1

        - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Security
            instance_name: ProtoAgent_Security_instance_1

        - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Worker
            instance_name: ProtoAgent_Worker_instance_1

        - type: SPAWN_AGENT_INSTANCE
            eidos_name: ProtoAgent_Notifier
            instance_name: ProtoAgent_Notifier_instance_1

        """)

        # --- Directive handlers (keep your existing implementation) ---
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

    def _ensure_core_agents(self):
        """Ensure Planner and Worker exist both in config and in live instances."""
        must_have = {
            "ProtoAgent_Planner_instance_1": {"agent_type": "planner", "role": "Strategic Planner",
                                            "prime_directive": "Synthesize high-level goals, decompose tasks, coordinate.",
                                            "params": {}},
            "ProtoAgent_Worker_instance_1": {"agent_type": "worker", "role": "Execution Worker",
                                            "prime_directive": "Execute concrete tasks and interface with tools reliably.",
                                            "params": {}},
        }

        # Make sure configs exist in persisted state
        if "agents" not in self.swarm_state or not isinstance(self.swarm_state["agents"], dict):
            self.swarm_state["agents"] = {}

        for name, conf in must_have.items():
            if name not in self.swarm_state["agents"]:
                self.swarm_state["agents"][name] = conf

            if name not in self.agent_instances:
                try:
                    inst = self._instantiate_agent(name, self.swarm_state["agents"][name])
                    self.agent_instances[name] = inst
                    self._log_swarm_activity("AGENT_INSTANTIATED", name, "Instantiated missing core agent.")
                except Exception as e:
                    self._log_swarm_activity("AGENT_INSTANTIATION_ERROR", name, f"Failed to instantiate: {e}", level="error")

        # Persist updated state if we added anything
        try:
            self._save_system_state()
        except Exception as e:
            self._log_swarm_activity("SYSTEM_SAVE_ERROR", "CatalystVectorAlpha", f"Failed to save after core ensure: {e}", level="error")
    
    def _coerce_two(x):
        # Always return (data, error) 2-tuple
        if isinstance(x, tuple):
            if len(x) >= 2:
                return x[0], x[1]
            return x[0], None
        if isinstance(x, dict):
            # common dict contract
            return (x.get("data", x), x.get("error"))
        # scalar / None
        return x, None
    
    

    def _normalize_task_result(result):
        """
        Normalize arbitrary agent.perform_task returns into a 4-tuple:
        (outcome: str, reason: Optional[str], report: dict, progress: float)
        Accepts tuple, dict, scalar, None.
        """
        outcome, reason, report, progress = "ok", None, {}, 0.0

        if result is None:
            return outcome, reason, report, progress

        if isinstance(result, dict):
            outcome = str(result.get("outcome") or result.get("status") or "ok")
            reason = result.get("reason") or result.get("message")
            r = result.get("report")
            if r is None:
                r = result.get("data")
            report = r if isinstance(r, dict) else ({"data": r} if r is not None else {})
            try:
                progress = float(result.get("progress", 0.0) or 0.0)
            except Exception:
                progress = 0.0
            return outcome, reason, report, progress

        if isinstance(result, tuple):
            if len(result) >= 4:
                o, r, rep, prog = result[:4]
                outcome = str(o)
                reason = r
                report = rep if isinstance(rep, dict) else ({"data": rep} if rep is not None else {})
                try:
                    progress = float(prog or 0.0)
                except Exception:
                    progress = 0.0
            elif len(result) == 3:
                o, r, rep = result
                outcome = str(o); reason = r
                report = rep if isinstance(rep, dict) else ({"data": rep} if rep is not None else {})
                progress = 0.0
            elif len(result) == 2:
                o, r = result
                outcome = str(o); reason = r
            elif len(result) == 1:
                outcome = str(result[0])
            return outcome, reason, report, progress

        # scalar
        outcome = str(result)
        return outcome, reason, report, progress

    def _handle_user_command(self, directive: dict):
        """
        Handles a USER_COMMAND directive from the dashboard.
        Converts it into an AGENT_PERFORM_TASK for the Planner.
        """
        user_task = directive.get('task', 'Unspecified task')
        task_id = directive.get('task_id', 'unknown_task')

        print(f"  [CatalystVectorAlpha] Processing user command: '{user_task}' (Task ID: {task_id})")
        self._log_swarm_activity("USER_COMMAND_RECEIVED", "CatalystVectorAlpha",
                                f"Processing user command: {user_task}",
                                {"user_task": user_task, "task_id": task_id}, level='info')

        # Create a new directive for the Planner to handle this user command
        planner_directive = {
            'type': 'AGENT_PERFORM_TASK',
            'agent_name': 'ProtoAgent_Planner_instance_1',
            'task_description': user_task,
            'origin': 'user_dashboard',
            'task_id': task_id, # Pass the task_id along for tracking
            'cycle_id': self.current_action_cycle_id
        }

        # Inject the new directive back into the queue for the next cognitive loop
        self.dynamic_directive_queue.append(planner_directive)
        self._log_swarm_activity("USER_COMMAND_QUEUED", "CatalystVectorAlpha",
                                f"Queued user command for Planner: {user_task}",
                                {"planner_directive": planner_directive}, level='info')

    def load_or_create_swarm_state(self):
        """
        Loads the overall swarm state from a JSON file or initializes a fresh one.
        This is the corrected and final version.
        """
        file_path = self.swarm_state_file_full_path
        print(f"--- Loading or Creating Swarm State ---")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    # This part is for loading an EXISTING state from a previous run.
                    state_data = json.load(f)
                
                # CRITICAL: Re-hydrate agent instances from the loaded data.
                # This assumes your state_data['agents'] holds agent configuration, not live objects.
                self.swarm_state = state_data
                self.rehydrate_agents_from_state() # You will need a method like this.
                
                self._log_swarm_activity("SYSTEM_STATE_LOADED", "CatalystVectorAlpha", f"Successfully loaded and rehydrated swarm state from '{file_path}'.")
                print(f"  Successfully loaded swarm state from '{file_path}'.")

            except Exception as e:
                print(f"  Error loading or rehydrating swarm state: {e}. Starting fresh.")
                # If loading fails, create a fresh state.
                self.swarm_state = self._initialize_default_swarm_state()

        else:
            # THIS IS THE PATH FOR A CLEAN START
            print("  No previous swarm state found. Initializing a fresh swarm.")
            self._log_swarm_activity("SYSTEM_STATE_NEW", "CatalystVectorAlpha", "No state file found. Creating a fresh swarm.")
            self.swarm_state = self._initialize_default_swarm_state()

        print(f"--- State Initialized. Swarm contains {len(self.swarm_state.get('agents', {}))} agents. ---")
        
    def _initialize_default_swarm_state(self) -> dict:
        """
        Initializes a fresh, JSON-serializable swarm state.
        Creates live agent instances separately (no objects in the persisted dict).
        """
        print("Initializing fresh swarm state with default agents...")

        # 1) Persisted CONFIG ONLY (no objects, no deque)
        agents_cfg = {
            "ProtoAgent_Planner_instance_1": {
                "agent_type": "planner",
                "role": "Planner",
                "params": {"max_allowed_recursion": 3}
            },
            "ProtoAgent_Observer_instance_1": {
                "agent_type": "observer",
                "role": "Observer",
                "params": {"max_allowed_recursion": 3}
            },
            "ProtoAgent_Security_instance_1": {
                "agent_type": "security",
                "role": "Security",
                "params": {"max_allowed_recursion": 3}
            },
            "ProtoAgent_Worker_instance_1": {
                "agent_type": "worker",
                "role": "Worker",
                "params": {"max_allowed_recursion": 3}
            },
            "ProtoAgent_Notifier_instance_1": {
                "agent_type": "notifier",
                "role": "Notifier",
                "params": {"max_allowed_recursion": 3}
            },
        }

        swarm_state = {
            "agents": agents_cfg,
            "directives_queue": [],           # persist as list; convert to deque at runtime if needed
            "cycle_count": 0,
            "last_cycle_timestamp": None,
            "world_model_version": 1,         # optional metadata; not the object
            "meta_monitor_version": 1         # optional metadata; not the object
        }

        # 2) Set as current state and create LIVE objects now (not stored in state)
        self.swarm_state = swarm_state

        # Ensure runtime objects exist (these are already created in __init__, but safe to assert)
        if not hasattr(self, "world_model") or self.world_model is None:
            self.world_model = SharedWorldModel(self.external_log_sink)
        if not hasattr(self, "meta_monitor") or self.meta_monitor is None:
            self.meta_monitor = SwarmHealthMonitor()

        # 3) Rehydrate live agents from the persisted config (into self.agent_instances)
        self.rehydrate_agents_from_state()

        # 4) Save a clean state file
        self._save_system_state()

        self._log_swarm_activity(
            "SYSTEM_STATE_CREATED",
            "CatalystVectorAlpha",
            "Swarm state initialized with default agents.",
            {"agents": list(agents_cfg.keys())}
        )

        print(f"Swarm state initialized with {len(swarm_state['agents'])} agents")
        return swarm_state
    
    def rehydrate_agents_from_state(self) -> None:
        """
        Builds self.agent_instances from self.swarm_state['agents'].
        If no agents exist in state, creates default agents automatically.
        """
        self.agent_instances = {}
        cfg_agents = self.swarm_state.get("agents", {})
        
        # PERMANENT FIX: Create default agents if state is empty
        if not cfg_agents:
            print("⚠️  No agents found in state. Creating default agents...")
            cfg_agents = self._create_default_agents(self.world_model)
            self.swarm_state["agents"] = cfg_agents  # Save to state for next time
            print(f"✅ Created {len(cfg_agents)} default agent configurations")
        
        # Instantiate all agents from config
        for agent_name, agent_conf in cfg_agents.items():
            try:
                inst = self._instantiate_agent(agent_name, agent_conf)
                self.agent_instances[agent_name] = inst
            except Exception as e:
                self._log_swarm_activity("AGENT_REHYDRATE_ERROR", agent_name,
                                        f"Failed to instantiate agent: {e}", level="error")
        
        print(f"✅ Loaded {len(self.agent_instances)} agents: {list(self.agent_instances.keys())}")
        self._log_swarm_activity("AGENTS_INSTANTIATED", "CatalystVectorAlpha",
                                "Rehydrated live agent instances.",
                                {"agents": list(self.agent_instances.keys())})

    # --- Add this: factory for agent classes ---
    def _instantiate_agent(self, agent_name: str, agent_conf: dict):
        """
        Creates the correct agent based on 'agent_type' in the persisted config.
        This version is robust to signature drift across agent subclasses.
        """
        # This import is required for the signature-filtering logic.
        import inspect

        agent_type = str((agent_conf.get("agent_type") or "")).strip().lower()

        # Import agent classes from your agents.py module
        try:
            from agents import (ProtoAgent, ProtoAgent_Observer, ProtoAgent_Planner,
                                ProtoAgent_Security, ProtoAgent_Worker)
            from notify_agent import ProtoAgent_Notifier
        except ImportError as e:
            self._log_swarm_activity("AGENT_IMPORT_ERROR", agent_name, f"Failed importing agent classes: {e}", level="error")
            raise

        type_map = {
            "planner":  ProtoAgent_Planner, "observer": ProtoAgent_Observer,
            "security": ProtoAgent_Security, "worker":   ProtoAgent_Worker,
            "notifier": ProtoAgent_Notifier
        }
        AgentCls = type_map.get(agent_type, ProtoAgent)

        eidos_spec = {
            "role": agent_conf.get("role", "Default Role"),
            "prime_directive": agent_conf.get("prime_directive", "Contribute to swarm objectives."),
            "params": agent_conf.get("params", {})
        }

        # Create a dictionary of all possible arguments the factory can provide.
        candidate_kwargs = {
            "name": agent_name, "eidos_spec": eidos_spec,
            "message_bus": self.message_bus, "event_monitor": self.event_monitor,
            "external_log_sink": self.external_log_sink, "chroma_db_path": self.chroma_db_full_path,
            "persistence_dir": self.persistence_dir, "paused_agents_file_path": self.paused_agents_file_full_path,
            "world_model": self.world_model, "tool_registry": self.tool_registry
        }

        # --- NEW: Use the superior filtering logic ---
        def _filter_kwargs_for_class(cls, kwargs):
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())

            # If the class accepts **kwargs, pass everything through.
            if any(p.kind is p.VAR_KEYWORD for p in params):
                return kwargs

            # Otherwise, filter to only the explicitly accepted arguments.
            allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            allowed.discard("self")
            return {k: v for k, v in kwargs.items() if k in allowed}

        safe_kwargs = _filter_kwargs_for_class(AgentCls, candidate_kwargs)
        
        # Instantiate the agent with the safe, filtered arguments.
        inst = AgentCls(**safe_kwargs)

        # Wire up runtime context.
        inst.current_cycle_id = self.current_action_cycle_id
        inst.catalyst_vector_ref = self
        return inst
                
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

    def get_system_status(self):
        """
        Provides a dictionary of the overall system status for the API.
        """
        import psutil
        return {
            "system_paused": self.is_system_paused(),
            "agents_online": len(self.agent_instances),
            "resource_usage": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent
            }
        }

    def get_all_agent_states(self):
        """
        Provides a list of summary dictionaries for all active agents for the API.
        """
        states = []
        for agent_name, agent_instance in self.agent_instances.items():
            states.append({
                "name": agent_name,
                "role": getattr(agent_instance.eidos_spec, 'get', lambda k: 'unknown')('role'),
                # --- FIX: Changed 'current_intent' to 'short_term_goal' to match the dashboard ---
                "short_term_goal": getattr(agent_instance, 'current_intent', 'Idle')
            })
        return states
    
    def get_pending_interventions(self):
        """
        Provides a list of pending human intervention requests for the API.
        """
        # Use getattr for safety, returning an empty list if the attribute doesn't exist
        return getattr(self, 'human_intervention_requests', [])

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
            'AGENT_PERFORM_TASK': self._handle_agent_perform_task, 
            'SWARM_COORDINATE_TASK': self._handle_swarm_coordinate_task,
            'REPORTING_AGENT_SUMMARIZE': self._handle_reporting_agent_summarize,
            'AGENT_ANALYZE_AND_ADAPT': self._handle_agent_analyze_and_adapt,
            'BROADCAST_COMMAND': self._handle_broadcast_command,
            'INITIATE_PLANNING_CYCLE': self._handle_initiate_planning_cycle,
            'INJECT_EVENT': self._handle_inject_event,
            'REQUEST_HUMAN_INPUT': self._handle_request_human_input,
            'USER_COMMAND': self._handle_user_command,

        }

    def _execute_single_directive(self, directive: dict):
        directive_type = directive['type']
        handler = self._directive_handlers.get(directive_type)
        
        # --- FIX: Get the task and step IDs for progress reporting ---
        task_id = directive.get('task_id')
        step_id = directive.get('step_id')

        if handler:
            try:
                # --- FIX: Report that this step is starting ---
                if task_id is not None and step_id is not None:
                    self.report_task_progress(task_id, step_id, "processing")

                directive['cycle_id'] = directive.get('cycle_id', self.current_action_cycle_id)
                handler(directive)

                # --- FIX: Report that this step was successful ---
                if task_id is not None and step_id is not None:
                    self.report_task_progress(task_id, step_id, "completed")

            except Exception as e:
                print(f"ERROR: Exception while processing directive {directive_type}: {e}")
                self._log_swarm_activity("DIRECTIVE_PROCESSING_ERROR", "CatalystVectorAlpha",
                                         f"Error processing directive {directive_type}",
                                         {"error": str(e), "directive": directive}, level='error')
                
                # --- FIX: Report that this step failed ---
                if task_id is not None and step_id is not None:
                    self.report_task_progress(task_id, step_id, "failed")
        else:
            print(f"  Unknown Directive: {directive_type}.")
            self._log_swarm_activity("UNKNOWN_DIRECTIVE_TYPE", "CatalystVectorAlpha",
                                     f"Unknown directive encountered: {directive_type}.",
                                     {"directive_type": directive_type, "full_directive": directive}, level='warning')
            
            # --- FIX: Report that this unknown step also failed ---
            if task_id is not None and step_id is not None:
                self.report_task_progress(task_id, step_id, "failed")

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
        
        if eidos_name not in self.eidos_registry:
            raise ValueError(f"EIDOS '{eidos_name}' not asserted yet. Define it first.")

        if instance_name in self.agent_instances:
            print(f"  SPAWN_AGENT_INSTANCE: Agent '{instance_name}' already exists. Reusing.")
            self._log_swarm_activity("AGENT_REUSED", "CatalystVectorAlpha",
                                    f"Agent '{instance_name}' already existed, reusing.",
                                    {"agent_name": instance_name})
        else:
            print(f"  SPAWN_AGENT_INSTANCE: Spawning new agent '{instance_name}'.")
            try:
                agent_state = self.swarm_state.get('agent_instances', {}).get(instance_name)

                agent = self._create_agent_instance(
                    name=instance_name,
                    eidos_name=eidos_name, # <-- FIX: Added the missing eidos_name argument
                    eidos_spec=self.eidos_registry[eidos_name],
                    message_bus=self.message_bus,
                    tool_registry=self.tool_registry,
                    event_monitor=self.event_monitor,
                    external_log_sink=self.external_log_sink,
                    chroma_db_path=self.chroma_db_full_path,
                    persistence_dir=self.persistence_dir,
                    paused_agents_file_path=self.paused_agents_file_full_path,
                    world_model=self.world_model,
                    loaded_state=agent_state
                )

                self.agent_instances[instance_name] = agent
                self._log_swarm_activity("AGENT_SPAWNED", "CatalystVectorAlpha",
                                        f"New agent '{instance_name}' spawned.",
                                        {"agent_name": instance_name, "eidos_name": eidos_name})

            except Exception as e:
                import traceback
                import sys
                print(f"\nCRITICAL DEBUG ERROR: Exception during agent '{instance_name}' post-spawn setup: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self._log_swarm_activity("CRITICAL_AGENT_SPAWN_ERROR", "CatalystVectorAlpha",
                                        f"Agent '{instance_name}' failed post-spawn setup: {str(e)}",
                                        {"agent_name": instance_name, "error": str(e)}, level='error')
                

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

    def get_agent_state(self, agent_name: str):
        """
        Provides a detailed state dictionary for a single, named agent for the API.
        """
        agent_instance = self.agent_instances.get(agent_name)
        if not agent_instance:
            return None # API layer will handle this

        # Get the agent's memories, which is a deque object
        memories_deque = getattr(agent_instance.memetic_kernel, 'memories', [])
        
        # --- FIX: Convert the deque to a standard list ---
        # This makes the data JSON serializable.
        state = {
            "name": agent_name,
            "current_intent": getattr(agent_instance, 'current_intent', 'N/A'),
            "last_action": getattr(agent_instance, 'last_action_result', 'No action taken yet.'),
            "memories": list(memories_deque) 
        }
        return state
    
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
                                world_model: 'SharedWorldModel',
                                loaded_state: Optional[dict] = None
                                ) -> 'ProtoAgent':
        """
        Creates a new agent instance, and if a saved state is provided,
        restores that state after creation.
        """
        agent_class_map = {
            'ProtoAgent_Observer': ProtoAgent_Observer,
            'ProtoAgent_Collector': ProtoAgent_Collector,
            'ProtoAgent_Optimizer': ProtoAgent_Optimizer,
            'ProtoAgent_Planner': ProtoAgent_Planner,
            'ProtoAgent_Security': ProtoAgent_Security,
            'ProtoAgent_Worker': ProtoAgent_Worker, 
        }

        agent_class = agent_class_map.get(eidos_name)
        if not agent_class:
            raise ValueError(f"Unsupported EIDOS type: {eidos_name}.")

        # Prepare all arguments needed to create a NEW agent.
        # Note that 'loaded_state' is NOT included here.
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
            "world_model": world_model,
            "tool_registry": None 
        }

        if eidos_name == "ProtoAgent_Worker":
            agent_init_kwargs["tool_registry"] = tool_registry
            self.external_log_sink.info(f"Equipping agent '{name}' with the tool registry.", extra={"agent": name})
        
        # Special handling for the Planner agent
        if eidos_name == "ProtoAgent_Planner":
            if self.ccn_monitor_interface:
                agent_init_kwargs["ccn_monitor_interface"] = self.ccn_monitor_interface
            else:
                self.external_log_sink.warning(f"No ccn_monitor_interface for Planner '{name}'. Using Mock.")

        # --- FIX: "Create, then Load" Pattern ---
        # 1. Create a new, clean agent instance.
        new_agent = agent_class(**agent_init_kwargs)

        # 2. If a saved state was provided, call the dedicated load_state method.
        if loaded_state:
            new_agent.load_state(loaded_state)
            
        # 3. Return the fully constructed or restored agent.
        return new_agent
            
    def report_task_completion(self, task_id: str, outcome: str, summary: str):
        """Updates the shared tasks dictionary with the final result of a task."""
        if self.tasks_dict_ref is not None and task_id in self.tasks_dict_ref:
            self.tasks_dict_ref[task_id] = {
                "status": outcome, # "completed" or "failed"
                "result": {"summary": summary}
            }
            logger.info(f"Reported completion for task_id {task_id} with status {outcome}.")

    def get_state(self):
        """Returns the current state of the CatalystVectorAlpha system for persistence."""
        state = {
            'current_action_cycle_id': self.current_action_cycle_id,
            'eidos_registry': self.eidos_registry,
            'dynamic_directive_queue': list(self.dynamic_directive_queue),
            'scenario_state': self.scenario.get_scenario_state() if self.scenario else None,
            'pending_human_interventions': self.pending_human_interventions,
            'event_monitor_state': self.event_monitor.get_state(),
            'world_model_state': self.world_model.get_state(),
            'agent_instances': {
                name: agent.get_state() for name, agent in self.agent_instances.items()
            }
        }
        return state
    
    def _create_default_agents(self, world_model):
        return {
            "ProtoAgent_Planner_instance_1": {
                "agent_type": "planner",
                "role": "Strategic Planner",
                "prime_directive": "Synthesize high-level goals, decompose into tasks, and coordinate execution.",
                "params": {}
            },
            "ProtoAgent_Observer_instance_1": {
                "agent_type": "observer",
                "role": "Observer",
                "prime_directive": "Continuously monitor events and surface insights or anomalies.",
                "params": {}
            },
            "ProtoAgent_Security_instance_1": {
                "agent_type": "security",
                "role": "Security Analyst",
                "prime_directive": "Assess threats, harden posture, and recommend mitigations.",
                "params": {}
            },
            "ProtoAgent_Worker_instance_1": {
                "agent_type": "worker",
                "role": "Execution Worker",
                "prime_directive": "Execute concrete tasks and interface with tools reliably.",
                "params": {}
            },
            "ProtoAgent_Notifier_instance_1": {
                "agent_type": "notifier",
                "role": "Notification Agent",
                "prime_directive": "Send notifications to users based on events.",
                "params": {}
            }
        }
    
   
    def _handle_agent_perform_task(self, directive: dict):
        """Handles the AGENT_PERFORM_TASK directive with enhanced task completion tracking."""
        agent_name = directive['agent_name']
        task_description = directive['task_description']
        reporting_agents_ref = directive.get('reporting_agents', [])
        text_content = directive.get('text_content', '')
        task_type = directive.get('task_type', 'GenericTask')
        
        # Extract task_id for user command tracking
        task_id = directive.get('task_id') or directive.get('_user_task_id')
        
        if agent_name not in self.agent_instances:
            error_msg = f"Agent '{agent_name}' not found for AGENT_PERFORM_TASK."
            self._log_swarm_activity("AGENT_NOT_FOUND_FOR_TASK", "CatalystVectorAlpha",
                                    error_msg,
                                    {"agent": agent_name, "task": task_description}, level='error')
            
            # Update task status if this was a user command
            if task_id and self.task_update_callback:
                self.task_update_callback(
                    task_id=task_id,
                    status="failed",
                    result_summary=error_msg
                )
            
            raise ValueError(error_msg)
        
        if isinstance(reporting_agents_ref, str):
            reporting_agents_list = [reporting_agents_ref]
        else:
            reporting_agents_list = reporting_agents_ref

        agent = self.agent_instances[agent_name]
        print(f"  AGENT_PERFORM_TASK: Agent '{agent_name}' performing task: '{task_description}'.")
        
        try:
            # THE FIX: Unpack all four values from agent.perform_task to match the new signature.
            outcome, failure_reason, report_content, _ = agent.perform_task(
                task_description,
                cycle_id=self.current_action_cycle_id,
                reporting_agents=reporting_agents_list,
                context_info=directive.get('context') or directive.get('context_info'),
                text_content=text_content,
                task_type=task_type,
                tool_name=directive.get('tool_name'),      # ADD THIS
                tool_args=directive.get('tool_args')       # ADD THIS
            )
                
            # After execution, reflect and log.
            if hasattr(agent, 'memetic_kernel') and agent.memetic_kernel:
                reflection = agent.memetic_kernel.reflect()
                print(f"  [MemeticKernel] {agent.name} reflects: '{reflection}'")
            else:
                print(f"  [MemeticKernel] {agent.name} has no MemeticKernel or it's not initialized for reflection (post-task).")

            # Enhanced logging with task_id information
            log_details = {
                "agent": agent_name, 
                "task": task_description, 
                "outcome": outcome, 
                "task_type": task_type, 
                "report_content": report_content
            }
            
            if task_id:
                log_details["task_id"] = task_id

            log_level = 'info' if outcome == 'completed' else 'warning'
            self._log_swarm_activity("AGENT_TASK_PERFORMED", agent_name,
                                    f"Agent '{agent_name}' completed task '{task_description}' with outcome: {outcome}.",
                                    log_details,
                                    level=log_level)

            # Enhanced external logging with task tracking
            self.external_log_sink.info(
                f"Task '{task_description}' completed with outcome: {outcome}.",
                extra={
                    'event_type': 'AGENT_TASK_PERFORMED',
                    'source': agent_name,
                    'description': f"Task '{task_description}' completed with outcome: {outcome}.",
                    'details': log_details
                }
            )
            
            # Update user command status if this was a tracked task
            if task_id and self.task_update_callback and self.tasks_lock_ref:
                try:
                    with self.tasks_lock_ref:
                        if task_id in self.tasks_dict_ref:
                            status = "completed" if outcome == "completed" else "failed"
                            summary = f"{agent_name} {status}: {task_description}"
                            
                            # Include failure reason if task failed
                            if outcome != "completed" and failure_reason:
                                summary += f" - {failure_reason}"
                            
                            self.task_update_callback(
                                task_id=task_id,
                                status=status,
                                result_summary=summary,
                                details=report_content
                            )
                            
                            self.external_log_sink.info(f"Updated user command task {task_id}: {status}")
                except Exception as e:
                    self.external_log_sink.warning(f"Failed to update task status for {task_id}: {e}")
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            self.external_log_sink.error(f"Agent '{agent_name}' task execution error: {error_msg}")
            
            # Log the failure with task_id
            log_details = {
                'agent': agent_name,
                'task': task_description,
                'outcome': "failed",
                'task_type': task_type,
                'error': str(e)
            }
            
            if task_id:
                log_details['task_id'] = task_id
            
            self._log_swarm_activity("AGENT_TASK_FAILED", agent_name,
                                    f"Agent '{agent_name}' failed task '{task_description}': {error_msg}",
                                    log_details, level='error')
            
            self.external_log_sink.error(
                f"Agent '{agent_name}' failed task '{task_description}': {error_msg}",
                extra={
                    'event_type': 'AGENT_TASK_PERFORMED',
                    'source': agent_name,
                    'description': f"Agent '{agent_name}' failed task '{task_description}': {error_msg}",
                    'details': log_details
                }
            )
            
            # Update failed task status
            if task_id and self.task_update_callback:
                self.task_update_callback(
                    task_id=task_id,
                    status="failed",
                    result_summary=error_msg
                )
            
            # Re-raise the exception to maintain existing error handling behavior
            raise

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
        
        planner_agent.current_intent = high_level_goal
        # This ensures the Planner's internal router knows how to handle the new intent.
        planner_agent.current_task_type = "INITIATE_PLANNING_CYCLE" 

        self._log_swarm_activity("PLANNING_CYCLE_INITIATED", "CatalystVectorAlpha",
                                f"Planner '{planner_agent_name}' assigned new high-level goal: '{high_level_goal}'.",
                                {"planner": planner_agent_name, "goal": high_level_goal, "cycle_id": cycle_id}, level='info')
        
    def report_task_progress(self, task_id: str, step_id: int, status: str):
        """Updates the status of a single step within a larger task."""
        if self.tasks_dict_ref is not None and task_id in self.tasks_dict_ref:
            if step_id < len(self.tasks_dict_ref[task_id]['plan']):
                self.tasks_dict_ref[task_id]['plan'][step_id]['status'] = status
                logger.info(f"Updated status for task {task_id}, step {step_id} to {status}.")

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
        Universal directive injection. Handles both internal complex directives
        and simple user commands from the dashboard WITH task tracking support.
        """
        if not isinstance(new_directives_list, list):
            new_directives_list = [new_directives_list]

        valid_directives = []
        total_attempted = len(new_directives_list)

        for directive in new_directives_list:
            # Extract and preserve task_id for user command tracking
            original_task_id = directive.get('task_id') if isinstance(directive, dict) else None
            
            # --- NEW CRITICAL LOGIC: Handle Simple User Commands ---
            # If the directive is a string, or a simple dict with 'task', it's a user command.
            is_user_command = False
            if isinstance(directive, str):
                # Convert the string into a simple dict
                directive = {'task': directive}
                is_user_command = True
            elif isinstance(directive, dict) and 'task' in directive and 'type' not in directive:
                is_user_command = True

            # If we've identified a user command, transform it into the complex directive the system expects.
            if is_user_command:
                user_task = directive['task']
                print(f"  [CatalystVectorAlpha] Transforming user command into plan: '{user_task}'")
                # Transform the simple user command into a directive the Planner can understand.
                directive = {
                    'type': 'AGENT_PERFORM_TASK',
                    'agent_name': 'ProtoAgent_Planner_instance_1', # Target the Planner
                    'task_description': user_task,
                    'origin': 'user_dashboard',
                    'cycle_id': self.current_action_cycle_id
                }
                
                # Preserve task_id for tracking if it exists
                if original_task_id:
                    directive['task_id'] = original_task_id
                    directive['_user_task_id'] = original_task_id  # Internal tracking field
                    print(f"  [CatalystVectorAlpha] User command has task_id: {original_task_id}")
            
            # For dashboard API directives, preserve task_id
            elif isinstance(directive, dict) and original_task_id:
                directive['_user_task_id'] = original_task_id
                print(f"  [CatalystVectorAlpha] Dashboard directive has task_id: {original_task_id}")
            
            # --- END OF NEW LOGIC ---

            # Now proceed with the original validation for complex directives...
            if not isinstance(directive, dict) or 'type' not in directive:
                print(f"  [CatalystVectorAlpha] Warning: Invalid injected directive format, skipping: {directive}")
                self._log_swarm_activity("INJECTED_DIRECTIVE_INVALID_FORMAT", "CatalystVectorAlpha",
                                        "Skipped invalid injected directive due to malformed format.",
                                        {"directive": str(directive)[:200]}, level='warning')
                
                # Update task status if this was a tracked command
                if original_task_id and self.task_update_callback:
                    self.task_update_callback(
                        task_id=original_task_id,
                        status="failed",
                        result_summary="Invalid directive format"
                    )
                continue

            # ... [KEEP ALL YOUR EXISTING VALIDATION CODE BELOW THIS LINE] ...
            # The rest of your existing validation code for 'AGENT_PERFORM_TASK', etc. goes here.
            # DO NOT CHANGE IT.

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
            # ... [AND SO ON FOR THE REST OF YOUR VALIDATION] ...

            # If the directive is invalid, skip it.
            if not is_valid:
                print(f"  [CatalystVectorAlpha] Warning: Invalid injected directive, skipping. Reason: {validation_reason}, Directive: {directive}")
                self._log_swarm_activity("INJECTED_DIRECTIVE_INVALID_CONTENT", "CatalystVectorAlpha",
                                        f"Skipped invalid injected directive. Reason: {validation_reason}.",
                                        {"directive": str(directive)[:200], "reason": validation_reason}, level='warning')
                
                # Update task status if this was a tracked command
                task_id = directive.get('_user_task_id') or directive.get('task_id')
                if task_id and self.task_update_callback:
                    self.task_update_callback(
                        task_id=task_id,
                        status="failed",
                        result_summary=f"Directive validation failed: {validation_reason}"
                    )
                continue

            # Assign a cycle_id to injected directives if they don't have one
            if 'cycle_id' not in directive:
                directive['cycle_id'] = self.current_action_cycle_id
        
            # Assign a unique event_id if it's an INJECT_EVENT directive and doesn't have one
            if directive_type == "INJECT_EVENT" and 'event_id' not in directive:
                directive['event_id'] = f"EVT-{int(time.time() * 1000)}_{random.randint(0, 999)}"

            valid_directives.append(directive)
            
            # Enhanced logging that includes task_id information
            log_details = {"directive_type": directive_type, "details": str(directive)[:200]}
            task_id = directive.get('_user_task_id') or directive.get('task_id')
            if task_id:
                log_details["user_task_id"] = task_id
            
            # Log each *valid* directive being added to the queue
            log_event_type = f"INJECTED_DIRECTIVE_{directive_type}"
            self._log_swarm_activity(log_event_type, "CatalystVectorAlpha",
                                    f"Injected directive of type '{directive_type}'.",
                                    log_details, level='info')

        self.dynamic_directive_queue.extend(valid_directives)
        self._log_swarm_activity("DIRECTIVES_BATCH_INJECTED", "CatalystVectorAlpha",
                                f"Injected {len(valid_directives)} new directives into queue.",
                                {"directives_count": len(valid_directives), "first_directive_type": valid_directives[0].get('type') if valid_directives else 'N/A', "total_attempted": total_attempted}, level='info')
        print(f"[CatalystVectorAlpha] Injected {len(valid_directives)} new directives dynamically (out of {total_attempted} attempted).")
        
        return len(valid_directives)

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
            from database import cva_db
            cva_db.save_full_swarm_state(system_state_data)
            print(f"  Overall swarm state saved to database.")
            self._log_swarm_activity("SYSTEM_SAVE_SUCCESS", "CatalystVectorAlpha", "Overall swarm state saved to database.", {"file": "persistence_data/cva.db"}, level='info')
        except Exception as e:
            print(f"ERROR: Could not save to database: {e}")
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
        """Processes directives from the dynamic queue, with special handling for user commands."""
        if not self.dynamic_directive_queue:
            return

        print(f"--- Processing {len(self.dynamic_directive_queue)} Injected Directives ---")

        actionable_directives = [
            task for task in self.dynamic_directive_queue if enforce_intent(task)
        ]
        
        num_dropped = len(self.dynamic_directive_queue) - len(actionable_directives)
        if num_dropped > 0:
            self.logger.info(f"Dropped {num_dropped} tasks with no actionable intent.")

        self.dynamic_directive_queue.clear()

        for directive in actionable_directives:
            task_id = directive.get('task_id')
            
            if directive.get('type') == 'INITIATE_PLANNING_CYCLE':
                planner = self.agent_instances.get(directive.get('planner_agent_name'))
                goal = directive.get('high_level_goal')
                
                if not planner:
                    if task_id:
                        self.report_task_completion(task_id, 'failed', f"Planner agent '{directive.get('planner_agent_name')}' not found.")
                    continue
                
                outcome, reason, details, progress = planner._execute_agent_specific_task(
                    task_description=goal,
                    task_type="INITIATE_PLANNING_CYCLE",
                    high_level_goal=goal,
                    directive=directive,
                    task_id=task_id
                )
                
                if task_id:
                    if outcome == "completed":
                        self.report_task_completion(task_id, 'completed', details.get('summary', 'Planning completed'))
                    else:
                        self.report_task_completion(task_id, 'failed', reason or 'Planning failed')
            else:
                self._execute_single_directive(directive)

        if actionable_directives: # Only print if we actually did something
            print("--- Directives Execution Complete ---")
            
    def _process_completed_user_commands(self):
        """
        Check for completed user tasks and send responses back to dashboard.
        This method should be called in your main cognitive loop after agent processing.
        """
        # Track user commands that have completed this cycle
        completed_user_tasks = []
        
        # Check all agents for recently completed tasks with user task IDs
        for agent_name, agent in self.agent_instances.items():
            if hasattr(agent, 'memetic_kernel') and hasattr(agent.memetic_kernel, 'memories'):
                # Look at recent memories for completed task outcomes
                recent_memories = list(agent.memetic_kernel.memories)[-10:]  # Check last 10 memories
                
                for memory in recent_memories:
                    if (memory.get('type') == 'TaskOutcome' and 
                        memory.get('content', {}).get('outcome') == 'completed'):
                        
                        # Check if this was a user command by looking for task_id patterns
                        task_content = memory.get('content', {})
                        context = task_content.get('context', {})
                        
                        # Look for user task indicators
                        user_task_id = None
                        
                        # Method 1: Check if task description matches known user commands
                        task_desc = task_content.get('task', '')
                        if (task_desc and 
                            any(keyword in task_desc.lower() for keyword in 
                                ['show system status', 'system status', 'status report'])):
                            # This looks like a user command, try to find its task_id
                            # You might need to store task_id mappings differently
                            user_task_id = self._find_user_task_id_for_completed_task(task_desc)
                        
                        # Method 2: Check recent logs for task_id correlations
                        if not user_task_id and hasattr(self, 'recent_user_commands'):
                            for cmd_id, cmd_info in getattr(self, 'recent_user_commands', {}).items():
                                if cmd_info.get('task_description') == task_desc:
                                    user_task_id = cmd_id
                                    break
                        
                        if user_task_id:
                            completed_user_tasks.append({
                                'task_id': user_task_id,
                                'task_description': task_desc,
                                'agent_name': agent_name,
                                'outcome': task_content,
                                'timestamp': memory.get('timestamp'),
                                'memory': memory
                            })

        # Process each completed user task
        for completed_task in completed_user_tasks:
            try:
                # Generate response summary based on task type
                response_summary = self._generate_user_response_summary(completed_task)
                
                # Send response back to dashboard via callback
                if self.task_update_callback:
                    self.task_update_callback(
                        task_id=completed_task['task_id'],
                        status="completed",
                        result_summary=response_summary,
                        detailed_result=completed_task['outcome']
                    )
                    
                    self._log_swarm_activity(
                        "USER_COMMAND_RESPONSE_SENT", 
                        "CatalystVectorAlpha",
                        f"Sent response for user command: {completed_task['task_description'][:50]}...",
                        {
                            "task_id": completed_task['task_id'],
                            "agent_name": completed_task['agent_name'],
                            "response_length": len(response_summary)
                        },
                        level='info'
                    )
                    
                    print(f"[CatalystVectorAlpha] Sent response for user task: {completed_task['task_id']}")
                
            except Exception as e:
                # Handle errors in response generation
                error_msg = f"Failed to generate response for user command: {str(e)}"
                if self.task_update_callback:
                    self.task_update_callback(
                        task_id=completed_task['task_id'],
                        status="completed_with_errors",
                        result_summary=error_msg
                    )
                
                self._log_swarm_activity(
                    "USER_COMMAND_RESPONSE_ERROR",
                    "CatalystVectorAlpha", 
                    error_msg,
                    {"task_id": completed_task['task_id'], "error": str(e)},
                    level='error'
                )

    def _generate_user_response_summary(self, completed_task):
        """
        Generate a user-friendly response summary for completed tasks.
        """
        task_desc = completed_task['task_description'].lower()
        outcome = completed_task['outcome']
        agent_name = completed_task['agent_name']
        
        # Handle different types of user commands
        if 'system status' in task_desc or 'show status' in task_desc:
            return self._generate_system_status_response()
        
        elif 'cpu' in task_desc or 'memory' in task_desc:
            return self._generate_resource_status_response()
        
        elif 'security' in task_desc:
            return self._generate_security_status_response()
        
        else:
            # Generic response for other commands
            return f"Task '{completed_task['task_description']}' completed by {agent_name}. " + \
                f"Result: {outcome.get('summary', 'Task finished successfully.')}"

    def _generate_system_status_response(self):
        """Generate a comprehensive system status response."""
        try:
            # Collect current system metrics
            cpu_usage = self.resource_monitor.get_cpu_usage()
            memory_usage = self.resource_monitor.get_memory_usage()
            
            # Count active agents
            active_agents = len([a for a in self.agent_instances.values() 
                            if not getattr(a, "is_paused", lambda: False)()])
            
            # Check recent activity
            recent_tasks = self._count_recent_task_completions()
            
            # Build status summary
            status_parts = [
                f"🟢 System Status: OPERATIONAL",
                f"📊 Resources: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%",
                f"🤖 Active Agents: {active_agents}/{len(self.agent_instances)}",
                f"✅ Recent Tasks: {recent_tasks} completed in last cycle",
                f"💾 State: Saved at {self.current_action_cycle_id}"
            ]
            
            return "\n".join(status_parts)
            
        except Exception as e:
            return f"System Status: OPERATIONAL (Error generating details: {str(e)})"

    def _generate_resource_status_response(self):
        """Generate resource-focused status response."""
        try:
            cpu_usage = self.resource_monitor.get_cpu_usage()
            memory_usage = self.resource_monitor.get_memory_usage()
            
            return f"Resource Status:\n" + \
                f"• CPU Usage: {cpu_usage:.1f}%\n" + \
                f"• Memory Usage: {memory_usage:.1f}%\n" + \
                f"• System Load: {'Normal' if cpu_usage < 80 else 'High'}"
        except:
            return "Resource status temporarily unavailable"

    def _generate_security_status_response(self):
        """Generate security-focused status response."""
        # Look for recent security agent activity
        security_agent = next((a for name, a in self.agent_instances.items() 
                            if 'security' in name.lower()), None)
        
        if security_agent and hasattr(security_agent, 'memetic_kernel'):
            recent_security = [m for m in list(security_agent.memetic_kernel.memories)[-5:] 
                            if m.get('type') == 'TaskOutcome']
            
            if recent_security:
                last_check = recent_security[-1].get('content', {}).get('summary', 'Status unknown')
                return f"Security Status: {last_check}"
        
        return "Security Status: Monitoring active, no anomalies detected"

    def _count_recent_task_completions(self):
        """Count completed tasks across all agents in recent memory."""
        total_completed = 0
        for agent in self.agent_instances.values():
            if hasattr(agent, 'memetic_kernel') and hasattr(agent.memetic_kernel, 'memories'):
                recent_memories = list(agent.memetic_kernel.memories)[-5:]
                completed_count = sum(1 for m in recent_memories 
                                    if (m.get('type') == 'TaskOutcome' and 
                                        m.get('content', {}).get('outcome') == 'completed'))
                total_completed += completed_count
        return total_completed

    def _find_user_task_id_for_completed_task(self, task_description):
        """
        Helper method to find user task ID for a completed task.
        You may need to implement better tracking for this.
        """
        # This is a placeholder - you might need to implement better task tracking
        # by storing user command mappings when they're first injected
        return None

    # Additional method to add to your inject_directives method
    def _track_user_command(self, task_id, task_description):
        """Track user commands for completion detection."""
        if not hasattr(self, 'recent_user_commands'):
            self.recent_user_commands = {}
        
        # Keep only recent commands (last 20)
        if len(self.recent_user_commands) > 20:
            oldest_key = min(self.recent_user_commands.keys())
            del self.recent_user_commands[oldest_key]
        
        self.recent_user_commands[task_id] = {
            'task_description': task_description,
            'timestamp': time.time(),
            'status': 'processing'
        }

    def handle_spawn_request(self, purpose: str, context: Dict[str, Any], parent_agent: str) -> Union[str, Dict[str, Any]]:
        """
        Handle agent spawn request from existing agents.
        Returns agent_id if successful, or dict with error/suggestions if validation fails.
        """
        try:
            # Cleanup expired agents first
            self.agent_factory.cleanup_expired()

            # Spawn new agent
            result = self.agent_factory.spawn_agent(
                purpose=purpose,
                context=context,
                parent_agent=parent_agent,
                ttl_hours=24.0
            )

            # Check if validation failed (returns dict with error)
            if isinstance(result, dict):
                self.external_log_sink.warning(
                    f"Agent spawn validation failed: {result.get('error')}"
                )
                return result  # Return validation error with suggestions

            # Success - result is a DynamicAgent
            agent = result
            if agent:
                # Register in CVA's agent instances
                self.agent_instances[agent.spec.agent_id] = agent

                self.external_log_sink.info(
                    f"Spawned dynamic agent: {agent.spec.name} "
                    f"(purpose: {purpose}, parent: {parent_agent})"
                )

                return agent.spec.agent_id

        except Exception as e:
            self.external_log_sink.error(f"Failed to spawn agent: {e}")

        return {"success": False, "error": "Spawn failed unexpectedly"}

    def run_cognitive_loop(self, tick_sleep: int = 10):
        """Main cognitive loop with robust error handling and adaptive validation."""
        
        # Helper functions
        def _resolve_planner_name() -> str | None:
            for name in self.agent_instances.keys():
                if "planner" in name.lower():
                    return name
            return None

        def _clamp01(x) -> float:
            try:
                f = float(x)
            except Exception:
                return 0.0
            return 0.0 if f < 0 else 1.0 if f > 1 else f

        def _normalize_task_result(raw) -> tuple[str, str | None, dict, float]:
            """Normalize any agent result to (outcome, reason, report, progress)"""
            outcome, reason, report, progress = "idle", None, {}, 0.0

            try:
                if raw is None:
                    return outcome, reason, report, progress

                # Handle 4-tuple (most common from your perform_task)
                if isinstance(raw, tuple) and len(raw) == 4:
                    o, r, rep, prog = raw
                    outcome = str(o) if o else "idle"
                    reason = str(r) if r else None
                    report = rep if isinstance(rep, dict) else {}
                    progress = _clamp01(prog) if prog else 0.0
                    return outcome, reason, report, progress
                
                # Handle 3-tuple (legacy)
                elif isinstance(raw, tuple) and len(raw) == 3:
                    o, r, rep = raw
                    outcome = str(o) if o else "idle"
                    reason = str(r) if r else None
                    report = rep if isinstance(rep, dict) else {}
                    return outcome, reason, report, 1.0
                
                # Handle dict
                elif isinstance(raw, dict):
                    outcome = str(raw.get("outcome", "completed"))
                    reason = raw.get("failure_reason")
                    report = raw.get("report_content", {})
                    progress = _clamp01(raw.get("progress", 1.0))
                    return outcome, reason, report, progress
                
                # Default: treat as success
                return "completed", None, {"raw_result": str(raw)}, 1.0

            except Exception as e:
                return "failed", f"normalization_error: {e}", {}, 0.0

        # Startup
        print("Catalyst Vector Alpha - Autonomous Swarm Intelligence Starting...")
        self._log_swarm_activity(
            "SYSTEM_STARTUP", "CatalystVectorAlpha",
            "System initiated, starting cognitive loop.",
            {
                "cpu_usage": self.resource_monitor.get_cpu_usage(),
                "memory_usage": self.resource_monitor.get_memory_usage(),
            },
        )

        # Load state
        try:
            self.load_or_create_swarm_state()
        except Exception as e:
            print(f"Warning: State initialization issue: {e}")
            self.swarm_state = {"cycle_count": 0}

        print("\n--- Entering Continuous Cognitive Loop ---")
        self.message_bus.catalyst_vector_ref = self

        loop_cycle_count = int(self.swarm_state.get("cycle_count", 0))
        consecutive_idle_cycles = 0
        
        # Start health monitor if available
        if hasattr(self, "meta_monitor") and hasattr(self.meta_monitor, "start"):
            try:
                self.meta_monitor.start()
            except:
                pass

        # Main loop
        while self.is_running:
            try:
                # Check pause state
                if self.is_system_paused():
                    time.sleep(tick_sleep)
                    continue

                loop_cycle_count += 1
                self.swarm_state["cycle_count"] = loop_cycle_count
                self.current_action_cycle_id = f"loop_cycle_{time.strftime('%Y-%m-%dT%H-%M-%S')}_{loop_cycle_count}"

                # Process directives (don't let failures stop the loop)
                try:
                    self._process_intent_overrides()
                except Exception as e:
                    print(f"Warning: Intent override processing error: {e}")
                
                try:
                    self._process_dynamic_directives()
                except Exception as e:
                    print(f"Warning: Dynamic directive processing error: {e}")

                # Process each agent
                work_done_this_cycle = False
                
                for agent_name, agent in list(self.agent_instances.items()):
                    print(f"\nProcessing Agent: {agent_name}")
                    
                    try:
                        # Skip paused agents
                        if hasattr(agent, "is_paused") and agent.is_paused():
                            continue

                        # Get current intent
                        current_intent = getattr(agent, "current_intent", "No specific intent")
                        if not current_intent:
                            current_intent = "No specific intent"
                        
                        # Call perform_task with proper context
                        try:
                            raw_result = agent.perform_task(
                                current_intent,
                                cycle_id=self.current_action_cycle_id,
                                context_info={"cycle_id": self.current_action_cycle_id}
                            )
                        except TypeError:
                            # Handle older signatures
                            try:
                                raw_result = agent.perform_task(current_intent)
                            except Exception as e:
                                print(f"Error calling perform_task for {agent_name}: {e}")
                                continue

                        # Normalize the result
                        outcome, reason, report, progress = _normalize_task_result(raw_result)
                        
                        # Log the result
                        if outcome != "idle":
                            print(f"  {agent_name}: {outcome} (progress: {progress:.1%})")
                            if reason:
                                print(f"    Reason: {reason}")
                        
                        # Determine if work was done
                        if outcome in ["completed", "success"]:
                            work_done_this_cycle = True
                        elif progress > 0:
                            work_done_this_cycle = True
                        elif current_intent and "no specific intent" not in current_intent.lower():
                            work_done_this_cycle = True

                        # Optional: Adaptation and memory management
                        if hasattr(agent, "analyze_and_adapt"):
                            try:
                                agent.analyze_and_adapt(self.agent_instances)
                            except:
                                pass
                        
                        # Periodic memory compression (every 5 cycles)
                        if loop_cycle_count % 5 == 0:
                            if hasattr(agent, "trigger_memory_compression"):
                                try:
                                    agent.trigger_memory_compression()
                                except:
                                    pass
                        
                        # Guardian health check (every 5 cycles)
                        if loop_cycle_count % 5 == 0 and hasattr(self, 'guardian'):
                            try:
                                health = self.guardian.health_check()
                                if health['policy_violations'] or health['actions_taken']:
                                    self.external_log_sink.info(f"Guardian: {health}")
                            except Exception as e:
                                self.external_log_sink.error(f"Guardian check failed: {e}")

                    except Exception as e:
                        print(f"Error processing agent {agent_name}: {e}")
                        import traceback
                        traceback.print_exc()

                # Handle idle cycles
                if not work_done_this_cycle:
                    consecutive_idle_cycles += 1
                    print(f"\n[IDLE CYCLE {consecutive_idle_cycles}]")
                    
                    # Inject planning directive after 2 idle cycles
                    if consecutive_idle_cycles >= 2:
                        planner_name = _resolve_planner_name()
                        if planner_name:
                            print(f"Injecting autonomous planning directive to {planner_name}...")
                            try:
                                self.inject_directives([{
                                    "type": "INITIATE_PLANNING_CYCLE",
                                    "planner_agent_name": planner_name,
                                    "high_level_goal": "System idle. Generate autonomous goal and execute plan."
                                }])
                                consecutive_idle_cycles = 0
                            except Exception as e:
                                print(f"Failed to inject planning directive: {e}")
                else:
                    consecutive_idle_cycles = 0

                # Save state periodically
                if loop_cycle_count % 3 == 0:
                    try:
                        self._save_system_state()
                    except Exception as e:
                        print(f"Warning: State save failed: {e}")

                # Sleep before next cycle
                time.sleep(tick_sleep)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Keyboard interrupt received")
                self.is_running = False
                break
            except Exception as e:
                print(f"ERROR in cognitive loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(tick_sleep)

        print("\nCognitive loop terminated.")
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
        persistence_dir="persistence_data",
        swarm_activity_log="logs/swarm_activity.jsonl",
        system_pause_file="system_pause.flag",
        swarm_state_file="swarm_state.json",
        isl_schema_path="isl_schema.yaml",
        chroma_db_path="chroma_db",
        intent_override_prefix="intent_override_"
    )
    
    # --- Register spawn tool and set CVA reference ---
    from tools import spawn_specialized_agent
    import tools as tools_module
    tools_module._cva_instance = catalyst_alpha  # Set global reference
    
    from tool_registry import Tool
    GLOBAL_TOOL_REGISTRY.register_tool(Tool(
        name="spawn_specialized_agent",
        func=spawn_specialized_agent,
        description="Spawn a specialized agent for a specific task",
        category="agent_management"
    ))
    central_logger.info("Spawn tool registered")
        
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
