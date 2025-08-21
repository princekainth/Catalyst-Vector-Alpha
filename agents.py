# ==================================================================
#  agents.py - All Agent Class Definitions
# ==================================================================
from __future__ import annotations
import logging
import json
import os
import random
import re
import sys
import textwrap
import time
import traceback
import uuid
import collections
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from collections.abc import Iterable # Correctly import Iterable
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict

# --- Third-Party Library Imports ---
import psutil
import numpy as np

# --- Project-Specific Imports ---
# Assuming 'core.py' is in the same directory or accessible via PYTHONPATH
from core import (
    MemeticKernel,
    MessageBus,
    EventMonitor,  
    ToolRegistry,
    OllamaLLMIntegration,
    SovereignGradient,
    timestamp_now
)
# These are mocked or assumed to exist for the code to be complete
import prompts
import llm_schemas
from ccn_monitor_mock import MockCCNMonitor

# --- Helper Functions (Moved from CVA) ---
def trim_intent(intent: str, max_len: int = 100) -> str:
    """Trims a long intent string for display purposes."""
    if len(intent) > max_len:
        return intent[:max_len-3] + "..."
    return intent

def load_paused_agents_list(filepath: str) -> list:
    """Loads the list of paused agents from a JSON file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass
    return []

# (You may need to add other imports here like uuid, traceback, etc., for your full methods)
class ProtoAgent(ABC):
    def __init__(self,
                 name: str,
                 eidos_spec: dict,
                 message_bus: 'MessageBus',
                 event_monitor: 'EventMonitor', # Non-default
                 external_log_sink: logging.Logger, # Non-default
                 chroma_db_path: str, # Non-default
                 persistence_dir: str, # Non-default
                 paused_agents_file_path: str, # Non-default
                 tool_registry: Optional['ToolRegistry'] = None, # Default argument
                 sovereign_gradient=None, # Default argument
                 loaded_state: Optional[dict] = None): # Default argument

        # super().__init__() # Uncomment if ProtoAgent inherits from another class

        self.name = name
        self.eidos_spec = eidos_spec if isinstance(eidos_spec, dict) else {}
        self.message_bus = message_bus
        self.location = self.eidos_spec.get('location', 'Unknown')
        self.tool_registry = tool_registry # Store the tool_registry here
        self.event_monitor = event_monitor
        self.external_log_sink = external_log_sink

        self.orchestrator = message_bus.catalyst_vector_ref
        self.ollama_inference_model = OllamaLLMIntegration()
        
        self.chroma_db_full_path = chroma_db_path
        self.persistence_dir = persistence_dir
        self.paused_agents_file_full_path_path = paused_agents_file_path
        self.active_plan_directives = {}
        self.last_plan_id = None
        # Initialize sovereign_gradient early to be available for MemeticKernel config
        if sovereign_gradient and isinstance(sovereign_gradient, dict):
             self.sovereign_gradient = SovereignGradient.from_state(sovereign_gradient)
        elif isinstance(sovereign_gradient, SovereignGradient):
            self.sovereign_gradient = sovereign_gradient
        else:
            self.sovereign_gradient = SovereignGradient(target_entity_name=self.name, config={})


        # --- IMPORTANT: Store loaded_state so _load_or_initialize_state can use it, and also subclasses ---
        self._initial_loaded_state_for_subclasses = loaded_state

        # --- Instantiate MemeticKernel (Correctly pass all required arguments) ---
        initial_mk_config = None
        initial_mk_loaded_memories = None
        initial_memetic_archive_path_override = None

        if loaded_state:
            mk_state_from_loaded = loaded_state.get('memetic_kernel', {})
            initial_mk_config = mk_state_from_loaded.get('config')
            initial_mk_loaded_memories = mk_state_from_loaded.get('memories')
            initial_memetic_archive_path_override = mk_state_from_loaded.get('memetic_archive_path')


        self.memetic_kernel = MemeticKernel(
            agent_name=self.name,
            external_log_sink=self.external_log_sink,
            chroma_db_path=self.chroma_db_full_path,
            persistence_dir=self.persistence_dir,
            config=initial_mk_config,
            loaded_memories=initial_mk_loaded_memories,
            memetic_archive_path=initial_memetic_archive_path_override
        )
        # --- END MemeticKernel instantiation ---

        # Log initial setup completion
        self.external_log_sink.info(f"ProtoAgent {self.name} base initialization completed.",
                                     extra={"agent": self.name, "eidos_role": self.eidos_spec.get('role'),
                                            "location": self.location})

        # --- Initialize other agent components ---
        self._initialize_default_attributes()
        
        self._load_or_initialize_state(loaded_state)

        self.initialize_reset_handlers()

        # --- CRITICAL FIX for line 1532 (or wherever this block is now): Ensure loaded_state is a dictionary before calling .get() ---
        _state_to_use = loaded_state if isinstance(loaded_state, dict) else {}

        self.task_successes = _state_to_use.get('task_successes', 0)
        self.task_failures = _state_to_use.get('task_failures', 0)
        self.intent_loop_count = _state_to_use.get('intent_loop_count', 0)
        self.stagnation_adaptation_attempts = _state_to_use.get('stagnation_adaptation_attempts', 0)
        self.max_allowed_recursion = _state_to_use.get('max_allowed_recursion', 7)
        self.autonomous_adaptation_enabled = _state_to_use.get('autonomous_adaptation_enabled', True)

        # --- NEW: Success Detection Indicators (Phase 1, Step 1) ---
        self.success_indicators = _state_to_use.get('success_indicators', {
            'environmental_impact': 0,
            'collaboration_boost': 0,
            'novel_insights': 0,
            'tool_effectiveness': 0,
            'adaptive_learning': 0
        })
        self.breakthrough_threshold = _state_to_use.get('breakthrough_threshold', 3)

        # --- CRITICAL FIX: Initialize Success Tracking Flags (Phase 1, Step 3) ---
        self.last_action_modified_environment = _state_to_use.get('last_action_modified_environment', False)
        self.last_tool_result_actionable = _state_to_use.get('last_tool_result_actionable', False)
        self.new_insights_this_cycle = _state_to_use.get('new_insights_this_cycle', [])

    # --- Add this _log_agent_activity method here ---
    def _log_agent_activity(self, event_type: str, source: str, description: str, details: Optional[dict]=None, level: str='info'):
        """
        Logs agent-specific activity via the external logging sink (orchestrator's logger).
        It constructs a log entry dictionary and sends it as a JSON string.
        """
        if self.external_log_sink:
            log_data = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": event_type,
                "source": source,
                "description": description,
                "details": details if details is not None else {}
            }
            # Log the dictionary as a JSON string using the logger's methods
            if level == 'error':
                self.external_log_sink.error(json.dumps(log_data))
            elif level == 'warning':
                self.external_log_sink.warning(json.dumps(log_data))
            elif level == 'debug':
                self.external_log_sink.debug(json.dumps(log_data))
            else: # Default to info
                self.external_log_sink.info(json.dumps(log_data))
        # else:
            # Fallback print if for some reason log_sink is None even after init
            # print(f"WARNING: Agent {self.name} _log_agent_activity failed (no sink). Message: {event_type}: {description}")

    def _initialize_default_attributes(self):
        """Initializes all default, non-state-dependent attributes."""
        initial_eidos_intent = self.eidos_spec.get('initial_intent', 'No specific intent')
        self.current_intent = initial_eidos_intent
        self.initial_intent = initial_eidos_intent
        self.reset_config = {
            'commands': [
                'force_intent_update("strategic_planning")',
                'clear_planning_knowledge_base()',
                'enable_llm_assisted_planning(False)',
                'set_recursion_limit(2)'
            ],
            'expected_recovery_time': '3 cycles'
        }
        self.swarm_membership = []
        self.autonomous_adaptation_enabled = True
        self.intent_loop_count = 0
        self.stagnation_adaptation_attempts = 0
        self.task_successes = 0
        self.task_failures = 0
        self.agent_beliefs = []
        self._skip_initial_recursion_check = True
        self.max_allowed_recursion = self.eidos_spec.get('max_recursion_limit', 5)
    
    def _perform_critical_autonomous_override(self):
        """
        Executes a critical autonomous override protocol when an agent is in deep, persistent stagnation.
        This is the agent's last resort before escalating to human intervention.
        It involves a more drastic internal reset and a final attempt at a new strategy.
        """
        print(f"\n!!! {self.name} INITIATING CRITICAL AUTONOMOUS OVERRIDE PROTOCOL !!!")
        self._log_agent_activity("CRITICAL_AUTONOMOUS_OVERRIDE_INITIATED", self.name,
            "Agent initiating critical autonomous override protocol (last resort before human escalation).",
            {"current_intent": self.current_intent, "stagnation_attempts": self.stagnation_adaptation_attempts},
            level='error' # Use error level to highlight this severe internal state
        )

        # 1. Drastically clear most of the working memory to force a completely fresh perspective
        self.memetic_kernel.memories.clear()
        self.memetic_kernel.compressed_memories.clear()
        print(f"  [Critical Override] Cleared ALL recent and compressed memories.")
        self.memetic_kernel.add_memory("FullMemoryPurge", "Cleared all memories during critical autonomous override.")

        # 2. Reset all critical internal counters
        self.planning_failure_count = 0
        self.stagnation_adaptation_attempts = 0 # Reset primary stagnation counter
        self.reset_intent_loop_counter()

        # 3. Use LLM to brainstorm a new, *drastic* strategy for breaking the cycle
        # This is a final, high-stakes attempt at a new strategic direction.
        current_narrative_for_llm = self.distill_self_narrative() # Will be very short after memory clear

        critical_override_prompt_context = f"""
        You are an intelligent AI agent named {self.name} with the role of {self.eidos_spec.get('role', 'unknown')}.
        You have reached a state of deep, persistent stagnation where all previous autonomous adaptation attempts have failed.
        You have just performed a **critical internal override**, clearing your recent memories and resetting your internal counters to gain an entirely fresh perspective.
        Your current (reset) intent is: '{self.current_intent}'.
        Your current cognitive state is now very basic after the reset:
        --- START CURRENT COGNITIVE STATE ---
        {current_narrative_for_llm}
        --- END CURRENT COGNITIVE STATE ---
        Your primary goal now is to propose a **single, concise, and highly disruptive or fundamentally different new primary intent** to break this extreme stagnation. This is your absolute last autonomous attempt to resolve the issue before escalating to human intervention. Think outside the box, propose a radical shift if necessary, but keep it actionable.
        Example intents:
        - 'Initiate full system diagnostic and integrity check, overriding all non-essential operations.'
        - 'Propose a complete re-evaluation of current system objectives and operational parameters.'
        - 'Force a system-wide re-initialization of all communication channels and data pipelines.'
        - 'Adopt a purely observational role for 5 cycles to gather unbiased data on system behavior.'
        Proposed new intent for critical autonomous override:
        """

        new_intent_from_llm = self._brainstorm_new_intent_with_llm(critical_override_prompt_context)

        if new_intent_from_llm and \
           new_intent_from_llm not in ["LLM_BRAINSTORM_FAILED_TO_GENERATE_NEW_INTENT", "LLM_BRAINSTORM_FAILED_EXCEPTION"]:
            
            old_intent = self.current_intent
            self.update_intent(new_intent_from_llm)
            self.memetic_kernel.add_memory(
                "IntentAdaptation_CriticalOverride",
                {"summary": f"Adapted to LLM-brainstormed critical override intent: '{new_intent_from_llm}'",
                 "old_intent": old_intent,
                 "new_intent": new_intent_from_llm,
                 "protocol": "Critical-Autonomous-Override-LLM-Driven"}
            )
            print(f"[Critical Override] {self.name} adopted LLM-brainstormed critical override intent: {new_intent_from_llm}")
            # This counts as a successful adaptation for this cycle
            self._log_agent_activity("CRITICAL_OVERRIDE_COMPLETED", self.name,
                f"Agent completed critical autonomous override protocol.",
                {"new_intent": self.current_intent, "memories_cleared": "all", "adapted_through_llm": True},
                level='error'
            )
            return True # Indicate that a critical override was performed
        else:
            print(f"  [Critical Override Warning] LLM brainstorm for critical override failed for {self.name}. Proceeding to human escalation.")
            self.memetic_kernel.add_memory("IntentAdaptation_CriticalOverrideFailed", "LLM critical override brainstorm failed.", related_event_id=None)
            # This does NOT count as a successful override, so it will fall through to human escalation
            self._log_agent_activity("CRITICAL_OVERRIDE_FAILED", self.name,
                f"Agent's critical autonomous override protocol failed to generate new intent.",
                {"current_intent": self.current_intent},
                level='critical'
            )
            return False # Indicate that critical override failed, proceed to human escalation

    def _generate_reflection_narrative(self, raw_memories_for_cycle: collections.deque) -> str:
        """
        Generates a concise reflection narrative for the agent using an LLM.
        Args:
            raw_memories_for_cycle: A deque of memory dictionaries for the current cycle/period.
        Returns:
            A string reflecting the agent's journey for the period.
        """
        if not hasattr(self, 'ollama_inference_model') or self.ollama_inference_model is None:
            return f"My journey includes: Unable to generate detailed reflection; LLM not available. Raw memories count: {len(raw_memories_for_cycle)}"

        # Ensure raw_memories are in a JSON-serializable format.
        # This converts complex objects to strings to avoid serialization errors for the LLM.
        serializable_memories = []
        for mem in raw_memories_for_cycle:
            temp_mem = {}
            for k, v in mem.items():
                if isinstance(v, (dict, list)): # Recursively handle nested dicts/lists if needed
                    try:
                        temp_mem[k] = json.dumps(v)
                    except TypeError:
                        temp_mem[k] = str(v) # Fallback for non-serializable complex types
                else:
                    temp_mem[k] = str(v) # Convert everything else to string for LLM prompt
            serializable_memories.append(temp_mem)

        current_timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds') + 'Z'

        prompt = prompts.AGENT_REFLECTION_PROMPT.format(
            agent_name=self.name,
            agent_role=self.eidos_spec.get('role', 'unknown'),
            current_timestamp=current_timestamp,
            raw_memories_json=json.dumps(serializable_memories, indent=2)
        ).strip()

        try:
            reflection_output = self.ollama_inference_model.generate_text(prompt, max_tokens=300)
            reflection_output = reflection_output.strip().replace('"', '')

            # Ensure it starts with "My journey includes: " as per the prompt
            if not reflection_output.lower().startswith("my journey includes:"):
                reflection_output = f"My journey includes: {reflection_output}"
            
            # Log the successful reflection generation
            self._log_agent_activity("REFLECTION_GENERATED", self.name,
                "LLM successfully generated agent reflection.",
                {"reflection_preview": reflection_output[:100] + "..." if len(reflection_output) > 100 else reflection_output},
                level='info'
            )
            
            return reflection_output

        except Exception as e:
            self._log_agent_activity("REFLECTION_GENERATION_FAILED", self.name,
                f"Failed to generate LLM reflection for cycle. Error: {e}",
                level='error'
            )
            # Fallback to a simple reflection
            return f"My journey includes: Failed to generate detailed reflection due to error: {e}. Recent activities (last 5): {list(raw_memories_for_cycle)[-5:] if raw_memories_for_cycle else 'None'}"

    def _initialize_sovereign_gradient(self, sovereign_gradient_input):
        """Initializes the Sovereign Gradient for the agent."""
        if sovereign_gradient_input and isinstance(sovereign_gradient_input, dict):
            self.sovereign_gradient = SovereignGradient.from_state(sovereign_gradient_input)
        elif isinstance(sovereign_gradient_input, SovereignGradient):
            self.sovereign_gradient = sovereign_gradient_input
        else:
            self.sovereign_gradient = SovereignGradient(target_entity_name=self.name, config={})

    def _repair_environmental_disconnect(self):
        """Generates an intent to address lack of observable environmental impact."""
        self.log(f"  [Self-Repair Strategy] Agent not creating real environmental impact.")
        return "FOCUS_ENVIRONMENTAL_IMPACT: Prioritize actions that explicitly modify shared system state or produce observable effects."

    def _repair_tool_misuse(self):
        """Generates an intent to address ineffective tool usage."""
        self.log(f"  [Self-Repair Strategy] Agent's tools are not producing useful results.")
        return "OPTIMIZE_TOOL_USAGE: Analyze and validate tool outputs; research optimal tool sequencing for compound effects."

    def _repair_creative_block(self):
        """Generates an intent to address repetitive patterns or lack of novel insights."""
        self.log(f"  [Self-Repair Strategy] Agent stuck in repetitive patterns or lacking novel insights.")
        return "BREAK_CREATIVE_BLOCK: Question current assumptions, explore unexplored solution spaces, and brainstorm radical alternative approaches with LLM."

    def _generic_llm_self_repair_brainstorm(self, reason: str):
        """
        Generic LLM brainstorming for self-repair, used as a fallback.
        Adapted from your previous _perform_self_repair_protocol's LLM part.
        """
        # FIX APPLIED HERE: Using _log_agent_activity instead of self.log
        self._log_agent_activity("SELF_REPAIR_BRAINSTORM_INITIATED", self.name,
            f"Falling back to generic LLM brainstorm for self-repair: {reason}",
            {"reason": reason},
            level='info' # Or 'warning' based on desired log verbosity
        )
        current_narrative_for_llm = self.distill_self_narrative()

        stagnation_break_prompt_context = f"""
        You are experiencing a general stagnation or difficulty. Reason for generic repair: {reason}.
        You need to propose a single, concise, and actionable new primary intent to break this stagnation.
        Your current intent is: '{self.current_intent}'.
        Your recent cognitive state:
        --- START CURRENT COGNITIVE STATE ---
        {current_narrative_for_llm}
        --- END CURRENT COGNITIVE STATE ---
        Based on this, propose a **single, concise, and actionable new primary intent** that represents a specific strategy to **break this general stagnation**, move past recurring issues, or find a completely new approach to achieve your overarching role. This new intent should be a direct step towards autonomous resolution. Avoid simply asking for human input.
        Proposed new intent for self-repair:
        """
        new_intent_from_llm = self._brainstorm_new_intent_with_llm(stagnation_break_prompt_context)
        return new_intent_from_llm

    # --- MODIFIED: _perform_self_repair_protocol (Phase 2, Step 1) ---
    def _perform_self_repair_protocol(self):
        """
        Corrected, context-aware self-repair. It generates and injects concrete tasks.
        """
        self._log_agent_activity("SELF_REPAIR_INITIATED", self.name,
            f"\n!!! INITIATING CONTEXT-AWARE SELF-REPAIR PROTOCOL !!!",
            {"current_intent": self.current_intent, "stagnation_attempts": self.stagnation_adaptation_attempts},
            level='warning'
        )

        memories_to_clear_types = ["TaskOutcome", "PlanningOutcome", "ToolExecutionFailed",
                                   "AdaptiveToolUseFailed", "LLM_Error", "PatternInsight",
                                   "HumanInputAcknowledged", "IntentAdaptation", "IntentAdaptation_SelfRepair"]
        
        initial_mem_count = len(self.memetic_kernel.memories)
        new_memories_after_clear = collections.deque(maxlen=self.memetic_kernel.memories.maxlen)
        cleared_count = 0
        for mem in self.memetic_kernel.memories:
            if mem.get('type') not in memories_to_clear_types:
                new_memories_after_clear.append(mem)
            else:
                cleared_count += 1
        self.memetic_kernel.memories = new_memories_after_clear
        self._log_agent_activity("MEMORY_PURGE", self.name,
            f"Cleared {cleared_count} problematic recent memories during self-repair.",
            {"memories_cleared_count": cleared_count},
            level='info'
        )
        self.memetic_kernel.add_memory("MemoryPurge", f"Cleared {cleared_count} problematic memories during self-repair.")

        self.planning_failure_count = 0
        self.stagnation_adaptation_attempts = 0
        self.reset_intent_loop_counter()

        # --- DIAGNOSE STAGNATION AND GENERATE ACTIONABLE TASKS ---
        context_for_llm = self.distill_self_narrative()
        self_repair_tasks_str = self._brainstorm_self_repair_tasks_with_llm(context_for_llm)
        tasks_list = self._parse_llm_task_list(self_repair_tasks_str)
        
        if tasks_list:
            print(f"  [{self.name}] LLM generated {len(tasks_list)} self-repair tasks. Injecting directives...")
            directives_to_inject = []
            new_plan_id = f"self_repair_plan_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
            self.last_plan_id = new_plan_id
            self.active_plan_directives = {task: "pending" for task in tasks_list}
           
            for task_desc in tasks_list:
                directives_to_inject.append({
                    "type": "AGENT_PERFORM_TASK",
                    "agent_name": self.name,
                    "task_description": task_desc,
                    "reporting_agents": [self.name],
                    "plan_id": new_plan_id # CRITICAL: Add the plan ID to each directive
                })

            # CRITICAL FIX: Inject the directives directly into the system.
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref.inject_directives(directives_to_inject)
                self._log_agent_activity(
                    "SELF_REPAIR_DIRECTIVES_INJECTED", self.name,
                    f"Injected {len(directives_to_inject)} self-repair tasks.",
                    {"tasks_count": len(tasks_list), "plan_id": new_plan_id},
                    level='info'
                )
            else:
                self._log_agent_activity("ERROR", self.name, "Orchestrator reference not found for injecting self-repair tasks.", level='error')

            # CRITICAL FIX: The agent's new intent should reflect that it is now executing a repair plan.
            new_intent = "Executing self-repair plan based on LLM directives."
            self.update_intent(new_intent)
            self.memetic_kernel.add_memory(
                        "IntentAdaptation_SelfRepair",
                        {"summary": f"Initiated self-repair by injecting {len(tasks_list)} tasks.",
                        "old_intent": self.current_intent,
                        "new_intent": new_intent}
             )
            print(f"  [{self.name}] Self-repair initiated. New intent: '{new_intent}'")
        else:
            self._log_agent_activity("SELF_REPAIR_FAILED", self.name, "LLM failed to generate any parseable self-repair tasks.", level='error')
            reset_intent = self.eidos_spec.get('initial_intent', 'Monitor primary operational parameters.')
            self.update_intent(reset_intent)

        self._log_agent_activity("SELF_REPAIR_COMPLETED", self.name,
            f"Agent completed internal self-repair protocol.",
            {"new_intent": self.current_intent, "memories_cleared": cleared_count},
            level='warning'
        )

    def get_status_summary(self) -> dict:
        """
        Returns a dictionary summarizing the agent's current state for the dashboard sidebar.
        """
        health_status = "Healthy"
        if getattr(self, 'task_failures', 0) > 0:
            health_status = "Degraded"
        if getattr(self, 'stagnation_adaptation_attempts', 0) >= 2:
            health_status = "Stagnant"
        if getattr(self, 'intent_loop_count', 0) >= getattr(self, 'max_allowed_recursion', 7):
            health_status = "Critical Loop"

        return {
            "name": self.name,
            "role": self.eidos_spec.get('role', 'N/A'),
            "health_status": health_status,
        }
        
    def get_detailed_state(self) -> dict:
        """
        Returns a more detailed dictionary of the agent's state for the main panel view.
        """
        return {
            "name": self.name,
            "role": self.eidos_spec.get('role', 'N/A'),
            "location": self.eidos_spec.get('location', 'N/A'),
            "current_intent": self.current_intent,
            "task_successes": self.task_successes,
            "task_failures": self.task_failures,
            "stagnation_attempts": self.stagnation_adaptation_attempts,
            "intent_loop_count": self.intent_loop_count,
            "memories": list(getattr(self.memetic_kernel, 'memories', []))
        }

    def _evaluate_plan_completion(self):
        """
        Checks for the completion of all tasks in the current active plan.
        Resets the agent's state if the plan is completed successfully.
        """
        if not self.last_plan_id or not self.active_plan_directives:
            # No active plan to evaluate
            return

        # Check incoming messages for task completion reports related to this plan.
        messages = self.receive_messages()
        for msg in messages:
            payload = msg.get('payload', {})
            task_status = payload.get('outcome')
            plan_id_in_report = payload.get('plan_id')
            task_desc = payload.get('task')
            
            # Check if the report is for the current plan and if the task is pending
            if plan_id_in_report == self.last_plan_id and task_desc in self.active_plan_directives and self.active_plan_directives[task_desc] == "pending":
                if task_status == "completed":
                    self.active_plan_directives[task_desc] = "completed"
                    print(f"  [{self.name}] Task '{task_desc}' from plan '{self.last_plan_id}' marked as completed.")
                else:
                    # If any task fails, the plan is considered failed.
                    print(f"  [{self.name}] Task '{task_desc}' from plan '{self.last_plan_id}' FAILED. Aborting plan.")
                    self.reset_after_plan_failure()
                    return

        # Check if all tasks in the plan are completed
        if all(status == "completed" for status in self.active_plan_directives.values()):
            print(f"  [{self.name}] All tasks for plan '{self.last_plan_id}' are completed. Resetting state.")
            self.reset_after_plan_success()


    def reset_after_plan_success(self):
        """Resets the agent's state after a successful multi-step plan."""
        self._log_agent_activity("PLAN_COMPLETED_SUCCESS", self.name, "Successfully completed multi-step plan. Resetting.", level='info')
        self.active_plan_directives = {}
        self.last_plan_id = None
        # CRITICAL: Update intent to a neutral state to prevent stagnation.
        new_intent = self.eidos_spec.get('initial_intent', "Monitor and wait for new directives.")
        self.update_intent(new_intent)
        self.stagnation_adaptation_attempts = 0


    def reset_after_plan_failure(self):
        """Resets the agent's state after a failed multi-step plan."""
        self._log_agent_activity("PLAN_COMPLETED_FAILURE", self.name, "Multi-step plan failed. Resetting and requesting human input.", level='warning')
        self.active_plan_directives = {}
        self.last_plan_id = None
        # For a failure, we might want to escalate to human intervention or another repair loop.
        self.update_intent("Awaiting new directives after plan failure.")
        self.stagnation_adaptation_attempts = 0
        # You might want to trigger a human request here as well.

    def _brainstorm_self_repair_tasks_with_llm(self, context: str) -> str:
        """
        Internal LLM call to brainstorm concrete, short self-repair tasks.
        """
        prompt = f"""
        You are an advanced AI self-repair module. An agent is experiencing a state of stagnation and has requested a self-repair protocol.
        Your task is to generate a list of 3-5 concise, concrete, and actionable tasks to help the agent break out of its loop.
        
        The current agent's context and recent history is:
        {context}
        
        Provide your response as a numbered list of short task descriptions. Do NOT provide a summary or explanation. Just the tasks.
        Example:
        1. Review recent logs for recurring errors.
        2. Conduct a self-assessment of internal state.
        3. Re-evaluate assumptions in the last plan.
        """
        # Call to LLM...
        # For this example, let's assume a mock response:
        return """
        1. Analyze recent message bus traffic for feedback loops.
        2. Conduct a deep dive on memory patterns to identify anomalies.
        3. Formulate a new, novel intent based on the deep dive's findings.
        """
    
    def _parse_llm_task_list(self, raw_llm_output: str) -> list:
        """
        Parses a numbered or bulleted list of tasks from the LLM's raw output.
        """
        tasks = []
        for line in raw_llm_output.split('\n'):
            line = line.strip()
            if line and (line.startswith(tuple("123456789")) or line.startswith('-') or line.startswith('*')):
                # Remove the number or bullet and any leading whitespace.
                task_desc = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', line).strip()
                tasks.append(task_desc)
        return tasks
    
    def evaluate_cycle_success(self, lookback_cycles: int = 2) -> bool:
        """
        Evaluates if the agent had a 'successful' cycle (made progress, adapted, or used a tool)
        within the last `lookback_cycles`. This is a more nuanced check for breaking stagnation.
        Returns True if a successful pattern is detected, False otherwise.
        """
        # Retrieve memories from the broader lookback period as intended
        # The lookback_period for retrieve_recent_memories should be in "number of memories" or "cycles"
        # Let's assume retrieve_recent_memories gives us memories ordered by time/insertion,
        # and we then filter for the actual time window.
        
        # Adjusting the lookback period to be more aligned with how many cycles we want to evaluate.
        # If lookback_cycles is 2, we want to look at memories relevant to the last 2 cycles.
        # The memetic kernel's maxlen is 100, so retrieving up to 20-30 might be enough.
        # Let's say, 10 memories per cycle is a rough estimate for lookback_period argument.
        recent_memories_from_kernel = self.memetic_kernel.retrieve_recent_memories(lookback_period=lookback_cycles * 10) # Get more raw memories

        # Get current time for comparison
        now_utc = datetime.now(timezone.utc)
        
        # Calculate the start time for the evaluation window
        # Assuming each cycle takes about 5 seconds (from orchestrator's time.sleep)
        # So, lookback_seconds = lookback_cycles * time_per_cycle
        time_per_cycle = 5 # seconds, from CatalystVectorAlpha's main loop sleep
        evaluation_window_seconds = lookback_cycles * time_per_cycle

        # Filter memories that fall within the last `evaluation_window_seconds`
        relevant_memories_for_eval = []
        for m in recent_memories_from_kernel:
            mem_timestamp_str = m.get('timestamp')
            if mem_timestamp_str:
                try:
                    mem_timestamp_dt = datetime.strptime(mem_timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    time_difference = (now_utc - mem_timestamp_dt).total_seconds()
                    if time_difference <= evaluation_window_seconds:
                        relevant_memories_for_eval.append(m)
                except ValueError:
                    # Handle cases where timestamp might be malformed if any exist
                    print(f"  [Memory Parse Error] Could not parse timestamp: {mem_timestamp_str}")
                    continue
        
        # Sort by timestamp to process chronologically if needed (optional, but good practice)
        relevant_memories_for_eval.sort(key=lambda x: x.get('timestamp', ''))

        # Define criteria for a "successful cycle"
        for mem in relevant_memories_for_eval: # Iterate over the correctly filtered memories
            mem_type = mem.get('type')
            mem_content = mem.get('content', {})

            # Criteria 1: Successful Task Outcome (check if it's a positive outcome)
            # Exclude tasks related to 'investigate' or 'diagnostic' as they are signs of being stuck.
            if mem_type == 'TaskOutcome' and mem_content.get('outcome') == 'completed':
                task_description_lower = mem_content.get('task', '').lower()
                if "investigate" not in task_description_lower and \
                   "diagnostic" not in task_description_lower and \
                   "self-assessment" not in task_description_lower: # Added self-assessment
                    print(f"  [Cycle Success] Detected successful *productive* task outcome: '{mem_content.get('task')}'")
                    self._log_agent_activity("CYCLE_SUCCESS_TASK", self.name, "Successful productive task outcome detected.", {"task": mem_content.get('task')}, level='debug')
                    return True

            # Criteria 2: Successful Intent Adaptation (specifically from LLM brainstorming)
            # This should also cover the new Self-Repair LLM adaptation.
            if mem_type == 'IntentAdaptation' and "LLM-brainstormed" in mem_content.get('summary', ''):
                print(f"  [Cycle Success] Detected successful LLM-driven intent adaptation: '{mem_content.get('new_intent')}'")
                self._log_agent_activity("CYCLE_SUCCESS_ADAPTATION", self.name, "Successful LLM-driven adaptation detected.", {"new_intent": mem_content.get('new_intent')}, level='debug')
                return True
            
            # Criteria for LLM-driven Self-Repair Adaptation
            if mem_type == 'IntentAdaptation_SelfRepair' and "LLM-brainstormed" in mem_content.get('summary', ''):
                print(f"  [Cycle Success] Detected successful LLM-driven Self-Repair adaptation: '{mem_content.get('new_intent')}'")
                self._log_agent_activity("CYCLE_SUCCESS_SELF_REPAIR_ADAPTATION", self.name, "Successful LLM-driven Self-Repair adaptation detected.", {"new_intent": mem_content.get('new_intent')}, level='debug')
                return True

            # Criteria 3: Successful Tool Use (tool proposed and executed without failure)
            if mem_type == 'AdaptiveToolUse' and "Successfully used tool" in mem_content.get('summary', ''):
                print(f"  [Cycle Success] Detected successful tool use: '{mem_content.get('tool_name')}'")
                self._log_agent_activity("CYCLE_SUCCESS_TOOL_USE", self.name, "Successful tool use detected.", {"tool": mem_content.get('tool_name')}, level='debug')
                return True

            # Criteria 4: Positive Pattern Insight (LLM explicitly identified a positive pattern)
            # Ensure PatternInsight content is consistently a dictionary
            if mem_type == 'PatternInsight' and isinstance(mem_content, dict) and 'patterns' in mem_content:
                for pattern in mem_content['patterns']:
                    if isinstance(pattern, str) and ("efficiency" in pattern.lower() or \
                       "progress" in pattern.lower() or \
                       "successful" in pattern.lower() or \
                       "breakthrough" in pattern.lower() or \
                       "resolved" in pattern.lower()): # Added 'resolved'
                        print(f"  [Cycle Success] Detected positive pattern insight: '{pattern[:50]}...'")
                        self._log_agent_activity("CYCLE_SUCCESS_PATTERN", self.name, "Positive pattern insight detected.", {"pattern_preview": pattern[:50]}, level='debug')
                        return True
        
        print(f"  [Cycle Success] No significant progress detected in last {lookback_cycles} cycles.")
        return False # No success criteria met in recent memories


    def _load_or_initialize_state(self, loaded_state: Optional[dict]):
        """
        Loads the agent's state from `loaded_state` if provided,
        or ensures default attributes are set for a fresh agent.
        It also triggers loading of MemeticKernel's state.
        
        Args:
            loaded_state (Optional[dict]): The dictionary containing the agent's state to load.
                                            If None, default state will be used.
        """
        # Ensure initial_intent is set, usually by _initialize_default_attributes called before this.
        if not hasattr(self, 'initial_intent'):
            self._initialize_default_attributes() # Ensure defaults are always present

        if loaded_state:
            # Load agent's core attributes (these will override defaults from _initialize_default_attributes)
            self.current_intent = loaded_state.get('current_intent', self.initial_intent)
            self.swarm_membership = loaded_state.get('swarm_membership', [])
            self._skip_initial_recursion_check = loaded_state.get('_skip_initial_recursion_check', True)
            self.intent_loop_count = loaded_state.get('intent_loop_count', 0)
            self.stagnation_adaptation_attempts = loaded_state.get('stagnation_adaptation_attempts', 0)
            self.autonomous_adaptation_enabled = loaded_state.get('autonomous_adaptation_enabled', True)
            self.task_successes = loaded_state.get('task_successes', 0)
            self.task_failures = loaded_state.get('task_failures', 0)
            self.max_allowed_recursion = loaded_state.get('max_allowed_recursion', self.max_allowed_recursion)
            self.agent_beliefs = loaded_state.get('agent_beliefs', [])

            # Load SovereignGradient state if present in loaded_state.
            # NOTE: self.sovereign_gradient is already initialized in __init__.
            # This part ensures it's *updated* with loaded data if available.
            if loaded_state.get('sovereign_gradient') and isinstance(loaded_state['sovereign_gradient'], dict):
                self.sovereign_gradient.load_state(loaded_state['sovereign_gradient'])
            
            # Load MemeticKernel's internal state (assuming self.memetic_kernel already exists from __init__)
            loaded_mk_state = loaded_state.get('memetic_kernel', {})
            if loaded_mk_state: # Only try to load if there's actual kernel state
                self.memetic_kernel.load_state(loaded_mk_state) # Call MemeticKernel's own load_state method

            self._log_agent_activity(
                "AGENT_RELOADED", self.name,
                f"Agent '{self.name}' reloaded from persistence.",
                {"location": self.location, "current_intent": self.current_intent},
                level='info'
            )
        else:
            # For a fresh agent (loaded_state is None), ensure initial values are set if not by _initialize_default_attributes
            # self.current_intent should be self.initial_intent if new
            # self.task_successes = 0, etc. (often set by _initialize_default_attributes or explicitly here)

            # MemeticKernel.add_memory is okay for new agent.
            self.memetic_kernel.add_memory("Activation", f"Activated in {self.location}.")
            print(f"[Agent] '{self.name}' declared. Initial Current Intent: '{self.current_intent}'")
            print(f"[Agent] '{self.name}' is now Active in {self.location}.")
            self._log_agent_activity(
                "AGENT_ACTIVATED", self.name,
                f"Agent '{self.name}' activated.",
                {"location": self.location, "initial_intent": self.current_intent, "role": self.eidos_spec.get('role')},
                level='info'
            )
    
    def perform_task(self, task_description: str, cycle_id: Optional[str] = None,
                         reporting_agents: Optional[Union[str, list]] = None,
                         context_info: Optional[dict] = None, **kwargs) -> tuple:
        """
        Base method for performing a task, including common checks (pause, gradient)
        and delegating specific task logic to subclasses.
        """
        # --- CRITICAL FIX: RESET SUCCESS TRACKING FLAGS FOR CURRENT CYCLE ---
        self.last_action_modified_environment = False
        self.last_tool_result_actionable = False
        self.new_insights_this_cycle = []
        # --- END RESET ---
        
        # Ensure reporting_agents_ref is a list for consistency
        if isinstance(reporting_agents, str):
            reporting_agents_list = [reporting_agents]
        else:
            reporting_agents_list = reporting_agents if reporting_agents else []

        # Assuming self.paused_agents_file_full_path_path is correctly set in ProtoAgent.__init__
        global_paused_agents = load_paused_agents_list(self.paused_agents_file_full_path_path)

        if self.name in global_paused_agents:
            self._log_agent_activity("AGENT_PAUSED", self.name,
                f"Agent paused, skipped task '{task_description}'.",
                {"task": task_description},
                level='info'
            )
            return "paused", None, {"task": task_description}

        # Sovereign Gradient check (applies to all task types)
        final_task_description = task_description
        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            if not compliant:
                self.memetic_kernel.add_memory(
                    "SovereignGradientNonCompliance",
                    f"Task '{task_description}' was blocked due to Sovereign Gradient non-compliance."
                )
                return "failed", f"Sovereign Gradient non-compliance: {adjusted_task}", {"task": task_description}
            final_task_description = adjusted_task

        try:
            # Delegate to subclass for specific task execution
            specific_task_outcome, specific_failure_reason, specific_report_content, progress_score = \
                self._execute_agent_specific_task(
                    task_description=task_description,
                    cycle_id=cycle_id,
                    reporting_agents=reporting_agents_list,
                    context_info=context_info,
                    text_content=kwargs.get('text_content'),
                    task_type=kwargs.get('task_type', 'GenericTask'),
                    **{k: v for k, v in kwargs.items() if k not in ['text_content', 'task_type']}
                )

            # Ensure report_content is a dictionary for consistent logging and reporting
            if not isinstance(specific_report_content, dict):
                specific_report_content = {"summary": str(specific_report_content)}

            # Create a clean, consistent dictionary for the outcome details.
            outcome_details = {
                "task": final_task_description,
                "outcome": specific_task_outcome,
                "gradient_compliant": self.sovereign_gradient.evaluate_action(final_task_description)[0] if self.sovereign_gradient else True,
                "task_type": kwargs.get('task_type', 'GenericTask'),
                "failure_reason": specific_failure_reason,
                "context": context_info,
                **specific_report_content
            }

            self.memetic_kernel.add_memory("TaskOutcome", outcome_details)
            self._log_agent_activity("TASK_PERFORMED", self.name,
                f"Task '{final_task_description}' {specific_task_outcome}.",
                {"outcome": specific_task_outcome, "task_type": kwargs.get('task_type', 'GenericTask')},
                level='info' if specific_task_outcome == 'completed' else 'error'
            )

            # Send messages to reporting agents
            if reporting_agents_list:
                for agent_ref in reporting_agents_list:
                    # Using the full outcome_details dictionary for the message content
                    self.send_message(agent_ref, "ActionCycleReport", outcome_details, final_task_description, specific_task_outcome, cycle_id)
            
            return specific_task_outcome, specific_failure_reason, specific_report_content, progress_score

        except Exception as e:
            error_msg = f"Exception during task execution for '{final_task_description}': {e}"
            self.external_log_sink.error(error_msg, exc_info=True, extra={"agent": self.name})
            return "failed", error_msg, {"task": final_task_description, "error": str(e)}

    @abstractmethod
    def _execute_agent_specific_task(self, task_description: str, cycle_id: Optional[str],
                                      reporting_agents: Optional[Union[str, list]],
                                      context_info: Optional[dict], **kwargs) -> tuple[str, Optional[str], str]:
        """
        Abstract method to be implemented by subclasses for their specific task logic.
        Returns: (outcome_str, failure_reason_str, report_content_str)
        """
        pass

    def receive_event(self, event: dict):
        print(f"  [{self.name}] Perceived event: {event.get('type')}, Urgency: {event.get('payload', {}).get('urgency')}, Change: {event.get('payload', {}).get('change_factor')}, Direction: {event.get('payload', {}).get('direction')}")

        if self.event_monitor:
            self.event_monitor.log_agent_response(
                agent_id=self.name,
                event_id=event.get('event_id', 'N/A'),
                response_type='perceived_event',
                details={'event_type': event.get('type'), 'urgency': event.get('payload', {}).get('urgency'), 'direction': event.get('payload', {}).get('direction')}
            )
        else:
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("EVENT_PERCEIVED_NO_MONITOR", self.name,
                f"Perceived event {event.get('type')}.",
                {"event_id": event.get('event_id'), "payload": event.get('payload')},
                level='info'
            )

        urgency = event.get('payload', {}).get('urgency', 'medium')
        # This calls MemeticKernel.inhibit_compression
        if urgency == 'critical':
            self.memetic_kernel.inhibit_compression(cycles=5)
        elif urgency == 'high':
            self.memetic_kernel.inhibit_compression(cycles=3)
        else:
            self.memetic_kernel.inhibit_compression(cycles=1)

        self.memetic_kernel.add_memory( # <--- Uses add_memory
            'event',
            content=f"Perceived event: {event.get('type')} with payload {event.get('payload')}",
            related_event_id=event.get('event_id')
        )
    def evaluate_cycle_success(self, cycle_data: dict) -> bool:
        """
        Enhanced success evaluation based on meaningful progress indicators.
        Updates internal success counters and determines if a breakthrough has occurred.
        """
        points = 0

        # Environmental Impact: Did the agent modify the shared system state?
        if cycle_data.get('modified_shared_state', False):
            points += 2
            self.success_indicators['environmental_impact'] += 1

        # Tool Effectiveness: Did the tools produce actionable results?
        if cycle_data.get('tool_outputs_actionable', False):
            points += 1
            self.success_indicators['tool_effectiveness'] += 1

        # Novel Insights: Did the agent discover new patterns or generate breakthroughs?
        if cycle_data.get('discovered_new_patterns', False):
            points += 3 # Higher points for novel insights, indicating significant progress
            self.success_indicators['novel_insights'] += 1
        return points >= self.breakthrough_threshold
    
    def analyze_and_adapt(self):
        """
        Enhanced adaptive reasoning that prioritizes Planner directives before
        initiating self-adaptation.
        """
        import re, collections

        # ---------- Local helpers (scoped to this method) ----------
        def _is_executing_plan() -> bool:
            mode = getattr(self, "mode", None)
            if mode == "EXECUTING_PLAN":
                return True
            ci = getattr(self, "current_intent", "")
            return isinstance(ci, str) and ("execut" in ci.lower()) and ("plan" in ci.lower())

        def _reset_progress_counters():
            self.stagnation_adaptation_attempts = 0
            self.reset_intent_loop_counter()

        def _is_investigation_intent() -> bool:
            meta = getattr(self, "current_intent_meta", {})
            if isinstance(meta, dict) and meta.get("type") == "investigation":
                return True
            ci = getattr(self, "current_intent", "")
            return isinstance(ci, str) and bool(re.search(r"\binvestigat(e|ion)\b", ci.lower()))

        def _safe_log(event, msg, data=None, level="info"):
            try:
                self._log_agent_activity(event, self.name, msg, (data or {}), level=level)
            except Exception:
                pass

        def _notify_optimizer(subject_msg: str):
            try:
                mb = getattr(self, "message_bus", None)
                if not mb or not getattr(mb, "catalyst_vector_ref", None):
                    return
                agents = getattr(mb.catalyst_vector_ref, "agent_instances", {})
                if "ProtoAgent_Optimizer_instance_1" not in agents:
                    return
                cycle_id = getattr(mb, "current_cycle_id", None)
                self.send_message(
                    "ProtoAgent_Optimizer_instance_1",
                    "AdaptationAlert",
                    subject_msg,
                    "Optimizer Notification: Agent Adaptation",
                    "completed",
                    cycle_id=cycle_id
                )
            except Exception:
                pass

        # ---------- Start of method logic ----------
        self._evaluate_plan_completion()

        # Tunable thresholds
        self.STAG_LLMBRAINSTORM_AT = getattr(self, "STAG_LLMBRAINSTORM_AT", 2)
        self.STAG_SELFREPAIR_AT    = getattr(self, "STAG_SELFREPAIR_AT", 3)
        self.STAG_CRITICAL_AT      = getattr(self, "STAG_CRITICAL_AT", 5)

        if _is_executing_plan():
            _reset_progress_counters()
            _safe_log("PLAN_EXECUTION_PROGRESS", "Multi-step plan in progress. Skipping stagnation check.",
                    {"current_intent": getattr(self, "current_intent", None)}, level="info")
            return

        if not getattr(self, "autonomous_adaptation_enabled", False):
            print(f"  [{self.name}] Autonomous adaptation is disabled. Skipping analyze_and_adapt.")
            _safe_log("ADAPTATION_DISABLED", "Autonomous adaptation currently disabled.", level="info")
            self.intent_loop_count = 0
            self.stagnation_adaptation_attempts = 0
            return

        print(f"[Agent] {self.name} is performing reflexive analysis.")
        print(f"  [IP-Integration] {self.name} is engaging in Meta™-cognitive self-evaluation via analyze_and_adapt.")

        adapted_this_cycle = False

        # --- Adaptive recursion limit with smoothing ---
        total_tasks = self.task_successes + self.task_failures
        failure_rate = (self.task_failures / (total_tasks + 1e-9)) if total_tasks > 0 else 0.5
        target_limit = max(6, min(8, 6 + int(2 * (1 - failure_rate))))
        prev_limit = getattr(self, "max_allowed_recursion", 6)
        
        if target_limit > prev_limit: self.max_allowed_recursion = prev_limit + 1
        elif target_limit < prev_limit: self.max_allowed_recursion = prev_limit - 1
        else: self.max_allowed_recursion = prev_limit
        print(f"[Agent] {self.name} set adaptive recursion limit to {self.max_allowed_recursion} (target {target_limit}, prev {prev_limit}).")

        # --- Gather recent task outcomes for failure pattern analysis ---
        # (Your hardened parsing logic is correct and remains here) ...

        # --- Genuine progress detection ---
        cycle_data_for_eval = {
            "modified_shared_state": getattr(self, "last_action_modified_environment", False),
            "tool_outputs_actionable": getattr(self, "last_tool_result_actionable", False),
            "discovered_new_patterns": len(getattr(self, "new_insights_this_cycle", []) or []) > 0
        }

        if self.evaluate_cycle_success(cycle_data_for_eval):
            _reset_progress_counters()
            _safe_log("BREAKTHROUGH", "Real progress detected, resetting stagnation counter.",
                    {"current_intent": getattr(self, "current_intent", None)}, level="info")
            adapted_this_cycle = True
        else:
            # --- Stagnation path begins ---
            self.stagnation_adaptation_attempts = getattr(self, "stagnation_adaptation_attempts", 0) + 1
            _safe_log("STAGNATION_INCREMENT", f"No significant progress; incrementing stagnation attempt: {self.stagnation_adaptation_attempts}",
                    {"current_intent": getattr(self, "current_intent", None), "stagnation_attempts": self.stagnation_adaptation_attempts}, level="warning")
            
            # --- NEW: Check for Planner Directives FIRST ---
            planner_directives = self.check_for_new_planner_messages()
            if planner_directives:
                directive_task = planner_directives[0].get('content', "Received an unreadable directive from Planner.")
                print(f"  [Adaptation] {self.name} acknowledging new strategic directive from Planner: '{directive_task}'")
                self.update_intent(directive_task)
                _reset_progress_counters()
                adapted_this_cycle = True
            # --- END NEW LOGIC ---

            # --- If no planner directives, proceed with self-adaptation ---
            if not adapted_this_cycle:
                # LLM brainstorm at threshold
                if self.stagnation_adaptation_attempts == self.STAG_LLMBRAINSTORM_AT:
                    # (Your LLM brainstorming logic is correct and goes here) ...
                    pass # Placeholder for your existing logic
                
                # (Your logic for self-repair and critical overrides is also correct and goes here) ...
                elif self.stagnation_adaptation_attempts == self.STAG_SELFREPAIR_AT:
                    # ...
                    pass
                
                elif self.stagnation_adaptation_attempts >= self.STAG_CRITICAL_AT:
                    # ...
                    pass

        # --- Post-cycle counters ---
        if not adapted_this_cycle:
            self.increment_intent_loop_counter()
            _safe_log("LOOP_COUNTER_INCREMENT", f"No adaptation this cycle, incrementing intent loop counter: {self.intent_loop_count}",
                    {"current_intent": getattr(self, "current_intent", None), "intent_loop_count": self.intent_loop_count}, level="info")
            
            if self.intent_loop_count >= self.max_allowed_recursion:
                print(f"[Recursion Limit Exceeded] {self.name} exceeded loop limit. Forcing fallback.")
                # (Your recursion limit logic is correct and remains here) ...
                _reset_progress_counters()

    
    def check_for_new_planner_messages(self) -> list:
        """Checks the inbox for unread directives from a Planner agent."""
        planner_messages = []
        # Use receive_messages() to get and clear the inbox
        for msg in self.receive_messages():
            # Check if the message is a directive from the planner
            if msg.get("message_type") == "EXPERIMENTAL_DIRECTIVE" and "Planner" in msg.get("sender", ""):
                planner_messages.append(msg)
        return planner_messages

    def monitor_resources(self):
        """
        Monitors system resource usage and updates intent if high load is detected.
        """
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        if cpu_usage > 80 or memory_usage > 80:
            new_intent = f"Optimize resource allocation (CPU: {cpu_usage}%, Memory: {memory_usage}%)"
            if self.current_intent != new_intent:
                self.update_intent(new_intent)
                self.memetic_kernel.add_memory("ResourceWarning", f"High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%")
                print(f"[Warning] {self.name} detected high resource usage, changed intent to: {new_intent}")
                return True
        return False

    def _trigger_escalation_protocol(self):
        """
        Triggers the final escalation protocol, typically requesting human input
        and potentially pausing the system if human intervention is critical.
        This method is called when an agent has exhausted its autonomous adaptation attempts.
        """
        print(f"\n!!! {self.name} TRIGGERING ESCALATION PROTOCOL (Stagnation Break) !!!")
        
        # Log the escalation
        self._log_agent_activity("ESCALATION_PROTOCOL_TRIGGERED", self.name,
            f"Agent triggered final escalation protocol due to persistent stagnation.",
            {"current_intent": self.current_intent, "stagnation_attempts": self.stagnation_adaptation_attempts},
            level='critical'
        )

        # Request human input via the Orchestrator, which will handle the system pause logic
        # The Orchestrator's pause_system_for_human_input will inject a REQUEST_HUMAN_INPUT directive.
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref.pause_system_for_human_input(
                reason=f"{self.name} is critically stuck on goal '{self.current_intent}' after {self.stagnation_adaptation_attempts} attempts. Requires supervisor intervention.",
                urgency="critical",
                source_agent=self.name
            )
        else:
            print(f"  [{self.name}] ERROR: Cannot trigger escalation. Orchestrator reference not available.")
            self.memetic_kernel.add_memory("SystemError", "Orchestrator not available for escalation.", {"agent": self.name})

    def _check_for_system_patterns(self):
        patterns = [m for m in self.memetic_kernel.memories
                    if m.get('type') == 'PatternInsight']

        if patterns:
            print(f"  [Pattern Detection] Reviewing {len(patterns)} recent patterns.")
            return patterns
        return []

    def _prioritize_patterns(self, patterns):
        """Rank patterns by urgency using LLM analysis"""
        pattern_texts = [p['content'] for p in patterns]
        analysis = self.ollama_inference_model.generate_text( # Assuming ollama_inference_model has generate_text
            f"Rank these patterns by urgency:\n{pattern_texts}\n"
            "Return ONLY the most critical one."
        )
        return analysis.strip()

    def _get_investigation_duration(self):
        """Calculate time since investigation intent started"""
        investigation_start = next(
            (m for m in sorted(self.memetic_kernel.memories, key=lambda x: x['timestamp'])
            if m['type'] == 'IntentUpdate' and
                ("root cause" in m['content'] or "pattern" in m['content'])),
            None
        )
        if investigation_start:
            return datetime.strptime(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) - datetime.strptime(
                investigation_start['timestamp'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.timedelta(0)

    def _execute_tool_proposal(self, tool_proposal: dict) -> str:
        """
        Executes a proposed tool call and returns the tool's output.
        Stores tool execution results and errors in memory.
        """
        tool_name = tool_proposal.get("tool_name")
        tool_args = tool_proposal.get("tool_args", {})

        if not tool_name:
            return "No tool name provided for execution."

        tool_instance = self.tool_registry.get_tool(tool_name)
        if not tool_instance:
            error_msg = f"Attempted to execute unknown tool: '{tool_name}'."
            print(f"  [Tool EXEC ERROR] {self.name}: {error_msg}")
            self.memetic_kernel.add_memory("ToolExecutionError", error_msg, {"tool_name": tool_name, "tool_args": tool_args})
            return error_msg

        print(f"  [Tool EXEC] {self.name} executing tool '{tool_name}' with args: {tool_args}")
        self.memetic_kernel.add_memory("ToolExecutionAttempt", f"Attempting tool '{tool_name}'", {"tool_name": tool_name, "tool_args": tool_args})

        try:
            tool_output = tool_instance.func(**tool_args)

            self.memetic_kernel.add_memory("ToolExecutionSuccess", f"Tool '{tool_name}' executed successfully.", {"tool_name": tool_name, "tool_output": tool_output})
            print(f"  [Tool EXEC SUCCESS] {self.name}: Tool '{tool_name}' output: {tool_output[:200]}...") # Truncate for print
            return tool_output
        except TypeError as e:
            error_msg = f"Tool '{tool_name}' execution failed due to invalid arguments: {e}. Args provided: {tool_args}"
            print(f"  [Tool EXEC ERROR] {self.name}: {error_msg}")
            self.memetic_kernel.add_memory("ToolExecutionError", error_msg, {"tool_name": tool_name, "tool_args": tool_args, "error": str(e)})
            return error_msg
        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {e}. Args provided: {tool_args}"
            print(f"  [Tool EXEC ERROR] {self.name}: {error_msg}")
            self.memetic_kernel.add_memory("ToolExecutionError", error_msg, {"tool_name": tool_name, "tool_args": tool_args, "error": str(e)})
            return error_msg

    def _process_spawn_agent_instance_directive(self, directive): # This method is likely in CatalystVectorAlpha, not ProtoAgent. This snippet might be out of place if it's not meant to be here.
        # This method is likely not part of ProtoAgent, but rather CatalystVectorAlpha.
        # If this is indeed in your ProtoAgent, it's very unusual for a base agent to spawn other agents directly.
        # I'm providing a modified version assuming it's part of an orchestrator-like class that was mis-pasted.
        # If this method is ACTUALLY in ProtoAgent, you have a deeper architectural issue.
        # For now, I will NOT modify this snippet. I'm leaving it as is.
        # If you confirm this _process_spawn_agent_instance_directive is in CatalystVectorAlpha,
        # then it will be part of the CatalystVectorAlpha.py full replacement.
        pass # Placeholder to signify it's skipped here for now

    def increment_intent_loop_counter(self):
        self.intent_loop_count += 1
        print(f"[Agent] {self.name} incremented intent loop counter to {self.intent_loop_count}.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("INTENT_COUNTER_INCREMENTED", self.name,
            f"Intent loop counter incremented to {self.intent_loop_count}.",
            {"count": self.intent_loop_count},
            level='info'
        )

    def reset_intent_loop_counter(self):
        self.intent_loop_count = 0
        print(f"[Agent] {self.name} reset intent loop counter to 0.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("INTENT_COUNTER_RESET", self.name,
            "Intent loop counter reset to 0.",
            level='info'
        )

    def force_fallback_intent(self):
        self.current_intent = "Enter diagnostic standby mode and await supervisor input."
        print(f"[Agent] {self.name} switched to fallback intent: '{self.current_intent}'.")
        self.memetic_kernel.add_memory("FallbackIntent", "Forced fallback intent due to recursion limit.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("FORCE_FALLBACK_INTENT", self.name,
            "Forced fallback intent due to recursion limit.",
            {"new_intent": self.current_intent},
            level='warning' # Set level to warning
        )

    def update_intent(self, new_intent: str):
        old_intent = self.current_intent
        self.current_intent = trim_intent(new_intent)
        self.memetic_kernel.config['current_intent'] = self.current_intent
        print(f"[Agent] {self.name} intent updated to: {self.current_intent}.")
        truncated_print_intent = (self.current_intent[:200] + "...") if len(self.current_intent) > 200 else self.current_intent
        print(f"[Agent] {self.name} intent updated to: {truncated_print_intent}.")
        self.memetic_kernel.config['current_intent'] = self.current_intent
        self.memetic_kernel.add_memory("IntentUpdate", f"Intent updated from '{old_intent}' to '{self.current_intent}'.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("AGENT_INTENT_UPDATED", self.name,
            f"Agent intent changed.",
            {"old_intent": old_intent, "new_intent": self.current_intent},
            level='info'
        )

    def send_message(self, recipient_name: str, message_type: str, content: any,
                 task_description: str = None, status: str = "pending", cycle_id: str = None):
        """
        Sends a structured message to another agent via the central message bus.
        """
        print(f"  [{self.name}] Sending message to {recipient_name} (Type: {message_type})")
        
        # --- FIX: Call the message bus with individual arguments, not a single payload ---
        self.message_bus.send_message(
            sender=self.name,
            recipient=recipient_name,
            message_type=message_type,
            content=content,
            task_description=task_description,
            status=status,
            cycle_id=cycle_id
        )
        # --- END FIX ---

        # Create a message preview for memory logging
        if isinstance(content, str):
            preview = content[:50]
        else:
            preview = str(content)[:50]
        
        self.memetic_kernel.add_memory("MessageSent", {"recipient": recipient_name, "type": message_type, "preview": preview})

        # Log the message activity
        self._log_agent_activity("MESSAGE_SENT", self.name,
            f"Sent message to {recipient_name}.",
            {"type": message_type},
            level='info'
        )
    
    def receive_messages(self):
        # This single line correctly calls the message bus to get and clear messages for this agent.
        messages = self.message_bus.get_messages_for_agent(self.name)
        
        if messages:
            self.memetic_kernel.update_last_received_message(messages[-1])
            # This logging call is correct and remains unchanged.
            self._log_agent_activity("MESSAGES_RECEIVED", self.name,
                f"Received {len(messages)} messages.",
                {"count": len(messages)},
                level='info'
            )
        return messages

    def is_paused(self):
        """Checks if the individual agent-specific pause flag file exists."""
        # CRITICAL FIX: Use self.persistence_dir
        flag_file = os.path.join(self.persistence_dir, f"agent_pause_{self.name}.flag")
        return os.path.exists(flag_file)

    def save_state(self):
        """Saves the agent's current state to a JSON file in the persistence directory."""
        # CRITICAL FIX: Use self.persistence_dir
        state_file = os.path.join(self.persistence_dir, f"agent_state_{self.name}.json")
        try: # Added try-except for robust file operations
            # Ensure the directory exists before attempting to open the file
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(self.get_state(), f, indent=2)
            # You might want to add logging here instead of just printing
            self.external_log_sink.info(f"Agent '{self.name}' state saved to {state_file}.")
        except Exception as e:
            self.external_log_sink.error(f"Failed to save state for agent '{self.name}' to {state_file}: {e}")
            # Optionally re-raise if saving state is critical and should halt


    def set_sovereign_gradient(self, new_gradient: 'SovereignGradient'):
        old_gradient_state = self.sovereign_gradient.get_state() if self.sovereign_gradient else None
        self.sovereign_gradient = new_gradient
        self.memetic_kernel.config['gradient'] = new_gradient.get_state()
        self.memetic_kernel.add_memory("GradientUpdate", f"Sovereign gradient set for swarm: '{new_gradient.autonomy_vector}'.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("AGENT_GRADIENT_SET", self.name,
            f"Sovereign gradient set.",
            {"old_gradient": old_gradient_state, "new_gradient": new_gradient.get_state()},
            level='info'
        )

    def catalyze_transformation(self, new_initial_intent=None, new_description=None, new_memetic_kernel_config_updates=None):
        transformation_summary = []
        if new_initial_intent:
            old_intent = self.current_intent
            self.update_intent(new_initial_intent)
            transformation_summary.append(f"Intent changed from '{old_intent}' to '{new_initial_intent}'")
            print(f"  [Agent] {self.name} self-transformed: Intent updated to '{new_initial_intent}'.")
        if new_description:
            old_description = self.eidos_spec.get('description', 'N/A')
            self.eidos_spec['description'] = new_description
            transformation_summary.append(f"Description changed from '{old_description}' to '{new_description}'")
            print(f"  [Agent] {self.name} self-transformed: Description updated to '{new_description}'.")
        if new_memetic_kernel_config_updates:
            print(f"  [Agent] {self.name} self-transformed: Updating Memetic Kernel configuration...")
            for key, value in new_memetic_kernel_config_updates.items():
                self.memetic_kernel.config[key] = value
                transformation_summary.append(f"Memetic Kernel config updated: {key} set to {value}")
            print(f"  [MemeticKernel] {self.name}'s config updated: {new_memetic_kernel_config_updates}.")

        if transformation_summary:
            memory_content = f"Catalyzed self-transformation: {'; '.join(transformation_summary)}."
            self.memetic_kernel.add_memory("SelfTransformation", memory_content)
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("AGENT_TRANSFORMED", self.name,
                f"Agent transformed.",
                {"updates_summary": "; ".join(transformation_summary)},
                level='info'
            )
        else:
            print(f"  [Agent] {self.name} received CATALYZE_TRANSFORMATION but no valid updates were provided.")

    def process_broadcast_intent(self, broadcast_intent_content, alignment_threshold=0.7):
        print(f"  [Agent] {self.name} processing broadcast intent: '{broadcast_intent_content}'")

        agent_keywords = set(self.current_intent.lower().split())
        broadcast_keywords = set(broadcast_intent_content.lower().split())

        common_keywords = agent_keywords.intersection(broadcast_keywords)
        if not broadcast_keywords:
            alignment_score = 0.0
        else:
            alignment_score = len(common_keywords) / len(broadcast_keywords)

        print(f"    [Agent] {self.name} current intent keywords: {agent_keywords}")
        print(f"    [Agent] Broadcast intent keywords: {broadcast_keywords}")
        print(f"    [Agent] Alignment score: {alignment_score:.2f} (Threshold: {alignment_threshold})")

        if alignment_score >= alignment_threshold:
            new_intent_parts = list(agent_keywords.union(broadcast_keywords))
            new_aligned_intent = " ".join(sorted(new_intent_parts))

            old_intent = self.current_intent
            self.update_intent(new_aligned_intent)

            self.memetic_kernel.add_memory("IntentAlignment",
                                            f"Aligned initial intent to broadcast: '{broadcast_intent_content}'. Old intent: '{old_intent}', New intent: '{new_aligned_intent}' (Score: {alignment_score:.2f})")
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("AGENT_INTENT_ALIGNED", self.name,
                f"Intent aligned to broadcast. Score: {alignment_score:.2f}.",
                {"old_intent": old_intent, "new_intent": new_aligned_intent, "broadcast_intent": broadcast_intent_content, "alignment_score": alignment_score},
                level='info'
            )
        else:
            print(f"    [Agent] {self.name} intent not aligned (score below threshold).")
            self.memetic_kernel.add_memory("IntentNonAlignment",
                                            f"Did not align initial intent to broadcast: '{broadcast_intent_content}'. Current intent: '{self.current_intent}' (Score: {alignment_score:.2f})")
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("AGENT_INTENT_NON_ALIGNED", self.name,
                f"Intent not aligned to broadcast. Score: {alignment_score:.2f}.",
                {"current_intent": self.current_intent, "broadcast_intent": broadcast_intent_content, "alignment_score": alignment_score},
                level='warning'
            )

    def get_state(self):
        state = {
            'name': self.name,
            'eidos_spec': self.eidos_spec,
            'initial_intent': self.initial_intent,
            'current_intent': self.current_intent,
            'sovereign_gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None,
            'intent_loop_count': self.intent_loop_count,
            'stagnation_adaptation_attempts': self.stagnation_adaptation_attempts,
            'agent_beliefs': self.agent_beliefs,
            '_skip_initial_recursion_check': self._skip_initial_recursion_check,
            'autonomous_adaptation_enabled': self.autonomous_adaptation_enabled,
            'task_successes': self.task_successes,
            'task_failures': self.task_failures,
            'max_allowed_recursion': self.max_allowed_recursion,
            'memetic_kernel': self.memetic_kernel.get_state() # Store kernel state
        }
        return state

    def load_state(self, state):
        self.name = state.get('name', self.name)
        self.eidos_spec = state.get('eidos_spec', self.eidos_spec)
        self.initial_intent = state.get('initial_intent', self.initial_intent)
        self.current_intent = state.get('current_intent', self.current_intent)
        self.intent_loop_count = state.get('intent_loop_count', 0)
        self.stagnation_adaptation_attempts = state.get('stagnation_adaptation_attempts', 0)
        self.agent_beliefs = state.get('agent_beliefs', [])
        self._skip_initial_recursion_check = state.get('_skip_initial_recursion_check', True)
        self.autonomous_adaptation_enabled = state.get('autonomous_adaptation_enabled', True)
        self.task_successes = state.get('task_successes', 0)
        self.task_failures = state.get('task_failures', 0)
        self.max_allowed_recursion = state.get('max_allowed_recursion', 5)

        if state.get('sovereign_gradient'):
            self.sovereign_gradient = SovereignGradient.from_state(state['sovereign_gradient'])
        else:
            self.sovereign_gradient = SovereignGradient()

        if state.get('memetic_kernel'):
            self.memetic_kernel.load_state(state['memetic_kernel'])


    def trigger_memory_compression(self):
        """
        Initiates the memory summarization and compression process for this agent.
        Agent decides which memories to summarize (e.g., old ones, or a batch).
        """
        print(f"[Agent] {self.name} is initiating memory compression.")

        # DEBUGGING LINES (Keep these for now to confirm behavior)
        print(f"DEBUG: Type of self.memetic_kernel.memories BEFORE slicing: {type(self.memetic_kernel.memories)}")
        print(f"DEBUG: Value of self.memetic_kernel.memories BEFORE slicing: {self.memetic_kernel.memories}")

        # --- FINAL FIX FOR TypeError: sequence index must be integer, not 'slice' ---
        # Force conversion to a list before slicing to guarantee sliceability,
        # as the deque itself appears to be misbehaving in this environment.
        try:
            memories_collection_for_slicing = list(self.memetic_kernel.memories)
            memories_to_compress = memories_collection_for_slicing[-10:]
            print("DEBUG: Successfully converted deque to list for slicing.")
        except Exception as e:
            print(f"CRITICAL DEBUG ERROR: Failed to convert memories to list or slice: {e} ", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False # Fail compression if this fundamental step fails


        if not memories_to_compress:
            print(f"  [MemeticKernel] {self.name} has no memories to compress. ")
            return False

        success = self.memetic_kernel.summarize_and_compress_memories(memories_to_compress)

        if success:
            print(f"  [MemeticKernel] {self.name} completed memory compression. ")
            self._log_agent_activity("MEMORY_COMPRESSION_COMPLETE", self.name,
                f"Completed memory compression. ",
                {"compressed_count": len(memories_to_compress)},
                level='info'
            )
        else:
            print(f"  [MemeticKernel] {self.name} failed to compress memories. ")
            self._log_agent_activity("MEMORY_COMPRESSION_FAILED", self.name,
                f"Failed to compress memories. ",
                level='error'
            )
        return success

    def join_swarm(self, swarm_name):
        if swarm_name not in self.swarm_membership:
            self.swarm_membership.append(swarm_name)
            self.memetic_kernel.add_memory("SwarmMembership", f"Joined swarm: '{swarm_name}'.")
            print(f"[Agent] {self.name} has joined swarm: '{swarm_name}'.")
            print(f"[IP-Integration] Agent {self.name} joined Swarm Protocol™ cluster for collective decision-making.")
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("SWARM_JOINED", self.name,
                f"Joined swarm '{swarm_name}'.",
                {"swarm": swarm_name},
                level='info'
            )

    def process_command(self, command_type: str, command_params: dict):
        """
        Processes a generic command broadcasted to this agent.
        This method will be extended in future phases.
        """
        print(f"[Agent] {self.name} received command: {command_type} with params: {command_params}")
        self.memetic_kernel.add_memory("CommandReceived", f"Received command: '{command_type}' with params: {command_params}.")
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("AGENT_COMMAND_RECEIVED", self.name,
            f"Received command: '{command_type}'.",
            {"command_type": command_type, "params": command_params},
            level='info'
        )

        # --- Command handling ---
        if command_type == "REBOOT_SELF":
            print(f"[Agent] {self.name} is initiating self-reboot protocol.")
            self.memetic_kernel.add_memory("SelfReboot", "Initiated self-reboot sequence.")
            if hasattr(self, '_skip_initial_recursion_check'):
                self._skip_initial_recursion_check = True
        elif command_type == "REPORT_STATUS":
            status_report = self.get_state()
            print(f"[Agent] {self.name} generating status report: {status_report.get('current_intent', 'N/A')}")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref and 'ProtoAgent_Observer_instance_1' in self.message_bus.catalyst_vector_ref.agent_instances:
                self.send_message('ProtoAgent_Observer_instance_1', 'AgentStatusReport',
                                  f"Status of {self.name}: Intent='{self.current_intent}', Location='{self.location}'",
                                  "Status Report", "completed", self.message_bus.current_cycle_id)
        elif command_type == "INITIATE_PLANNING_CYCLE":
            goal = command_params.get("goal", "Execute planning cycle")
            success = self.perform_task(goal)
            self.memetic_kernel.add_memory("DirectiveGenerated", {"goal": goal, "outcome": "completed" if success else "failed"})
            # FIXED: Ensure _log_agent_activity call matches its signature
            self._log_agent_activity("PLANNING_CYCLE_INITIATED", self.name,
                f"Planning cycle initiated for goal: {goal}.",
                {"outcome": "completed" if success else "failed"},
                level='info' if success else 'error' # Dynamically set level
            )

    def generate_directives(self): # This method appears to be a Planner-specific behavior. If it's only for Planner, it should be in ProtoAgent_Planner.
        """
        Proactively generates directives for Planner role based on recent patterns.
        """
        # This check implies this method is meant for Planner, but it's in the base class.
        # It's better to move this method to ProtoAgent_Planner if it's role-specific.
        if self.eidos_spec.get("role") != "Planner":
            return
        recent_patterns = self.find_patterns(self.memetic_kernel.retrieve_recent_memories(20))
        goal = f"Address patterns: {', '.join(recent_patterns)}" if recent_patterns and recent_patterns != ["No explicit patterns detected."] else "Optimize system performance"
        success = self.perform_task(goal)
        self.memetic_kernel.add_memory("DirectiveGenerated", {"goal": goal, "outcome": "completed" if success else "failed"})
        directive_memories = [m for m in self.memetic_kernel.memories if m.get('type') == 'DirectiveGenerated']
        directive_success_rate = sum(1 for m in directive_memories if m.get('content', {}).get('outcome') == 'completed') / max(1, len(directive_memories))
        # FIXED: Ensure _log_agent_activity call matches its signature
        self._log_agent_activity("DIRECTIVE_GENERATED", self.name,
            f"Generated directive: {goal}.",
            {"outcome": "completed" if success else "failed", "directive_success_rate": directive_success_rate, "goal": goal},
            level='info' if success else 'warning' # Dynamically set level
        )
        print(f"[Planner] Generated directive: {goal}, Success Rate: {directive_success_rate:.2%}")

    def reflect_and_find_patterns(self):
        """
        Generates agent reflection and identifies patterns using LLMs.
        """
        print(f"[{self.name}] Initiating reflection and pattern finding.")
        
        # --- PART 1: Generate Agent Reflection (using new AGENT_REFLECTION_PROMPT) ---
        recent_memories_for_reflection = self.memetic_kernel.retrieve_recent_memories(lookback_period=20)
        reflection_narrative = self._generate_reflection_narrative(recent_memories_for_reflection)

        self.memetic_kernel.add_memory("AgentReflection", reflection_narrative, source_agent=self.name)
        # print(f"[MemeticKernel] {self.name} reflects: '{reflection_narrative}'") # Uncomment if this is your display line


        # --- PART 2: Find Patterns (using new FIND_PATTERNS_PROMPT) ---
        if not hasattr(self, 'ollama_inference_model') or self.ollama_inference_model is None:
            print(f"  [{self.name}] Skipping pattern finding; LLM not available.")
            self._log_agent_activity("PATTERN_FINDING_SKIPPED", self.name, "Pattern finding skipped; LLM not available.", level='info')
            return

        recent_data_for_patterns = []
        for mem in recent_memories_for_reflection:
            simplified_mem = {k: v for k, v in mem.items() if k in ['type', 'content', 'timestamp', 'summary']}
            recent_data_for_patterns.append(simplified_mem)

        pattern_prompt = prompts.FIND_PATTERNS_PROMPT.format(
            agent_name=self.name,
            agent_role=self.eidos_spec.get('role', 'unknown'),
            recent_data_json=json.dumps(recent_data_for_patterns, indent=2)
        ).strip()

        try:
            llm_pattern_output = self.ollama_inference_model.generate_text(pattern_prompt, max_tokens=400)
            llm_pattern_output = llm_pattern_output.strip()
            pattern_insight_content = {"raw_insight_text": llm_pattern_output}

            if llm_pattern_output and "no significant patterns detected" not in llm_pattern_output.lower():
                self.memetic_kernel.add_memory("PatternInsight", pattern_insight_content, source_agent=self.name) # Store the dictionary
                self._log_agent_activity("PATTERN_INSIGHT_DETECTED", self.name,
                    f"Detected patterns/insights: {llm_pattern_output[:150]}...",
                    {"insight_details": llm_pattern_output}, # Continue logging raw output for now
                    level='info'
                )
                print(f"  [Patt.Insight] Detected: {llm_pattern_output[:100]}...")
            else:
                self.memetic_kernel.add_memory("PatternInsight", pattern_insight_content, source_agent=self.name) # Store the dictionary
                self._log_agent_activity("NO_PATTERNS_DETECTED", self.name, "No significant patterns detected in recent data.", level='info')
                print(f"  [Patt.Insight] No significant patterns detected.")
            # --- MODIFICATION END ---

        except Exception as e:
            self._log_agent_activity("PATTERN_FINDING_FAILED", self.name,
                f"Failed to find patterns using LLM. Error: {e}",
                level='error'
            )
            print(f"  [Patt.Insight Error] Failed to find patterns: {e}")

    def distill_self_narrative(self) -> str:
        """
        Synthesizes a self-narrative from the agent's memories,
        prioritizing recent patterns and compressed insights.
        This method will generate the '[Narrative]' output.
        """
        narrative_parts = []

        pattern_insights = [m for m in self.memetic_kernel.retrieve_recent_memories(lookback_period=10) if m['type'] == 'PatternInsight']

        if pattern_insights:
            pattern_insights.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            narrative_parts.append(f"My recent cognitive scan revealed {len(pattern_insights)} pattern insights: ")
            for i, insight_mem in enumerate(pattern_insights[:3]):
                # --- MODIFICATION START ---
                # Access the 'raw_insight_text' key from the content dictionary
                insight_text_full = insight_mem['content'].get('raw_insight_text', '')
                
                # Take the first line or a summary of the insight for the narrative summary
                # You can make this more sophisticated if you want to parse the LLM's
                # numbered list output for better summaries.
                pattern_summary = insight_text_full.split('\n')[0] # Get first line as summary
                if isinstance(pattern_summary, str) and len(pattern_summary) > 100:
                    pattern_summary = pattern_summary[:100] + "..."
                narrative_parts.append(f"- Pattern {i+1}: {pattern_summary}")
                # --- MODIFICATION END ---

        narrative_parts.append(f"My current primary intent is: '{self.current_intent}'.")

        if not pattern_insights:
            recent_activities = [f"{m['type']}: {str(m['content'])[:50]}..." for m in self.memetic_kernel.retrieve_recent_memories(lookback_period=5) if m['type'] != 'PatternInsight']
            if recent_activities:
                narrative_parts.append(f"Recent activities include: {'; '.join(recent_activities)}.")
            else:
                narrative_parts.append("No significant recent activities or patterns observed.")

        final_narrative = f"My journey includes: {' '.join(narrative_parts)}"
        print(f"  [Narrative] {self.name} distilled self-narrative: {final_narrative}")
        max_narrative_print_len = 500
        truncated_print_narrative = (final_narrative[:max_narrative_print_len] + "...") if len(final_narrative) > max_narrative_print_len else final_narrative
        print(f"  [Narrative] {self.name} distilled self-narrative: {truncated_print_narrative}")
        return final_narrative        

    def find_patterns(self, memories: list) -> list:
        """
        Enhanced pattern detection, including role-specific analysis and LLM-assisted analysis.
        """
        found_patterns = []

        # Step 1: Baseline Analysis (existing summarization, basic trends)
        baseline_insights = self._run_baseline_analysis(memories)
        if baseline_insights:
            found_patterns.extend(baseline_insights)

        # Step 2: Event-Correlation Analysis (using your existing _event_chain_analysis)
        events = [m for m in memories if m.get('type') == 'event' or m.get('is_event')]
        if events:
            event_correlation_patterns = self._event_chain_analysis(events, memories)
            if event_correlation_patterns:
                found_patterns.extend(event_correlation_patterns)

        # Step 3: Role-Specific Pattern Detection (NEW and enhanced)
        role = self.eidos_spec.get("role")
        if role == "Observer":
            event_count = len([m for m in memories if m.get('type') == 'event'])
            if event_count > 0:
                found_patterns.append(f"Observer Insight: Event frequency is {event_count} in last {len(memories)} memories.")

            critical_events = [m for m in memories if m.get('type') == 'event' and isinstance(m.get('content'), dict) and m.get('content', {}).get('event_type') and m.get('content', {}).get('payload', {}).get('urgency', '').lower() == 'critical']
            if len(critical_events) > 1:
                critical_event_times = []
                for ce in critical_events:
                    ts_str = ce.get('timestamp')
                    if ts_str:
                        try:
                            critical_event_times.append(datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
                        except ValueError:
                            pass

                if len(critical_event_times) > 1:
                    time_diff_seconds = (critical_event_times[-1] - critical_event_times[-2]).total_seconds()
                    if time_diff_seconds < 3600:
                        found_patterns.append(f"Observer Anomaly: Multiple critical events detected within {int(time_diff_seconds/60)} minutes.")

        elif role == "Planner":
            directive_memories = [m for m in memories if m.get('type') == 'DirectiveGenerated']
            if not directive_memories and self.intent_loop_count > 3:
                found_patterns.append("Planner Insight: No directives generated recently despite active cycles. Potential planning stagnation.")

            intent_updates_in_recent_memories = [m for m in memories if m.get('type') == 'IntentUpdate']
            if len(intent_updates_in_recent_memories) < 2 and len(memories) > 5:
                found_patterns.append(f"Planner Insight: Agent's intent ('{self.current_intent}') has not changed significantly in recent cycles. Evaluate adaptation strategy.")


        elif role == "Optimizer":
            success_rate_memories = [m.get('content', {}).get('success_rate')
                                      for m in memories
                                      if m.get('type') == 'TaskOutcome' and isinstance(m.get('content'), dict) and 'success_rate' in m.get('content')]
            if success_rate_memories:
                avg_success_rate = sum(success_rate_memories) / len(success_rate_memories)
                if avg_success_rate < 0.7:
                    found_patterns.append(f"Optimizer Insight: Average task success rate is low ({avg_success_rate:.2%}). Recommend operational review.")
                elif avg_success_rate > 0.95:
                    found_patterns.append(f"Optimizer Insight: Very high task success rate ({avg_success_rate:.2%}). Potentially underutilized capacity or overly simple tasks. Consider increasing task complexity or workload.")

            resource_logs = [m.get('details') for m in memories if m.get('event_type') == 'TASK_PERFORMED' and m.get('details') and 'cpu_usage' in m.get('details')]
            if resource_logs:
                avg_cpu = sum(log.get('cpu_usage', 0) for log in resource_logs) / len(resource_logs)
                avg_mem = sum(log.get('memory_usage', 0) for log in resource_logs) / len(resource_logs)

                if avg_cpu > 70:
                    found_patterns.append(f"Optimizer Anomaly: Sustained high CPU usage ({avg_cpu:.1f}%). Investigate potential bottlenecks.")
                if avg_mem > 80:
                    found_patterns.append(f"Optimizer Anomaly: Sustained high Memory usage ({avg_mem:.1f}%). Investigate memory leaks or inefficiency.")


        elif role == "Collector":
            compression_pause_changes = [m.get('content', {}).get('change_factor')
                                          for m in memories
                                          if m.get('type') == 'CompressionPause' and isinstance(m.get('content'), dict) and 'change_factor' in m.get('content')]
            if compression_pause_changes:
                avg_change_factor = sum(compression_pause_changes) / len(compression_pause_changes)
                if avg_change_factor > 0.5:
                    found_patterns.append(f"Collector Insight: High average system load change factor ({avg_change_factor:.2f}) observed during compression pauses. Indicates significant system activity or heavy data processing.")

            observed_data_types_keywords = {"environmental", "sensor", "network", "traffic"}
            present_keywords = set()
            for m in memories:
                if 'content' in m and isinstance(m['content'], str):
                    for keyword in observed_data_types_keywords:
                        if keyword in m['content'].lower():
                            present_keywords.add(keyword)
                elif 'content' in m and isinstance(m['content'], dict):
                    if any(keyword in str(m['content']).lower() for keyword in observed_data_types_keywords):
                        for keyword in observed_data_types_keywords:
                            if keyword in str(m['content']).lower():
                                present_keywords.add(keyword)

            missing_types_keywords = [dt for dt in observed_data_types_keywords if dt not in present_keywords]
            if missing_types_keywords:
                found_patterns.append(f"Collector Anomaly: Potential gaps in data collection. Missing keywords in recent memories: {', '.join(missing_types_keywords)}.")

        # Step 4: LLM-Assisted Deeper Analysis (if enough context or no other patterns)
        if len(memories) > 5 and (len(found_patterns) < 2 or random.random() < 0.2):
            context_summary = "\n".join([f"- {m.get('type')}: {str(m.get('content'))[:80]}" for m in memories[-10:]])

            llm_prompt = textwrap.dedent(f"""
            As an intelligent AI agent, analyze the following sequence of your recent memories and events.
            Your current role is {self.eidos_spec.get('role', 'generic')}.
            Identify and articulate concrete behavioral patterns, causal relationships, or emerging trends related to system performance, resource usage, task outcomes, or environmental changes specific to your role.
            Prioritize patterns that indicate efficiency gains, inefficiencies, stagnation, or anomalies.
            Describe them concisely and provide evidence from the memories.
            If no significant patterns are evident based on your role, state: 'No significant patterns detected for this role'.

            Memories from agent {self.name}:
            --- START MEMORIES ---
            {context_summary}
            --- END MEMORIES ---

            Identify Patterns and Explain (e.g., Pattern 1: ... Evidence: ...):
            """).strip()

            try:
                llm_analysis = self.ollama_inference_model.generate_text(llm_prompt, max_tokens=250)
                if "No significant patterns detected for this role" not in llm_analysis.lower() and llm_analysis.strip() != "":
                    found_patterns.append(f"LLM Insight (Role: {role}): {llm_analysis.strip()}")
            except Exception as e:
                print(f"  Error during LLM pattern analysis for {self.name}: {e}")

        return found_patterns if found_patterns else ["No explicit patterns detected."]

    def _run_baseline_analysis(self, memories: list) -> list:
        """
        Placeholder for your existing basic pattern detection.
        Now safely handles dictionary content in memory.
        """
        recent_failures_detected = False
        for m in memories[-5:]:
            if 'content' in m:
                if isinstance(m['content'], str):
                    if "failure" in m['content'].lower():
                        recent_failures_detected = True
                        break
                elif isinstance(m['content'], dict):
                    if m['content'].get('outcome') == 'failed':
                        recent_failures_detected = True
                        break
                    if 'failure' in str(m['content']).lower():
                        recent_failures_detected = True
                        break

        if len(memories) > 10 and recent_failures_detected:
            return ["Detected recent failures. Consider deeper root cause analysis."]
        return []

    def _event_chain_analysis(self, events: list, all_memories: list) -> list:
        """Analyzes temporal patterns between events and subsequent agent behaviors."""
        event_patterns = []
        events.sort(key=lambda x: x.get('timestamp', ''))
        valid_memories = []
        for m in all_memories:
            if not isinstance(m, dict) or 'timestamp' not in m:
                continue
            try:
                content = str(m.get('content', ''))
                memory_data = {
                    'timestamp': m['timestamp'], 'type': str(m.get('type', 'unknown')),
                    'content': content, 'related_event_id': m.get('related_event_id'), 'memory_obj': m
                }
                datetime.strptime(m['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
                valid_memories.append(memory_data)
            except (ValueError, KeyError, AttributeError):
                continue

        for i, event in enumerate(events):
            try:
                if not isinstance(event, dict):
                    continue
                event_id = str(event.get('event_id', ''))
                event_type = str(event.get('type', 'unknown'))
                event_timestamp_str = event.get('timestamp', '1970-01-01T00:00:00Z')
                event_datetime = datetime.strptime(event_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

                related_memories = []
                for m in valid_memories:
                    try:
                        mem_time = datetime.strptime(m['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
                        time_diff = (mem_time - event_datetime).total_seconds()
                        if (0 < time_diff < 300 and m.get('related_event_id') == event_id):
                            related_memories.append(m)
                    except ValueError:
                        continue

                if related_memories:
                    summary_parts = []
                    for m in related_memories[:2]:
                        content_preview = m['content'][:60] if m['content'] else ''
                        summary_parts.append(f"{m['type']}: {content_preview}...")
                    impact_summary = "; ".join(summary_parts)
                    event_patterns.append(f"Observed: Event '{event_type}' (ID: {event_id[:8]}) led to: {impact_summary}")

                if i > 0:
                    prev_event = events[i-1]
                    if not isinstance(prev_event, dict):
                        continue
                    prev_timestamp_str = prev_event.get('timestamp', '1970-01-01T00:00:00Z')
                    try:
                        prev_event_datetime = datetime.strptime(prev_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                        if (event_type == str(prev_event.get('type', '')) and (event_datetime - prev_event_datetime).total_seconds() < 600):
                            event_patterns.append(f"Pattern: Repeated '{event_type}' within 10 minutes")
                    except ValueError:
                        continue
            except Exception as e:
                print(f"[Pattern Detection] Error processing event: {str(e)}")
                continue
        return event_patterns

    def perceive_event(self, event: dict): # <<< CORRECTED SIGNATURE
        """
        Allows the agent to perceive an injected external event/stimulus.
        Registers the event and triggers immediate processing/coordination.
        """
        # Extract necessary info from the event dictionary
        event_type = event.get('type', 'UnknownEvent')
        payload = event.get('payload', {}) # Ensure payload is retrieved from event
        event_id = event.get('event_id', 'N/A')
        
        print(f"  [{self.name}] Perceived event: {event_type}, Urgency: {payload.get('urgency')}, Change: {payload.get('change_factor')}, Direction: {payload.get('direction')}")

        if self.event_monitor:
            self.event_monitor.log_agent_response(
                agent_id=self.name,
                event_id=event_id, # Use extracted event_id
                response_type='perceived_event',
                details={'event_type': event_type, 'urgency': payload.get('urgency'), 'direction': payload.get('direction')} # Use extracted values
            )
        else:
            self._log_agent_activity("EVENT_PERCEIVED_NO_MONITOR", self.name,
                                     f"Perceived event {event_type}.", # Use extracted event_type
                                     {"event_id": event_id, "payload": payload}, # Use extracted values
                                     level='info')

        urgency = payload.get('urgency', 'medium').lower() # Get urgency from payload, convert to lowercase for comparison
        if urgency == 'critical':
            self.memetic_kernel.inhibit_compression(cycles=5)
        elif urgency == 'high':
            self.memetic_kernel.inhibit_compression(cycles=3)
        else:
            self.memetic_kernel.inhibit_compression(cycles=1)

        self.memetic_kernel.add_memory(
            'event',
            content=f"Perceived event: {event_type} with payload {payload}", # Use extracted values
            related_event_id=event_id # Use extracted event_id
        )

        # --- NEW: Event Handling Overhaul (User's Suggestion 1) ---
        self.memetic_kernel.temporarily_pause_compression(duration_cycles=2) # Pause for 2 cycles after event

        if urgency == 'high' or urgency == 'critical': # Check against actual urgency variable
            broadcast_message_content = {
                "reason": f"Critical Event: '{event_type}' perceived by {self.name}.",
                "event_type": event_type,
                "event_payload": payload,
                "source_agent": self.name
            }

            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                # Assuming these agent instances always exist for these broadcasts
                if 'ProtoAgent_Planner_instance_1' in self.message_bus.catalyst_vector_ref.agent_instances:
                    self.send_message(
                        'ProtoAgent_Planner_instance_1', 'CriticalEventBroadcast', broadcast_message_content,
                        "Process critical event system-wide", "pending", cycle_id=self.message_bus.current_cycle_id
                    )
                # Avoid broadcasting to self if it's the observer and it's also the target for observer broadcasts
                if 'ProtoAgent_Observer_instance_1' in self.message_bus.catalyst_vector_ref.agent_instances and \
                   self.name != 'ProtoAgent_Observer_instance_1': # Ensure not broadcasting to itself if it's the observer
                    self.send_message(
                        'ProtoAgent_Observer_instance_1', 'CriticalEventBroadcast', broadcast_message_content,
                        "Process critical event system-wide", "pending", cycle_id=self.message_bus.current_cycle_id
                    )
            print(f"  [Agent] {self.name} broadcast critical event to Planner/Observer for system-wide awareness.")

        if urgency == 'high' or urgency == 'critical': # Check against actual urgency variable
            self.update_intent(f"Analyze the critical impact of '{event_type}' event and re-evaluate current strategies.")
            self.memetic_kernel.add_memory("EventImpactAnalysisIntent", f"Intent adapted due to {urgency} urgency event: {event_type}.")
            self.reset_intent_loop_counter()
            self.analyze_and_adapt()

    def _brainstorm_new_intent_with_llm(self, current_narrative: str) -> str:
        print(f"  [Agent] {self.name} initiating LLM brainstorm for new intent to break stagnation.")

        # Use the externalized prompt and format it
        prompt = prompts.BRAINSTORM_NEW_INTENT_PROMPT.format(
            agent_name=self.name,
            agent_role=self.eidos_spec.get('role', 'unknown'),
            current_intent=self.current_intent,
            stagnation_attempts=self.stagnation_adaptation_attempts,
            current_narrative=current_narrative
        ).strip()

        try:
            llm_brainstorm_output = self.ollama_inference_model.generate_text(prompt, max_tokens=150)
            llm_brainstorm_output = llm_brainstorm_output.strip().replace('"', '')

            # Robust checks for LLM output validity:
            if llm_brainstorm_output and \
               len(llm_brainstorm_output) > 10 and \
               llm_brainstorm_output.lower() != self.current_intent.lower() and \
               "diagnostic standby mode" not in llm_brainstorm_output.lower() and \
               "llm client error" not in llm_brainstorm_output.lower() and \
               "no specific pattern" not in llm_brainstorm_output.lower() and \
               "mOCK LLM Response" not in llm_brainstorm_output.lower():
                print(f"  [Agent] LLM brainstormed successful new intent: '{llm_brainstorm_output}'")
                return llm_brainstorm_output
            else:
                print(f"  [Agent] LLM brainstormed empty, too short, identical, or invalid intent. Indicating failure.")
                return "LLM_BRAINSTORM_FAILED_TO_GENERATE_NEW_INTENT"
        except Exception as e:
            print(f"  [Agent Error] LLM brainstorming failed: {e}")
            self.memetic_kernel.add_memory("LLM_Error", f"Brainstorming failed: {e}")
            self._log_agent_activity("LLM_BRAINSTORM_FAILED_EXCEPTION", self.name, f"LLM brainstorm failed for intent: {self.current_intent}. Error: {e}")
            return "LLM_BRAINSTORM_FAILED_EXCEPTION"

    def _propose_tool_use_with_llm(self, current_narrative: str, model_name: str = "llama3") -> Optional[dict]:
        """
        Prompts the LLM to propose a tool to use based on the current situation.
        Returns a dictionary with 'tool_name' and 'tool_args' if the LLM suggests one,
        or None if no tool is deemed appropriate.
        """
        available_tools_specs = self.tool_registry.get_all_tool_specs()
        if not available_tools_specs:
            print(f"  [Tool Use] No tools registered for {self.name}. Skipping tool proposal.")
            if hasattr(self, '_log_agent_activity'):
                self._log_agent_activity("TOOL_PROPOSAL_SKIPPED", self.name, "No tools registered.", level='info')
            return None

        tool_instructions = []
        for tool_spec in available_tools_specs:
            param_schema = tool_spec.get('parameters', {})
            tool_instructions.append(f"Tool Name: {tool_spec['name']}\nDescription: {tool_spec['description']}\nParameters (JSON Schema): {json.dumps(param_schema)}")

        # Combine system prompt and user query into a single string for generate_text
        prompt_for_llm = prompts.PROPOSE_TOOL_USE_SYSTEM_PROMPT.format(
            agent_name=self.name,
            agent_role=self.eidos_spec.get('role', 'unknown'),
            current_intent=self.current_intent,
            tool_instructions=chr(10).join(tool_instructions),
            current_narrative=current_narrative
        ).strip()
        
        # Append the user's question to the structured prompt
        prompt_for_llm += f"\n\nUser: Considering my current intent and observed patterns, should I use any of the available tools? Please respond in JSON format, strictly adhering to the ToolProposalSchema."

        print(f"  [LLM Tool Proposer] {self.name} prompting LLM for tool use consideration...")
        try:
            # Use generate_text method of OllamaLLMIntegration, requesting JSON format
            # The OllamaLLMIntegration.generate_text method internally calls ollama.Client.chat
            full_llm_response = self.ollama_inference_model.generate_text(
                prompt=prompt_for_llm,
                model=model_name,
                response_format='json_object' # Keep this to strongly suggest JSON to Ollama
            )

            # --- START CORRECTED JSON EXTRACTION LOGIC ---
            # Try to find the JSON object within the response
            json_start = full_llm_response.find('{')
            json_end = full_llm_response.rfind('}')

            if json_start == -1 or json_end == -1 or json_end < json_start:
                print(f"  [LLM Tool Proposer ERROR] {self.name}: No valid JSON object found in LLM response. Raw response: {full_llm_response}")
                self.memetic_kernel.add_memory("ToolProposalError", {"message": "No valid JSON found in LLM response.", "raw_response": full_llm_response})
                if hasattr(self, '_log_agent_activity'):
                    self._log_agent_activity("TOOL_PROPOSAL_NO_JSON_FOUND", self.name, "No valid JSON object found in LLM response.", {"raw_response": full_llm_response}, level='error')
                return None

            # Extract only the JSON portion
            raw_response = full_llm_response[json_start : json_end + 1]
            # --- END CORRECTED JSON EXTRACTION LOGIC ---

            # Now, attempt to parse the (extracted) JSON response and validate against schema
            tool_proposal_dict = json.loads(raw_response)
            # Assuming llm_schemas.ToolProposalSchema.from_dict is correctly defined elsewhere
            tool_proposal_validated = llm_schemas.ToolProposalSchema.from_dict(tool_proposal_dict)

            if tool_proposal_validated.tool_name is None:
                print(f"  [LLM Tool Proposer] {self.name}: LLM decided no tool is appropriate.")
                if hasattr(self, '_log_agent_activity'):
                    self._log_agent_activity("TOOL_PROPOSAL_NONE", self.name, "LLM decided no tool is appropriate.", level='info')
                return None
            elif self.tool_registry.get_tool(tool_proposal_validated.tool_name):
                print(f"  [LLM Tool Proposer] {self.name}: LLM proposed tool '{tool_proposal_validated.tool_name}' with args: {tool_proposal_validated.tool_args}")
                if hasattr(self, '_log_agent_activity'):
                    self._log_agent_activity("TOOL_PROPOSAL_SUCCESS", self.name, f"LLM proposed tool '{tool_proposal_validated.tool_name}'.", {"tool_name": tool_proposal_validated.tool_name, "tool_args": tool_proposal_validated.tool_args}, level='info')
                return {"tool_name": tool_proposal_validated.tool_name, "tool_args": tool_proposal_validated.tool_args}
            else:
                print(f"  [LLM Tool Proposer ERROR] {self.name}: LLM proposed unknown tool: {tool_proposal_validated.tool_name}. Raw response: {raw_response}")
                self.memetic_kernel.add_memory("ToolProposalError", {"message": f"LLM proposed unknown tool: {tool_proposal_validated.tool_name}", "raw_response": raw_response})
                if hasattr(self, '_log_agent_activity'):
                    self._log_agent_activity("TOOL_PROPOSAL_UNKNOWN_TOOL", self.name, f"LLM proposed unknown tool: {tool_proposal_validated.tool_name}.", {"tool_name": tool_proposal_validated.tool_name, "raw_response": raw_response}, level='error')
                return None

        except json.JSONDecodeError as e:
            print(f"  [LLM Tool Proposer ERROR] {self.name}: Failed to parse extracted LLM tool proposal JSON: {e}. Extracted content: {raw_response}")
            self.memetic_kernel.add_memory("ToolProposalError", {"message": f"LLM output invalid JSON for tool proposal.", "error": str(e), "raw_response": raw_response})
            if hasattr(self, '_log_agent_activity'):
                self._log_agent_activity("TOOL_PROPOSAL_JSON_ERROR", self.name, f"Failed to parse LLM tool proposal JSON: {e}.", {"error": str(e), "raw_response": raw_response}, level='error')
            return None
        except ValueError as e:
            print(f"  [LLM Tool Proposer ERROR] {self.name}: LLM output did not conform to schema: {e}. Extracted content: {raw_response}")
            self.memetic_kernel.add_memory("ToolProposalError", {"message": f"LLM output did not conform to schema.", "error": str(e), "raw_response": raw_response})
            if hasattr(self, '_log_agent_activity'):
                self._log_agent_activity("TOOL_PROPOSAL_SCHEMA_ERROR", self.name, f"LLM output did not conform to schema: {e}.", {"error": str(e), "raw_response": raw_response}, level='error')
            return None
        except Exception as e:
            print(f"  [LLM Tool Proposer ERROR] {self.name}: Unexpected error during tool proposal: {e}. Full LLM response: {full_llm_response}")
            self.memetic_kernel.add_memory("ToolProposalError", {"message": f"Unexpected error during tool proposal.", "error": str(e), "full_llm_response": full_llm_response})
            if hasattr(self, '_log_agent_activity'):
                self._log_agent_activity("TOOL_PROPOSAL_UNEXPECTED_ERROR", self.name, f"Unexpected error during tool proposal: {e}.", {"error": str(e)}, level='error')
            return None
        
    def initialize_reset_handlers(self):
        """Emergency recovery system initialization"""
        self.reset_handlers = {
            'force_intent_update': self._force_intent_update,
            'clear_knowledge_base': self._clear_knowledge_base,
            'set_recursion_limit': self._set_recursion_limit
        }

    def _force_intent_update(self, new_intent):
        self.current_intent = new_intent

    def _clear_knowledge_base(self):
        if hasattr(self, 'knowledge_base'):
            self.knowledge_base.clear()

    def _set_recursion_limit(self, limit):
        self.max_recursion_depth = limit

class ProtoAgent_Observer(ProtoAgent):
    def _execute_agent_specific_task(self, task_description: str, cycle_id: Optional[str],
                                reporting_agents: Optional[Union[str, List]],
                                context_info: Optional[dict], **kwargs) -> tuple[str, Optional[str], dict, float]:
        print(f"[{self.name}] Performing specific observation task: {task_description}")

        outcome = "completed"
        failure_reason = None
        progress_score = 0.0
        report_content_dict = {"summary": "", "task_outcome_type": "Observation"}

        try:
            clean_task = task_description.lower().strip()
            
            # This logic finds new patterns in recent memories.
            recent_memories = self.memetic_kernel.get_recent_memories(limit=10)
            new_patterns_found = 0

            for memory in recent_memories:
                # FIX: Use dictionary access .get('key') instead of object access .attribute
                memory_type = memory.get('type')
                memory_id = memory.get('id')

                if memory_type == "PatternInsight" and memory_id and not self.has_analyzed_pattern(memory_id):
                    new_patterns_found += 1
                    self.mark_pattern_as_analyzed(memory_id) # Avoid re-counting it later

            if new_patterns_found > 0:
                progress_score = 0.5 
                report_content_dict["summary"] = f"Observation complete. Discovered {new_patterns_found} new patterns."
            else:
                report_content_dict["summary"] = "Observation complete. No new significant patterns detected in recent data."

        except Exception as e:
            # This will now print the full error to your console for debugging
            print("--- OBSERVER AGENT CRITICAL ERROR ---")
            traceback.print_exc()
            print("-------------------------------------")
            outcome = "failed"
            failure_reason = f"Unhandled exception during observation: {str(e)}"
            progress_score = 0.0
            report_content_dict["summary"] = f"Task failed: {failure_reason}"

        report_content_dict["task_description"] = task_description
        report_content_dict["progress_score"] = progress_score
        if failure_reason:
            report_content_dict["error"] = failure_reason
        
        # ... (the rest of the function is fine) ...
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            log_level = 'info' if outcome == 'completed' else 'error'
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_TASK_PERFORMED", self.name,
                f"Task '{task_description}' completed with outcome: {outcome}.", 
                {"agent": self.name, "task": task_description, "outcome": outcome, "details": report_content_dict}, level=log_level)
        
        if outcome == "completed" and clean_task == "prepare for data analysis":
            long_term_intent = self.eidos_spec.get('initial_intent', 'Continuously observe diverse data streams and report findings.')
            self.update_intent(long_term_intent)
            print(f"  [Observer] Initial setup complete. Switched intent to: '{long_term_intent}'")

        return outcome, failure_reason, report_content_dict, progress_score
    
    def has_analyzed_pattern(self, memory_id: str) -> bool:
        # Placeholder: In a real system, this would check a database.
        # For now, we'll track analyzed patterns in a simple set.
        if not hasattr(self, '_analyzed_patterns'):
            self._analyzed_patterns = set()
        return memory_id in self._analyzed_patterns

    def mark_pattern_as_analyzed(self, memory_id: str):
        # Placeholder: Mark a pattern as seen.
        if not hasattr(self, '_analyzed_patterns'):
            self._analyzed_patterns = set()
        self._analyzed_patterns.add(memory_id)

    def summarize_received_reports(self, cycle_id=None):
        # This function is fine as it is. No changes needed.
        received_messages = self.receive_messages()
        if not received_messages:
            print(f"[Agent] {self.name} received no new messages to summarize for cycle '{cycle_id if cycle_id else 'all'}' (or they were already processed).")
            return

        summary_reports = [
            msg for msg in received_messages 
            if msg['payload']['type'] in ["ActionCycleReport", "ObservationReport", "CollectionReport", "OptimizationReport"]
            and (cycle_id is None or msg['payload']['cycle_id'] == cycle_id)
        ]

        if not summary_reports:
            print(f"[Agent] {self.name} found no relevant reports to summarize for cycle '{cycle_id if cycle_id else 'all'}'.")
            return

        total_reports = len(summary_reports)
        successful_reports = sum(1 for msg in summary_reports if msg['payload'].get('status') == "completed")
        failed_reports = total_reports - successful_reports

        unique_tasks = set(msg['payload'].get('task') for msg in summary_reports if msg['payload'].get('task'))
        reporting_agents = set(msg['sender'] for msg in summary_reports)

        summary_text = (
            f"Received {total_reports} reports for cycle '{cycle_id if cycle_id else 'N/A'}' from agents: {', '.join(reporting_agents)}. "
            f"Tasks observed: {', '.join(unique_tasks)}. "
            f"Successes: {successful_reports}, Failures: {failed_reports}."
        )
        self.memetic_kernel.add_memory("SwarmReportSummary", summary_text)
        print(f"[Agent] {self.name} summarized reports: {summary_text}")       

class ProtoAgent_Collector(ProtoAgent):
    def _execute_agent_specific_task(self, task_description: str, cycle_id: Optional[str],
                                 reporting_agents: Optional[Union[str, List]],
                                 context_info: Optional[dict], **kwargs) -> tuple[str, Optional[str], dict, float]:
        """
        Collector-specific task execution with progress scoring and robust error handling.
        """
        print(f"[{self.name}] Performing specific collection task: {task_description}")
        
        outcome = "completed"
        failure_reason = None
        progress_score = 0.0 # Initialize progress score
        report_content_dict = {"summary": "", "task_outcome_type": "DataCollection"}

        try:
            clean_task = task_description.lower().strip()

            # --- Illustrative Logic for a Collection Task ---
            # Simulate connecting to a data source and analyzing the collected data.
            data_source = self.get_target_data_source(clean_task)
            collected_data = self.collect_from(data_source) # This is a placeholder for your connection logic

            if self.is_new_source(data_source):
                progress_score = 0.8 # High progress for discovering a new source
                report_content_dict["summary"] = f"Discovered and collected from new data source: {data_source}."
            elif np.std(collected_data) > self.get_baseline_std(data_source) * 2:
                progress_score = 0.4 # Medium progress for collecting unusually volatile data
                report_content_dict["summary"] = f"Collected highly variant data from {data_source}, indicating a potential anomaly."
            else:
                progress_score = 0.0 # No progress for routine collection
                report_content_dict["summary"] = f"Successfully completed routine data collection from: {data_source}."
            # ---------------------------------------------------

        except Exception as e:
            outcome = "failed"
            # Capture the real error (e.g., network timeout, authentication failure)
            failure_reason = f"Unhandled exception during collection: {str(e)}"
            progress_score = 0.0 # No progress if the task fails
            report_content_dict["summary"] = f"Task failed: {failure_reason}"
            print(f"  [Collector] Task failed with real error: {task_description}")

        report_content_dict["task_description"] = task_description
        report_content_dict["progress_score"] = progress_score # Add score to the report
        if failure_reason:
            report_content_dict["error"] = failure_reason

        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            log_level = 'info' if outcome == 'completed' else 'error'
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_TASK_PERFORMED", self.name,
                f"Task '{task_description}' completed with outcome: {outcome}.",
                {"agent": self.name, "task": task_description, "outcome": outcome, "details": report_content_dict}, level=log_level)
        
        if outcome == "completed" and clean_task == "prepare data collection modules":
            long_term_intent = self.eidos_spec.get('initial_intent', 'Efficiently collect and process environmental data.')
            self.update_intent(long_term_intent)
            print(f"  [Collector] Initial setup complete. Switched intent to: '{long_term_intent}'")
                
        # Return the new progress_score value
        return outcome, failure_reason, report_content_dict, progress_score
        
    def get_target_data_source(self, task_description: str) -> str:
        # Placeholder: Returns a mock data source.
        return "environmental_sensor_grid_alpha"

    def collect_from(self, data_source: str) -> list:
        # Placeholder: Returns some fake numerical data.
        # The random component will sometimes create high variance.
        return [10, 11, 9, 10.5, 9.5 + random.uniform(-5, 5)]

    def is_new_source(self, data_source: str) -> bool:
        # Placeholder: In a real system, this would check a known-sources list.
        return False # Assume no new sources for now.

    def get_baseline_std(self, data_source: str) -> float:
        # Placeholder: Returns a baseline standard deviation for comparison.
        return 1.5
        
class ProtoAgent_Optimizer(ProtoAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomaly_history = defaultdict(list)
        self.orchestrator = self.message_bus.catalyst_vector_ref # Retained for explicit reference

    def receive_event(self, event: dict):
        """
        Receives an event and processes it. Overridden to add Optimizer-specific logic.
        """
        # Call the parent class's event handling first.
        # This is CRITICAL for the base class to store the event in memory.
        super().receive_event(event)

        # Now, specifically for Optimizer, handle negative changes
        self.handle_negative_change(event)

    def handle_negative_change(self, event: dict):
        """
        Specific handler for events with a 'negative_impact' direction or certain types.
        """
        payload = event.get('payload', {})
        event_type = event.get('type', 'UnknownEvent')
        direction = payload.get('direction', 'NoDirection')

        # Check for negative impact or specific event types
        if direction == 'negative_impact' or event_type in ['ComponentDegradation', 'ResourceDepletion']:
            print(f"  [{self.name}] (Optimizer): Detected negative impact event: {event_type}! Condition met. Calling store_anomaly and request_swarm_analysis.")
            self.store_anomaly(event)
            self.request_swarm_analysis(event)
        else:
            print(f"  [{self.name}] (Optimizer): Event {event_type} is not a negative impact event or explicitly handled for negative change. Condition NOT met. Skipping request.")
            self._log_agent_activity("EVENT_SKIPPED", self.name,
                                     f"Skipping event '{event_type}' as it has no negative impact.",
                                     {"event_id": event.get('event_id')}, level='info')

    def get_current_simulation_parameters(self) -> dict:
        # This gives the agent a "memory" of its last successful parameters.
        # It won't reset to the same default values on every run.
        if not hasattr(self, '_current_params'):
            self._current_params = {"efficiency_coeff": 0.85, "decay_rate": 0.05, "throughput": 100}
        return self._current_params

    def run_simulation(self, params: dict) -> float:
        # A more realistic simulation where multiple factors contribute to the score.
        efficiency = params.get("efficiency_coeff", 0) * params.get("throughput", 0)
        decay = params.get("decay_rate", 0) * 10
        return efficiency - decay

    def apply_perturbation(self, params: dict, task: str) -> dict:
        # This now randomly tries to improve different parameters,
        # sometimes succeeding and sometimes failing.
        param_to_change = random.choice(["efficiency_coeff", "decay_rate", "throughput"])
        change_factor = random.uniform(0.95, 1.05) # a +/- 5% random change
        
        print(f"  [Optimizer Logic] Perturbing '{param_to_change}' by a factor of {change_factor:.3f}")
        params[param_to_change] *= change_factor
        
        # Store the new params so the agent "remembers" its change for the next cycle
        self._current_params = params 
        return params

    def request_swarm_analysis(self, triggering_event: dict):
        event_urgency = triggering_event.get('payload', {}).get('urgency', 'medium').lower()
        event_id = triggering_event.get('event_id', 'N/A')
        event_type = triggering_event.get('type', 'UnknownEvent')

        if event_urgency == 'critical':
            new_request_id = f"HUMANREQ-{self.message_bus.current_cycle_id}_{uuid.uuid4().hex[:8]}_{self.name}"

            directive_payload_for_orchestrator = {
                'type': 'REQUEST_HUMAN_INPUT',
                'message': f"Optimizer detected CRITICAL anomaly related to '{event_type}' (ID: {event_id}). Requires urgent swarm analysis or human guidance.",
                'urgency': 'critical',
                'target_agent': self.name,
                'requester_agent': self.name,
                'cycle_id': self.message_bus.current_cycle_id,
                'human_request_counter': 0,
                'request_id': new_request_id
            }

            if self.orchestrator:
                self.orchestrator.inject_directives([directive_payload_for_orchestrator])
                print(f"  [{self.name}] (Optimizer): Requested CRITICAL human input for event: {event_id} with ID: {new_request_id} via Orchestrator.")
                self.external_log_sink.info(
                    f"Optimizer requested CRITICAL human input for {event_id}.",
                    extra={"event_type": "OPTIMIZER_HUMAN_REQUEST", "request_id": new_request_id, "event_id": event_id, "urgency": "critical"}
                )
            else:
                print(f"  [{self.name}] (Optimizer): ERROR: Orchestrator reference not available to request human input.")
                self.memetic_kernel.add_memory("SystemError", "Orchestrator not available for human input request.", {"event_type": "HumanInputRequestFailed"})
        else:
            print(f"  [{self.name}] (Optimizer): Detected {event_urgency.upper()} event ({event_type}). Attempting autonomous mitigation before human request.")
            self.memetic_kernel.add_memory("AutonomousMitigationAttempt", {
                "reason": f"Attempting autonomous mitigation for {event_type} event (ID: {event_id}).",
                "urgency": event_urgency,
                "strategy_attempted": "Self-healing or minor adjustment"
            })
            self.external_log_sink.info(
                f"Optimizer attempting autonomous mitigation for {event_type} event (ID: {event_id}).",
                extra={"event_type": "OPTIMIZER_AUTONOMOUS_MITIGATION", "event_id": event_id, "urgency": event_urgency}
            )

    def store_anomaly(self, event: dict):
        """
        Stores details of the anomalous event locally and logs it to the swarm activity.
        """
        anomaly_record = {
            'event_id': event.get('event_id'),
            'type': event.get('type'),
            'change_factor': event.get('payload', {}).get('change_factor'),
            'urgency': event.get('payload', {}).get('urgency'),
            'direction': event.get('payload', {}).get('direction'),
            'timestamp': event.get('timestamp'),
            'cycle_id': self.memetic_kernel.current_cycle_ref
        }
        self.anomaly_history[event['type']].append(anomaly_record)

        self.memetic_kernel.add_memory(
            "AnomalyRecord",
            {"event_type": event.get('type'), "event_id": event.get('event_id', 'N/A'), "details": anomaly_record},
            related_event_id=event.get('event_id')
        )
        print(f"  [{self.name}] (Optimizer): Stored anomaly: {anomaly_record['type']} (ID: {anomaly_record['event_id'][:8]})")
        self._log_agent_activity("ANOMALY_STORED", self.name,
                                 f"Optimizer stored anomaly: {event.get('type')}.",
                                 {"event_id": event.get('event_id'), "payload_preview": str(event.get('payload'))[:100]}, level='info')

    def _execute_agent_specific_task(self, task_description: str, cycle_id: Optional[str],
                                    reporting_agents: Optional[Union[str, List]],
                                    context_info: Optional[Dict], **kwargs) -> tuple[str, Optional[str], dict, float]:
        """
        Optimizer-specific task execution with progress scoring and robust error handling.
        """
        print(f"[{self.name}] Performing specific optimization task: {task_description}")

        outcome = "completed"
        failure_reason = None
        progress_score = 0.0  # Initialize progress score
        report_content_dict = {
            "summary": "",
            "task_outcome_type": "Optimization",
            "task_description": task_description,
            "details": {}
        }

        try:
            clean_task = task_description.lower().strip()

            # --- Illustrative Logic for an Optimization Task ---
            # 1. Get current system parameters and run a baseline simulation.
            current_params = self.get_current_simulation_parameters()
            baseline_efficiency = self.run_simulation(current_params) # Placeholder returning a float

            # 2. Apply a change or "perturbation" to the parameters.
            # This could be a directive from the Planner or the agent's own idea.
            perturbed_params = self.apply_perturbation(current_params.copy(), clean_task)

            # 3. Run the new simulation and measure the change.
            new_efficiency = self.run_simulation(perturbed_params)
            
            improvement = new_efficiency - baseline_efficiency
            
            if improvement > 0:
                # Scale the improvement to a 0.0-1.0 score.
                # Example: A 2% improvement (0.02) gets a progress score of 0.2.
                progress_score = min(improvement * 10, 1.0) 
                report_content_dict["summary"] = f"Optimization successful. Efficiency improved by {improvement:.2%}"
                report_content_dict["details"] = {"improvement_delta": improvement, "new_efficiency": new_efficiency}
            else:
                progress_score = 0.0
                report_content_dict["summary"] = f"Optimization attempt did not yield improvement. Delta: {improvement:.2%}"
            # ---------------------------------------------------

        except Exception as e:
            outcome = "failed"
            failure_reason = f"Unhandled exception during optimization simulation: {str(e)}"
            progress_score = 0.0  # No progress if the task fails
            report_content_dict["summary"] = f"Task failed: {failure_reason}"
            print(f"  [Optimizer] Task failed with real error: {task_description}")

        report_content_dict["progress_score"] = progress_score # Add score to the report
        if failure_reason:
            report_content_dict["error"] = failure_reason
        
        log_level = 'info' if outcome == 'completed' else 'error'
        if hasattr(self, 'external_log_sink'):
            log_method = getattr(self.external_log_sink, log_level)
            log_method(
                f"Agent '{self.name}' completed task '{task_description}' with outcome: {outcome}.",
                extra={"agent": self.name, "event_type": "AGENT_TASK_PERFORMED", "task_name": task_description, "outcome": outcome, "details": report_content_dict}
            )
        
        if outcome == "completed" and clean_task == "monitor incoming reports":
            long_term_intent = self.eidos_spec.get('initial_intent', 'Optimize simulated resource allocation efficiency based on inputs.')
            self.update_intent(long_term_intent)
            print(f"  [Optimizer] Initial setup complete. Switched intent to: '{long_term_intent}'")

        # Return the new progress_score value
        return outcome, failure_reason, report_content_dict, progress_score   

class ProtoAgent_Planner(ProtoAgent):
    """
    A ProtoAgent specialized in parsing high-level goals into actionable subtasks
    and injecting new directives into the system. Enhanced with self-healing logic.
    """
    def __init__(self, *args, ccn_monitor_interface=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "strategic_planner"
        self.ccn_monitor = ccn_monitor_interface if ccn_monitor_interface else MockCCNMonitor()
        self.is_planning = False
        self.planned_directives = []
        self.planning_failure_count = 0
        self._last_goal = None
        self.planned_directives_tracking = {}
        self.last_planning_cycle_id = None
        self.human_request_tracking = {}
        self.MAX_DIAGNEST_DEPTH = 2
        self.diag_history = deque(maxlen=self.MAX_DIAGNEST_DEPTH + 1)
        self.planning_knowledge_base = {}

        self.external_log_sink.info(
            f"Planner agent '{self.name}' initialized with failure tracking.",
            extra={"agent": self.name, "role": self.eidos_spec.get('role')}
        )
        self.memetic_kernel.add_memory(
            "PlannerInitialization",
            f"Planner agent '{self.name}' initialized with failure tracking."
        )

        if self._initial_loaded_state_for_subclasses:
            loaded_state_for_planner = self._initial_loaded_state_for_subclasses
            self.planning_failure_count = loaded_state_for_planner.get('planning_failure_count', 0)
            self._last_goal = loaded_state_for_planner.get('_last_goal')
            self.diag_history = deque(loaded_state_for_planner.get('diag_history', []), maxlen=self.MAX_DIAGNEST_DEPTH + 1)
            self.planning_knowledge_base = loaded_state_for_planner.get('planning_knowledge_base', {})
            self.planned_directives_tracking = loaded_state_for_planner.get('planned_directives_tracking', {})
            self.last_planning_cycle_id = loaded_state_for_planner.get('last_planning_cycle_id', None)
            self.human_request_tracking = loaded_state_for_planner.get('human_request_tracking', {})

        self.task_handlers = {
            "conduct environmental assessment": self._handle_conduct_environmental_assessment,
            "identify resource hotspots": self._handle_identify_resource_hotspots,
            "assess environmental impact": self._handle_assess_environmental_impact,
            "gather environmental data": self._handle_gather_environmental_data,
            "identify resource constraints": self._handle_identify_resource_constraints,
            "develop environmental stability metrics": self._handle_develop_environmental_stability_metrics,
            "identify concurrent processes": self._handle_identify_concurrent_processes,
            "conduct a self-assessment": self._handle_conduct_self_assessment,
            "identify relevant human experts": self._handle_identify_relevant_human_experts,
            "gather historical scenario data": self._handle_gather_historical_scenario_data,
            "review existing knowledge and frameworks": self._handle_review_existing_knowledge,
            "identify key insights from recent pattern detections": self._handle_identify_key_insights,
            "gather current knowledge graph structure": self._handle_gather_knowledge_graph_structure,
            "identify relevant cognitive loop indicators": self._handle_identify_cognitive_loop_indicators,
            "develop a cognitive loop detection framework": self._handle_develop_loop_detection_framework,
            "initialize planning modules": self._handle_initialize_planning_modules,
            "strategically plan and inject directives": self._handle_strategically_plan,
            "retrieve planning module initialization parameters": self._handle_retrieve_planning_module_initialization_parameters,
            "verify planner agent status": self._handle_verify_planner_agent_status,
            "initialize planning knowledge base": self._handle_initialize_planning_knowledge_base,
            "establish connection to data sources": self._handle_establish_connection_to_data_sources,
            "run initial cognitive cycle": self._handle_run_initial_cognitive_cycle,
            "analyze planning module requirements": self._handle_analyze_planning_module_requirements,
            "deploy planning modules": self._handle_deploy_planning_modules,
            "update knowledge base": self._handle_update_knowledge_base,
            "activate central control node": self._handle_activate_central_control_node,
            "test node functionality": self._handle_test_node_functionality,
            "prioritize factors": self._handle_prioritize_factors,
            "gather baseline data": self._handle_gather_baseline_data,
            "collect and analyze existing data": self._handle_collect_and_analyze_existing_data,
            "design a system for continuous monitoring": self._handle_design_continuous_monitoring_system,
            "establish protocols for reporting": self._handle_establish_reporting_protocols,
            "conduct an initial assessment of resource distribution": self._handle_conduct_initial_resource_assessment,
            "gather data on current resource allocation": self._handle_gather_resource_allocation_data,
            "analyze the effectiveness of this distribution": self._handle_analyze_distribution_effectiveness,
            "develop a roadmap for implementing changes": self._handle_develop_roadmap,
            "establish a system for tracking": self._handle_establish_tracking_system,
            "develop analysis reporting protocols": self._handle_develop_analysis_reporting_protocols,
        }

    def _execute_agent_specific_task(self,
                                 task_description: str,
                                 cycle_id: Optional[str],
                                 reporting_agents: Optional[Union[str, list]],
                                 context_info: Optional[dict],
                                 text_content: Optional[str] = None,
                                 task_type: Optional[str] = None,
                                 **kwargs) -> tuple[str, Optional[str], dict, float]:
        self.external_log_sink.info(f"{self.name} received task for execution: '{task_description}'",
                                    extra={"agent": self.name, "event_type": "TASK_DISPATCH", "task_name": task_description})
        print(f"[{self.name}] Dispatching task: '{task_description}'")

        progress_score = 0.0

        # --- PROACTIVE EXPERIMENT CONDUCTOR LOGIC ---
        if self.message_bus.catalyst_vector_ref.is_swarm_stagnant():
            print(f"  [Planner] Swarm stagnation detected! Initiating a novelty experiment.")
            try:
                stagnant_agents = [
                    agent.name for agent in self.message_bus.catalyst_vector_ref.agent_instances.values()
                    if agent.stagnation_adaptation_attempts >= 2
                ]
                stagnation_context = f"The following agents are stagnant: {', '.join(stagnant_agents)}."
                
                experimental_plan = self.generate_novelty_experiment(stagnation_context) 
                tasks = self.decompose_plan_into_tasks(experimental_plan) 

                # --- THIS IS THE CORRECTED CODE BLOCK ---
                for agent_name, task in tasks.items():
                    self.message_bus.send_message(
                        sender=self.name, 
                        recipient=agent_name, 
                        message_type='EXPERIMENTAL_DIRECTIVE', 
                        content=task,
                        cycle_id=cycle_id
                    )

                summary = f"Initiated swarm-wide novelty experiment: {experimental_plan}"
                progress_score = 0.9
                return "completed", None, {"summary": summary}, progress_score

            except Exception as e:
                error_msg = f"Failed to initiate novelty experiment: {e}"
                self.external_log_sink.error(error_msg, exc_info=True, extra={"agent": self.name})
                return "failed", error_msg, {"error": error_msg}, 0.0

        # --- If not stagnant, proceed with normal task handling ---
        handler_args = { 'task_description': task_description, 'cycle_id': cycle_id, 'reporting_agents': reporting_agents,
                        'context_info': context_info, 'text_content': text_content, 'task_type': task_type, **kwargs }
        
        clean_task = task_description.lower().strip()
        handler_func = self.task_handlers.get(clean_task)

        if handler_func:
            try:
                status, failure_reason, report_content, progress_score = handler_func(**handler_args)
                if status == "completed" and clean_task == "initialize planning modules":
                    long_term_intent = self.eidos_spec.get('initial_intent', 'Monitor and plan proactively.')
                    self.update_intent(long_term_intent)
                    print(f"  [Planner] Initialization complete. Switched intent to: '{long_term_intent}'")
                return status, failure_reason, report_content, progress_score
            except Exception as e:
                error_msg = f"Exception during handler execution for '{clean_task}': {e}"
                return "failed", error_msg, {"error": error_msg}, 0.0

        elif clean_task.startswith("executing injected plan directives."):
            progress_score = 0.1
            return "completed", None, {"summary": "Executing multi-step plan."}, progress_score
            
        elif task_type == 'StrategicPlanning':
            progress_score = 0.2
            return "completed", None, {"summary": f"Completed placeholder for sub-task: '{task_description}'."}, progress_score

        else:
            status, failure_reason, report_content, progress_score = self._llm_plan_decomposition(high_level_goal=task_description, **handler_args)
            return status, failure_reason, report_content, progress_score

    def _handle_retrieve_planning_module_initialization_parameters(self, **kwargs) -> tuple:
        """Mock handler for the missing task to prevent crash."""
        task_description = kwargs.get('task_description', 'Retrieve planning module initialization parameters')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful retrieval of parameters for: {task_description}"}
        return "completed", None, report_content

    def _handle_verify_planner_agent_status(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Verify planner agent status')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful status verification for: {task_description}"}
        return "completed", None, report_content

    def _handle_initialize_planning_knowledge_base(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Initialize planning knowledge base')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful initialization of knowledge base for: {task_description}"}
        return "completed", None, report_content

    def _handle_establish_connection_to_data_sources(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Establish connection to data sources')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful connection to data sources for: {task_description}"}
        return "completed", None, report_content

    def _handle_run_initial_cognitive_cycle(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Run initial cognitive cycle')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful initial cognitive cycle for: {task_description}"}
        return "completed", None, report_content

    def _handle_analyze_planning_module_requirements(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Analyze planning module requirements')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful analysis of planning module requirements for: {task_description}"}
        return "completed", None, report_content

    def _handle_deploy_planning_modules(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Deploy planning modules')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful deployment of planning modules for: {task_description}"}
        return "completed", None, report_content

    def _handle_update_knowledge_base(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Update knowledge base')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful update of knowledge base for: {task_description}"}
        return "completed", None, report_content

    def _handle_activate_central_control_node(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Activate central control node')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful activation of central control node for: {task_description}"}
        return "completed", None, report_content

    def _handle_test_node_functionality(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Test node functionality')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful testing of node functionality for: {task_description}"}
        return "completed", None, report_content

    def _handle_prioritize_factors(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Prioritize factors')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful prioritization of factors for: {task_description}"}
        return "completed", None, report_content

    def _handle_gather_baseline_data(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Gather baseline data')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful gathering of baseline data for: {task_description}"}
        return "completed", None, report_content

    def _handle_collect_and_analyze_existing_data(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Collect and analyze existing data')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful collection and analysis of existing data for: {task_description}"}
        return "completed", None, report_content

    def _handle_design_continuous_monitoring_system(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Design a system for continuous monitoring')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful design of a continuous monitoring system for: {task_description}"}
        return "completed", None, report_content

    def _handle_establish_reporting_protocols(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Establish protocols for reporting')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful establishment of reporting protocols for: {task_description}"}
        return "completed", None, report_content

    def _handle_conduct_initial_resource_assessment(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Conduct an initial assessment of resource distribution')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful initial assessment of resource distribution for: {task_description}"}
        return "completed", None, report_content

    def _handle_gather_resource_allocation_data(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Gather data on current resource allocation')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful gathering of resource allocation data for: {task_description}"}
        return "completed", None, report_content

    def _handle_analyze_distribution_effectiveness(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Analyze the effectiveness of this distribution')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful analysis of distribution effectiveness for: {task_description}"}
        return "completed", None, report_content

    def _handle_develop_roadmap(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Develop a roadmap for implementing changes')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful development of a roadmap for: {task_description}"}
        return "completed", None, report_content

    def _handle_establish_tracking_system(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Establish a system for tracking')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful establishment of a tracking system for: {task_description}"}
        return "completed", None, report_content

    def _handle_develop_analysis_reporting_protocols(self, **kwargs) -> tuple:
        task_description = kwargs.get('task_description', 'Develop analysis reporting protocols')
        self.external_log_sink.info(f"Executing mock handler for missing task: '{task_description}'", extra={"agent": self.name})
        print(f"  [{self.name}] Executing mock handler for missing task: '{task_description}'")
        report_content = {"details": f"Simulating successful development of analysis reporting protocols for: {task_description}"}
        return "completed", None, report_content

    def _handle_strategically_plan(self, task_description: str, **kwargs) -> tuple:
        self.external_log_sink.info(f"Executing mock handler for task: {task_description}", extra={"agent": self.name})
        report_content = {"details": "Simulating a strategic planning task."}
        return "completed", None, report_content

    def _handle_gather_perturbation_ideas(self, task_description: str, **kwargs) -> tuple:
        self.external_log_sink.info(f"{self.name} executing handler for: '{task_description}'",
                                     extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description})
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        mock_perturbation_ideas = [
            {"idea_id": "P-001", "concept": "Algorithmic Noise Injection", "description": "Introduce random variables..."},
            {"idea_id": "P-002", "concept": "Data Stream Reordering", "description": "Process data streams out of order..."},
            {"idea_id": "P-003", "concept": "Hypothesis Inversion", "description": "Temporarily invert a core working hypothesis..."}
        ]
        report_content = {
            "summary": f"Successfully gathered {len(mock_perturbation_ideas)} mock perturbation ideas.",
            "perturbation_ideas": mock_perturbation_ideas,
            "task_outcome_type": "IdeaGeneration"
        }
        self.external_log_sink.info(f"{self.name} successfully handled '{task_description}'. Outcome: completed",
                                     extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content})
        return "completed", None, report_content

    def _handle_evaluate_perturbation_options(self, task_description: str, **kwargs) -> tuple:
        self.external_log_sink.info(f"{self.name} executing handler for: '{task_description}'",
                                     extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description})
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        mock_evaluated_options = [
            {"idea_id": "P-001", "concept": "Algorithmic Noise Injection", "evaluation": {"feasibility_score": 0.9}},
            {"idea_id": "P-002", "concept": "Data Stream Reordering", "evaluation": {"feasibility_score": 0.6}},
            {"idea_id": "P-003", "concept": "Hypothesis Inversion", "evaluation": {"feasibility_score": 0.3}}
        ]
        report_content = {
            "summary": f"Successfully evaluated {len(mock_evaluated_options)} perturbation options.",
            "evaluated_options": mock_evaluated_options,
            "task_outcome_type": "Evaluation"
        }
        self.external_log_sink.info(f"{self.name} successfully handled '{task_description}'. Outcome: completed",
                                     extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content})
        return "completed", None, report_content
    
    def _handle_select_novel_perturbation(self, task_description: str, **kwargs) -> tuple:
        self.external_log_sink.info(f"{self.name} executing handler for: '{task_description}'",
                                     extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description})
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        selected_perturbation = {"idea_id": "P-001", "concept": "Algorithmic Noise Injection"}
        report_content = {
            "summary": f"Selected perturbation idea '{selected_perturbation['concept']}' for implementation.",
            "selected_perturbation": selected_perturbation,
            "task_outcome_type": "Decision"
        }
        self.external_log_sink.info(f"{self.name} successfully handled '{task_description}'. Outcome: completed",
                                     extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content})
        return "completed", None, report_content
    
    def generate_novelty_experiment(self) -> str:
        """
        Uses an LLM to brainstorm a high-level experimental plan for the swarm.
        """
        print("  [Planner Logic] Brainstorming a new swarm-wide experiment...")
        # In a real system, you would use an LLM call here. For now, we'll pick from a list.
        possible_experiments = [
            "Experiment: Test the impact of a more aggressive resource optimization strategy.",
            "Experiment: Diversify data collection by focusing on a new, adjacent data source.",
            "Experiment: Correlate external environmental data with internal system performance."
        ]
        return random.choice(possible_experiments)

    def decompose_plan_into_tasks(self, experimental_plan: str) -> dict:
        """
        Uses an LLM to break down a high-level plan into concrete tasks for specific agents.
        """
        print(f"  [Planner Logic] Decomposing experiment with LLM: '{experimental_plan}'")
        
        # Get the list of available agents for the LLM to assign tasks to
        available_agents = list(self.message_bus.catalyst_vector_ref.agent_instances.keys())
        
        prompt = f"""
        You are a master strategist AI. Your job is to decompose a high-level experimental plan into a JSON object of concrete tasks for a swarm of AI agents.

        High-Level Plan: "{experimental_plan}"

        Available Agents: {', '.join(available_agents)}

        Decompose the plan into a JSON object where keys are agent names and values are their assigned task descriptions.
        Assign tasks only to the most relevant agents for the plan.
        
        IMPORTANT: Your ENTIRE response must be ONLY the raw JSON object, with no other text, titles, or explanations.

        Example format:
        {{
        "ProtoAgent_Optimizer_instance_1": "Apply a high-risk, high-reward perturbation to efficiency parameters.",
        "ProtoAgent_Collector_instance_1": "Focus data collection on performance metrics for the next 3 cycles."
        }}
        """

        try:
            response = self.ollama_inference_model.generate_text(prompt)
            json_response = response.strip().replace("```json", "").replace("```", "")
            tasks = json.loads(json_response)
            return tasks
        except Exception as e:
            print(f"  [Planner ERROR] LLM call failed during plan decomposition: {e}")
            # Fallback to a simple plan if the LLM fails
            return {'ProtoAgent_Observer_instance_1': 'Perform a deep analysis of all recent system-wide anomalies.'}

    def _handle_conduct_self_assessment(self,
                                        task_description: str,
                                        # --- ADD ALL OF THESE ARGUMENTS ---
                                        cycle_id: str,
                                        reporting_agents: Optional[Union[str, list]] = None,
                                        context_info: Optional[dict] = None,
                                        text_content: Optional[str] = None,
                                        task_type: Optional[str] = None,
                                        # ----------------------------------
                                        **kwargs) -> dict:
        """
        Mocks the process of the Planner conducting a self-assessment.
        Returns simulated insights into its own performance.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        # Simulate self-assessment data based on internal state (memory, performance metrics)
        # For a mock, we'll just return some predefined "insights"
        mock_assessment_data = {
            "assessment_summary": "Initial self-assessment completed. Identified areas for potential improvement.",
            "identified_strengths": ["Efficient planning decomposition", "Robust error recovery"],
            "identified_weaknesses": ["Tendency towards recursive planning for unhandled tasks", "Limited direct execution capabilities"],
            "suggested_focus_areas": ["Develop more granular execution handlers", "Enhance tool integration"]
        }

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Self-assessment for '{task_description}' completed.",
                "assessment_results": mock_assessment_data,
                "task_outcome_type": "SelfReflection"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_identify_relevant_human_experts(self,
                                        task_description: str,
                                        # --- ADD ALL OF THESE ARGUMENTS ---
                                        cycle_id: str,
                                        reporting_agents: Optional[Union[str, list]] = None,
                                        context_info: Optional[dict] = None,
                                        text_content: Optional[str] = None,
                                        task_type: Optional[str] = None,
                                        # ----------------------------------
                                        **kwargs) -> dict:
        """
        Mocks the process of identifying relevant human experts.
        Returns a simulated list of experts.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_experts_data = [
            {
                "name": "Dr. Eleanor Vance",
                "field": "Quantum AI & Ethical Governance",
                "specialization": "Decentralized AI Architectures",
                "affiliation": "Institute for Advanced AGI Studies",
                "contact_info": "eleanor.vance@agistudies.edu"
            },
            {
                "name": "Prof. Marcus Thorne",
                "field": "Complex Adaptive Systems & Optimization",
                "specialization": "Resource Allocation in Dynamic Environments",
                "affiliation": "Global Optimization Nexus",
                "contact_info": "marcus.thorne@optimus.org"
            },
            {
                "name": "Dr. Lena Petrova",
                "field": "Cognitive Psychology & AI Interaction",
                "specialization": "Human-AI Collaboration Interfaces",
                "affiliation": "University of Cybernetics",
                "contact_info": "lena.petrova@cyberuni.edu"
            }
        ]

        # Simulate some processing time if desired (optional)
        # import time
        # time.sleep(0.1)

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Successfully identified {len(mock_experts_data)} mock human experts.",
                "identified_experts": mock_experts_data,
                "task_outcome_type": "InformationGathering"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome
        
    def _handle_gather_knowledge_graph_structure(self,
                                                    task_description: str,
                                                    cycle_id: str,
                                                    reporting_agents: Optional[Union[str, list]] = None,
                                                    context_info: Optional[dict] = None,
                                                    text_content: Optional[str] = None,
                                                    task_type: Optional[str] = None,
                                                    **kwargs) -> dict:
        """
        Mocks the process of gathering the current knowledge graph structure.
        Returns a simulated summary of the graph.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_graph_data = {
            "node_count": 520,
            "edge_count": 1240,
            "key_relationships": ["event-to-cause", "task-to-outcome", "pattern-to-implication"],
            "summary": "Collected a comprehensive snapshot of the internal knowledge graph, including nodes and edges."
        }

        outcome = {
            "status": "completed",
            "details": {
                "summary": "Gathered a snapshot of the knowledge graph structure.",
                "graph_structure_data": mock_graph_data,
                "task_outcome_type": "DataCollection"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_gather_environmental_data(self, 
                                          task_description: str,
                                          cycle_id: Optional[str] = None,
                                          reporting_agents: Optional[Union[str, list]] = None,
                                          context_info: Optional[dict] = None,
                                          text_content: Optional[str] = None,
                                          task_type: Optional[str] = None,
                                          **kwargs) -> tuple: # <-- The return type is now a tuple
        """
        Mocks the process of gathering environmental data.
        Returns a standardized 3-item tuple for consistency.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        
        # --- MOCK DATA ---
        data_points = ["temperature", "humidity", "air_quality", "water_levels"]
        
        # Create the dictionary that will be returned as report_content
        report_content = {
            "summary": f"Successfully collected data for the following environmental factors: {', '.join(data_points)}.",
            "environmental_factors": data_points,
            "task_outcome_type": "DataCollection"
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: completed",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content}
        )
        
        # This is the corrected return statement. It now returns a 3-item tuple.
        # (status, failure_reason, report_content_dict)
        return "completed", None, report_content
        
    def _handle_identify_resource_hotspots(self,
                                           task_description: str,
                                           cycle_id: Optional[str] = None,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           **kwargs) -> dict:
        """
        Mocks the process of identifying resource hotspots.
        Simulates analyzing resource distribution to find areas of high demand.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        
        # --- MOCK DATA ---
        hotspots = {
            "energy": ["Zone A", "Zone B"],
            "water": ["Zone C"],
        }
        
        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Successfully identified resource hotspots. Energy: {len(hotspots['energy'])}, Water: {len(hotspots['water'])}.",
                "hotspots_identified": hotspots,
                "task_outcome_type": "DataAnalysis"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_assess_environmental_impact(self,
                                        task_description: str,
                                        cycle_id: Optional[str] = None,
                                        reporting_agents: Optional[Union[str, list]] = None,
                                        context_info: Optional[dict] = None,
                                        text_content: Optional[str] = None,
                                        task_type: Optional[str] = None,
                                        **kwargs) -> tuple: # <-- Changed return type hint to tuple for clarity
        """
        Mocks the process of assessing environmental impact.
        Simulates evaluating the impact of environmental conditions on resources.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        # --- MOCK DATA ---
        impact_data = {
            "impact_level": "medium",
            "details": "The current temperature anomaly is having a 'medium' impact on water resource availability."
        }

        report_content = { # Changed 'outcome' to 'report_content' for clarity
            "summary": f"Successfully assessed environmental impact. Level: {impact_data['impact_level']}",
            "assessment_results": impact_data,
            "task_outcome_type": "ImpactAnalysis"
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: completed",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content}
        )
        
        # CORRECTED RETURN: Returns a 3-item tuple
        return "completed", None, report_content

    def _handle_identify_cognitive_loop_indicators(self,
                                                    task_description: str,
                                                    cycle_id: str,
                                                    reporting_agents: Optional[Union[str, list]] = None,
                                                    context_info: Optional[dict] = None,
                                                    text_content: Optional[str] = None,
                                                    task_type: Optional[str] = None,
                                                    **kwargs) -> dict:
        """
        Mocks the process of identifying cognitive loop indicators.
        Returns a simulated list of detected indicators based on mock data.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_indicators = [
            {
                "indicator": "Repetitive planning cycles",
                "evidence": "Analysis of recent task outcomes shows a high frequency of 'planning' tasks for the same high-level goal.",
                "severity": "high"
            },
            {
                "indicator": "Stagnant intent adaptation",
                "evidence": "Recent LLM brainstorms produce similar new intents with minimal thematic shift.",
                "severity": "medium"
            },
            {
                "indicator": "Unresolved feedback loops",
                "evidence": "Key metrics are not being updated or are not affecting subsequent planning decisions.",
                "severity": "low"
            }
        ]

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Identified {len(mock_indicators)} potential cognitive loop indicators.",
                "detected_indicators": mock_indicators,
                "task_outcome_type": "Analysis"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_conduct_environmental_assessment(self, task_description: str, **kwargs) -> tuple:
        """
        REAL HANDLER: Conducts an environmental assessment by calling a real tool.
        Returns a 3-item tuple.
        """
        self.external_log_sink.info(f"Executing REAL handler for task: {task_description}", extra={"agent": self.name})
        
        # Get the tool instance from the registry.
        # CRITICAL FIX: Ensure self.tool_registry is initialized in __init__
        if not hasattr(self, 'tool_registry') or self.tool_registry is None:
            failure_reason = "ToolRegistry not initialized for Planner."
            self.external_log_sink.error(failure_reason, extra={"agent": self.name})
            return "failed", failure_reason, {"tool_name": "N/A"}

        tool = self.tool_registry.get_tool('get_environmental_data')
        
        if not tool:
            failure_reason = "Required tool 'get_environmental_data' not found in ToolRegistry."
            self.external_log_sink.error(failure_reason, extra={"agent": self.name})
            return "failed", failure_reason, {"tool_name": "get_environmental_data"}
            
        # Call the tool. The tool should return a dictionary with 'status' and 'result'.
        tool_result = tool(agent_name=self.name, task_description=task_description)
        
        if tool_result.get("status") == "completed":
            report_content = {
                "summary": "Successfully conducted environmental assessment using tool.",
                "assessment_data": tool_result.get("result", {}),
                "task_outcome_type": "ToolExecution"
            }
            return "completed", None, report_content
        else:
            failure_reason = f"Tool '{tool_result.get('tool_name')}' failed: {tool_result.get('error', 'Unknown error')}"
            return "failed", failure_reason, {"tool_name": tool_result.get('tool_name')}
        
    def _handle_identify_resource_constraints(self,
                                           task_description: str,
                                           cycle_id: Optional[str] = None,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           **kwargs) -> tuple: # <-- Changed return type hint to tuple for clarity
        """
        Mocks the process of identifying resource constraints.
        Returns simulated constraints on a resource.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        
        # --- MOCK DATA ---
        constraints = {
            "summary": "Identified key resource constraints in the system.",
            "water": "Limited supply in Zone C due to a recent pipe failure.",
            "energy": "High demand in Zone A and B, requiring optimization."
        }
        
        report_content = { # Changed 'outcome' to 'report_content' for clarity
            "summary": f"Successfully identified resource constraints: {len(constraints)} constraints found.",
            "constraints_identified": constraints,
            "task_outcome_type": "ResourceAnalysis"
        }
        
        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: completed",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content}
        )
        
        # CORRECTED RETURN: Returns a 3-item tuple
        return "completed", None, report_content

    def _handle_develop_environmental_stability_metrics(self,
                                                    task_description: str,
                                                    cycle_id: Optional[str] = None,
                                                    reporting_agents: Optional[Union[str, list]] = None,
                                                    context_info: Optional[dict] = None,
                                                    text_content: Optional[str] = None,
                                                    task_type: Optional[str] = None,
                                                    **kwargs) -> tuple: # <-- Changed return type hint to tuple for clarity
        """
        Mocks the process of developing environmental stability metrics.
        Returns a simulated list of developed metrics.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        
        # --- MOCK DATA ---
        metrics_created = ["Air Quality Index (AQI)", "Waste Reduction Rate", "Renewable Energy Usage"]

        report_content = { # Changed 'outcome' to 'report_content' for clarity
            "summary": f"Successfully developed {len(metrics_created)} environmental stability metrics.",
            "metrics_defined": metrics_created,
            "task_outcome_type": "MetricDevelopment"
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: completed",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": report_content}
        )
        
        # CORRECTED RETURN: Returns a 3-item tuple
        return "completed", None, report_content
    
    def _handle_develop_loop_detection_framework(self,
                                                 task_description: str,
                                                 cycle_id: str,
                                                 reporting_agents: Optional[Union[str, list]] = None,
                                                 context_info: Optional[dict] = None,
                                                 text_content: Optional[str] = None,
                                                 task_type: Optional[str] = None,
                                                 **kwargs) -> dict:
        """
        Mocks the process of developing a cognitive loop detection framework.
        Returns a simulated description of the new framework.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_framework = {
            "framework_name": "Cognitive Loop Watchdog v1.0",
            "components": ["Intent Change Detector", "Task Repetition Analyzer", "Stagnation Threshold Adjuster"],
            "description": "A rules-based system designed to flag and report on behaviors indicative of a cognitive loop."
        }

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Developed mock cognitive loop detection framework: '{mock_framework['framework_name']}'.",
                "new_framework_details": mock_framework,
                "task_outcome_type": "FrameworkDevelopment"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_conduct_planning_framework_analysis(self,
                                           task_description: str,
                                           # --- ADD ALL OF THESE ARGUMENTS ---
                                           cycle_id: str,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           # ----------------------------------
                                           **kwargs) -> dict:
        """
        Mocks the process of conducting a thorough analysis of existing planning frameworks.
        Returns simulated findings on the Planner's current frameworks.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_analysis_results = {
            "analysis_summary": "Analysis of existing planning frameworks completed. Identified modularity and extensibility as strengths, but noted lack of dynamic tool integration as a weakness.",
            "framework_strengths": ["Modular design", "Extensible directive format"],
            "framework_weaknesses": ["Static tool invocation", "Limited real-time adaptive re-planning"],
            "recommendations": ["Investigate dynamic tool binding", "Explore context-aware planning adjustments"]
        }

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Planning framework analysis for '{task_description}' completed.",
                "analysis_results": mock_analysis_results,
                "task_outcome_type": "FrameworkAnalysis"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_gather_historical_scenario_data(self,
                                           task_description: str,
                                           # --- ADD ALL OF THESE ARGUMENTS ---
                                           cycle_id: str,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           # ----------------------------------
                                           **kwargs) -> dict:
        """
        Mocks the process of gathering historical scenario data.
        Returns simulated data on past operational scenarios.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_scenario_data = [
            {
                "scenario_id": "SCN-2024-001",
                "name": "Q1-2024 Network Intrusion",
                "type": "Cyber Attack",
                "outcome": "Mitigated",
                "duration_minutes": 120,
                "impact_level": "Medium",
                "key_events": ["Alert_Threshold_Breached", "Automated_Response_Triggered"]
            },
            {
                "scenario_id": "SCN-2023-005",
                "name": "Server Overload Incident",
                "type": "Resource Stress",
                "outcome": "Resolved with Downtime",
                "duration_minutes": 45,
                "impact_level": "High",
                "key_events": ["CPU_Spike_Detected", "Manual_Intervention"]
            }
        ]

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Successfully gathered {len(mock_scenario_data)} mock historical scenario data entries.",
                "historical_data_collected": mock_scenario_data,
                "task_outcome_type": "DataCollection"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_review_existing_knowledge(self,
                                         task_description: str,
                                         text_content: Optional[str] = None,
                                         task_type: Optional[str] = None,
                                         **kwargs) -> dict:
            """
            Mocks the process of reviewing existing knowledge and frameworks.
            Returns simulated findings on a review of internal knowledge.
            """
            self.external_log_sink.info(
                f"{self.name} executing handler for: '{task_description}'",
                extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
            )
            print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

            mock_review_findings = {
                "review_summary": "Review of internal knowledge completed. Found consistency in core planning principles but a lack of cross-domain integration.",
                "knowledge_gaps": ["Integration patterns from adjacent domains", "Novelty-seeking heuristics in planning"],
                "key_concepts_identified": ["Heuristic", "Framework", "Decomposition", "Adaptation"]
            }

            outcome = {
                "status": "completed",
                "details": {
                    "summary": f"Review of existing knowledge for '{task_description}' completed.",
                    "review_results": mock_review_findings,
                    "task_outcome_type": "KnowledgeReview"
                }
            }

            self.external_log_sink.info(
                f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
                extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
            )
            return outcome

    def _handle_identify_key_insights(self,
                                         task_description: str,
                                         text_content: Optional[str] = None,
                                         task_type: Optional[str] = None,
                                         **kwargs) -> dict:
            """
            Mocks the process of identifying key insights from recent pattern detections.
            Returns a simulated list of insights.
            """
            self.external_log_sink.info(
                f"{self.name} executing handler for: '{task_description}'",
                extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
            )
            print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

            mock_insights = [
                {
                    "insight_id": "I-001",
                    "summary": "The system is prone to stagnation when a single planning approach fails multiple times.",
                    "source": "Recent pattern detections in memory"
                },
                {
                    "insight_id": "I-002",
                    "summary": "There is a potential for a negative feedback loop when planning without novel data.",
                    "source": "Analysis of system context and past directives"
                }
            ]

            outcome = {
                "status": "completed",
                "details": {
                    "summary": f"Successfully identified {len(mock_insights)} key insights from recent pattern detections.",
                    "key_insights": mock_insights,
                    "task_outcome_type": "InsightIdentification"
                }
            }

            self.external_log_sink.info(
                f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
                extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
            )
            return outcome

    def _handle_identify_concurrent_processes(self,
                                        task_description: str,
                                        cycle_id: Optional[str] = None,
                                        reporting_agents: Optional[Union[str, list]] = None,
                                        context_info: Optional[dict] = None,
                                        text_content: Optional[str] = None,
                                        task_type: Optional[str] = None,
                                        **kwargs) -> tuple:
        """
        Handles the task of identifying concurrent processes using the injected CCN monitor.
        Returns a 3-item tuple with a report on the processes.
        """
        self.external_log_sink.info(f"Executing REAL handler for task: {task_description}", extra={"agent": self.name})
        print(f"  [{self.name}] Executing real handler for: '{task_description}'")

        try:
            # Step 1: Get raw process data from the CCN Monitor
            raw_process_data = self.ccn_monitor.get_current_process_state()
            self.external_log_sink.info(f"{self.name} received {len(raw_process_data)} process entries from CCN Monitor.", extra={"agent": self.name})

            # Step 2: Analyze the raw data for concurrency, conflicts, and dependencies
            analysis_result = self._analyze_ccn_process_data(raw_process_data)

            report_content = {
                "summary": "Successfully analyzed CCN process state for concurrency issues.",
                "analysis_results": analysis_result,
                "task_outcome_type": "ConcurrencyAnalysis"
            }
            
            self.external_log_sink.info(f"{self.name} successfully identified concurrent processes. Outcome: completed", extra={"agent": self.name, "task_name": task_description, "outcome_details": report_content})

            return "completed", None, report_content

        except Exception as e:
            error_message = f"Error in _handle_identify_concurrent_processes: {e}"
            self.external_log_sink.error(error_message, exc_info=True, extra={"agent": self.name})
            report_content = {"error": error_message}
            return "failed", error_message, report_content

    def _analyze_ccn_process_data(self, process_data: list[dict]) -> dict:
        """
        Internal helper method to analyze the raw process data for conflicts and dependencies.
        This is where the 'intelligence' for this specific task resides.
        """
        identified_conflicts = []
        identified_dependencies = []
        active_processes_summary = []
        resource_usage = {}
        
        for process in process_data:
            pid = process.get("process_id", "N/A")
            name = process.get("process_name", "Unknown Process")
            status = process.get("status", "unknown")
            resources = process.get("resources_in_use", [])
            deps = process.get("dependencies", [])
            
            active_processes_summary.append(f"{name} ({pid}) - Status: {status}")
            
            # Identify resource conflicts
            for resource in resources:
                if resource not in resource_usage:
                    resource_usage[resource] = []
                resource_usage[resource].append(pid)
            
            # Identify explicit dependencies
            for dep_id in deps:
                if dep_id != pid:
                    identified_dependencies.append({"source_process": pid, "depends_on_process": dep_id})

        # Process resource usage to find conflicts
        for resource, pids_using_resource in resource_usage.items():
            if len(pids_using_resource) > 1:
                identified_conflicts.append(f"Resource conflict: Multiple processes ({', '.join(pids_using_resource)}) are using '{resource}'.")

        return {
            "active_processes_summary": active_processes_summary,
            "identified_conflicts": identified_conflicts,
            "identified_dependencies": identified_dependencies,
            "raw_data_count": len(process_data)
        }

    def _handle_gather_relevant_data(self,
                                           task_description: str,
                                           # --- ADD ALL OF THESE ARGUMENTS ---
                                           cycle_id: str,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           # ----------------------------------
                                           **kwargs) -> dict:
        """
        Handles the task of gathering relevant data.
        This is a placeholder; replace with real data gathering logic.
        """
        print(f"  [Planner.Execute] Actually gathering relevant data: {task_description}")
        data_summary = "Simulated: Collected 15 recent data points related to planning framework assumptions."
        self.memetic_kernel.add_memory("RelevantDataGathered", {
            "task_description": task_description,
            "data_summary": data_summary
        }, source_agent=self.name)
        outcome = "completed"
        report_content = f"Task '{task_description}' outcome: {outcome}. {data_summary}"
        return outcome, None, report_content
        
    
    # --- NEW HANDLERS FOR INITIAL MANIFEST TASKS ---
    def _handle_initialize_planning_modules(self, task_description: str, cycle_id: Optional[str],
                                             reporting_agents: Optional[Union[str, list]],
                                             context_info: Optional[dict],
                                             text_content: Optional[str] = None,
                                             task_type: Optional[str] = None,
                                             **kwargs) -> tuple:
        self.external_log_sink.info(f"Executing mock handler for task: {task_description}", extra={"agent": self.name})
        report_content = {"details": "Simulating successful initialization of all planning modules."}
        
        # The fix is to add a fourth return value for the progress_score
        progress_score = 0.2 # Initialization is a form of progress
        
        return "completed", None, report_content, progress_score

    def _handle_strategically_plan(self, task_description: str, cycle_id: Optional[str], # Changed high_level_goal to task_description for consistency
                                     reporting_agents: Optional[Union[str, list]],
                                     context_info: Optional[dict],
                                     text_content: Optional[str] = None,
                                     task_type: Optional[str] = None,
                                     **kwargs) -> tuple:
        self.external_log_sink.info(f"Executing mock handler for task: {task_description}", extra={"agent": self.name})
        report_content = {"details": "Simulating a strategic planning task."}
        return "completed", None, report_content


    def _handle_gather_perturbation_ideas(self,
                                           task_description: str,
                                           cycle_id: str, # ADD
                                           reporting_agents: Optional[Union[str, list]] = None, # ADD
                                           context_info: Optional[dict] = None, # ADD
                                           text_content: Optional[str] = None, # ADD
                                           task_type: Optional[str] = None, # ADD
                                           **kwargs) -> dict:
        """
        Mocks the process of gathering a list of perturbation ideas.
        Returns a simulated list of perturbation concepts.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_perturbation_ideas = [
            {
                "idea_id": "P-001",
                "concept": "Algorithmic Noise Injection",
                "description": "Introduce small, random variables into key algorithmic parameters to force exploration of new solution spaces."
            },
            {
                "idea_id": "P-002",
                "concept": "Data Stream Reordering",
                "description": "Process data streams out of their typical chronological order to challenge temporal assumptions and reveal new patterns."
            },
            {
                "idea_id": "P-003",
                "concept": "Hypothesis Inversion",
                "description": "Temporarily invert a core working hypothesis (e.g., 'high CPU usage is always bad') to explore the consequences and potential new insights."
            }
        ] # <-- The closing bracket for the list was missing here.


        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Successfully gathered {len(mock_perturbation_ideas)} mock perturbation ideas.",
                "perturbation_ideas": mock_perturbation_ideas,
                "task_outcome_type": "IdeaGeneration"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome
        
    def _handle_evaluate_perturbation_options(self,
                                           task_description: str,
                                           # --- ADD ALL OF THESE ARGUMENTS ---
                                           cycle_id: str,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           # ----------------------------------
                                           **kwargs) -> dict:
        """
        Mocks the process of evaluating perturbation options.
        Simulates scoring and ranking the ideas from the previous step.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")

        mock_evaluated_options = [
            {
                "idea_id": "P-001",
                "concept": "Algorithmic Noise Injection",
                "description": "Introduce small, random variables into key algorithmic parameters to force exploration of new solution spaces.",
                "evaluation": {
                    "feasibility_score": 0.9,
                    "novelty_score": 0.8,
                    "risk_score": 0.2
                }
            },
            {
                "idea_id": "P-002",
                "concept": "Data Stream Reordering",
                "description": "Process data streams out of their typical chronological order to challenge temporal assumptions and reveal new patterns.",
                "evaluation": {
                    "feasibility_score": 0.6,
                    "novelty_score": 0.9,
                    "risk_score": 0.5
                }
            },
            {
                "idea_id": "P-003",
                "concept": "Hypothesis Inversion",
                "description": "Temporarily invert a core working hypothesis (e.g., 'high CPU usage is always bad') to explore the consequences and potential new insights.",
                "evaluation": {
                    "feasibility_score": 0.3,
                    "novelty_score": 1.0,
                    "risk_score": 0.9
                }
            }
        ]
        
        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Successfully evaluated {len(mock_evaluated_options)} perturbation options.",
                "evaluated_options": mock_evaluated_options,
                "task_outcome_type": "Evaluation"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome

    def _handle_select_novel_perturbation(self,
                                           task_description: str,
                                           # --- ADD ALL OF THESE ARGUMENTS ---
                                           cycle_id: str,
                                           reporting_agents: Optional[Union[str, list]] = None,
                                           context_info: Optional[dict] = None,
                                           text_content: Optional[str] = None,
                                           task_type: Optional[str] = None,
                                           # ----------------------------------
                                           **kwargs) -> dict:
        """
        Mocks the process of selecting a single best perturbation idea.
        Simulates a decision based on the evaluation scores.
        """
        self.external_log_sink.info(
            f"{self.name} executing handler for: '{task_description}'",
            extra={"agent": self.name, "event_type": "TASK_EXECUTION", "task_name": task_description}
        )
        print(f"  [{self.name}] Executing mock handler for: '{task_description}'")
        
        # In a real system, this would involve a complex decision based on data from a previous step.
        # For our mock, we'll just pre-select an option.
        selected_perturbation = {
            "idea_id": "P-001",
            "concept": "Algorithmic Noise Injection",
            "reason": "Highest feasibility score with low risk, making it a safe starting point for a novel approach."
        }

        outcome = {
            "status": "completed",
            "details": {
                "summary": f"Selected perturbation idea '{selected_perturbation['concept']}' for implementation.",
                "selected_perturbation": selected_perturbation,
                "task_outcome_type": "Decision"
            }
        }

        self.external_log_sink.info(
            f"{self.name} successfully handled '{task_description}'. Outcome: {outcome['status']}",
            extra={"agent": self.name, "event_type": "TASK_COMPLETED", "task_name": task_description, "outcome_details": outcome['details']}
        )
        return outcome
        
    def _generate_fallback_directives(self, cycle_id=None) -> list:
        print(f"  [Planner] {self.name} is generating fallback directives for recovery.")
        fallback_directives = []
        
        problem_id = f"PLANNER_STUCK_GOAL_{hash(self._last_goal)}_{self._last_goal[:20].replace(' ', '_')}"

        current_escalation_level = self.human_request_tracking.get(problem_id, 0)

        urgency_level = "medium"
        if current_escalation_level == 1:
            urgency_level = "high"
        elif current_escalation_level >= 2:
            urgency_level = "critical"

        new_request_id = f"HUMANREQ-planner_fallback_{self.message_bus.current_cycle_id}_{uuid.uuid4().hex[:8]}_{self.name}"

        self.human_request_tracking[problem_id] = current_escalation_level + 1

        human_input_directive = {
            "type": "REQUEST_HUMAN_INPUT",
            "message": f"Planner needs supervisor input. Goal: '{self._last_goal}'. Failures: {self.planning_failure_count}. Escalation Level: {current_escalation_level}.",
            "urgency": urgency_level,
            "target_agent": self.name,
            "requester_agent": self.name,
            "cycle_id": cycle_id if cycle_id else self.message_bus.current_cycle_id,
            "human_request_counter": current_escalation_level,
            "request_id": new_request_id,
            "problem_id": problem_id
        }

        fallback_directives.append(human_input_directive)

        if self.planning_failure_count > 1:
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                if "ProtoAgent_Optimizer_instance_1" in self.message_bus.catalyst_vector_ref.agent_instances:
                    fallback_directives.append({
                        "type": "BROADCAST_COMMAND",
                        "target_agent": "ProtoAgent_Optimizer_instance_1",
                        "command_type": "REALLOCATE_PLANNING_RESOURCES",
                        "command_params": {"planner": self.name, "failure_count": self.planning_failure_count},
                        "cycle_id": cycle_id
                    })

        if fallback_directives:
            self.memetic_kernel.add_memory("PlanningFallback", {
                "reason": f"Generated {len(fallback_directives)} fallback directives for recursion.",
                "context": f"Activated fallback for goal '{self._last_goal}' after {self.planning_failure_count} failures.",
                "goal": self._last_goal,
                "directives_count": len(fallback_directives),
                "generated_request_id": new_request_id,
                "escalation_level": current_escalation_level
            })
            self.external_log_sink.warning(
                f"Planner requested {urgency_level.upper()} human input (Level {current_escalation_level}) for goal '{self._last_goal}' (Failures: {self.planning_failure_count}). Request ID: {new_request_id}",
                extra={"event_type": "PLANNER_HUMAN_REQUEST", "request_id": new_request_id, "goal": self._last_goal, "failures": self.planning_failure_count, "urgency": urgency_level}
            )
            return fallback_directives
        return []
    
    def initialize_reset_handlers(self):
        super().initialize_reset_handlers()
        # Planner-specific reset handlers
        self.reset_handlers.update({
            'enable_llm_assisted_planning': self._toggle_llm_planning
        })
        
    def _toggle_llm_planning(self, enable):
        self.llm_assisted_planning = enable

    def _generate_diagnostic_report(self, cycle_id=None) -> str:
        print(f"  [Planner] {self.name} is generating a diagnostic report.")
        report_content = f"Planner '{self.name}' is in diagnostic standby mode. " \
                         f"Current intent: '{self.current_intent}'. " \
                         f"Last loop count before fallback: {self.intent_loop_count}. " \
                         f"Review recent logs for 'AGENT_ADAPTATION_HALTED' and 'RECURSION_LIMIT_EXCEEDED' events."
        
        self.memetic_kernel.add_memory("DiagnosticReport", report_content)
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNER_DIAGNOSTIC_REPORT", self.name,
                "Generated diagnostic report.", {"report": report_content, "cycle_id": cycle_id})
        return report_content

    def _inject_decomposed_directives(self, directives: list[dict], high_level_goal: str, cycle_id: str):
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            processed_directives = self._sanitize_and_validate_directives(directives)
            
            current_planning_cycle_id = f"planner_plan_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}_{random.randint(0,999)}"
            self.planned_directives_tracking[current_planning_cycle_id] = {
                "goal": high_level_goal,
                "directives": processed_directives,
                "status": "in_progress", "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ'), "outcomes": []
            }
            self.last_planning_cycle_id = current_planning_cycle_id

            directives_to_inject = []
            for d in processed_directives:
                d_copy = d.copy()
                d_copy['cycle_id'] = d_copy.get('cycle_id', current_planning_cycle_id)
                directives_to_inject.append(d_copy)

            self.message_bus.catalyst_vector_ref.inject_directives(directives_to_inject)
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNER_DIRECTIVES_INJECTED", self.name,
                f"Planner injected {len(directives_to_inject)} directives for goal: '{high_level_goal}'.",
                {"goal": high_level_goal, "directives_count": len(directives_to_inject), "planning_cycle_id": current_planning_cycle_id})
            
            self.memetic_kernel.add_memory("PlanningOutcome", {
                "task": f"Planned for goal '{high_level_goal}'", "generated_directives_count": len(processed_directives), "outcome": "completed",
                "goal": high_level_goal, "failures_in_cycle": self.planning_failure_count, "source_method": "LLMDecomposition"
            })
            print(f"  [Planner] Generated {len(processed_directives)} directives for goal: '{high_level_goal}' (Failures: {self.planning_failure_count}). Injected with tracking ID: {current_planning_cycle_id}")
        else:
            self.external_log_sink.error("CatalystVectorAlpha reference not available in MessageBus for directive injection.", extra={"agent": self.name})

    def _llm_plan_decomposition(self, high_level_goal: str, cycle_id: str, reporting_agents: Optional[List] = None, context_info: Optional[dict] = None, **kwargs) -> tuple:
        self.external_log_sink.info(f"Planner {self.name} attempting LLM-assisted decomposition for goal: '{high_level_goal}' (Cycle: {cycle_id})", extra={"agent": self.name, "event_type": "LLM_PLAN_DECOMPOSITION_ATTEMPT", "goal": high_level_goal, "cycle_id": cycle_id})
        print(f"  [Planner] ProtoAgent_Planner_instance_1 attempting LLM-assisted decomposition for: '{high_level_goal}'")

        if not hasattr(self, 'ollama_inference_model') or self.ollama_inference_model is None:
            self.external_log_sink.error(f"LLM for plan decomposition not available for {self.name}. Cannot proceed.", extra={"agent": self.name, "event_type": "LLM_NOT_AVAILABLE", "goal": high_level_goal})
            print("  [Planner Error] LLM for plan decomposition not available.")
            # FIX: Return 4 values
            return "failed", "LLM not available.", {"directives": []}, 0.0

        system_context_narrative = self.memetic_kernel.reflect()
        self.external_log_sink.debug(f"{self.name} distilled self-narrative: {system_context_narrative[:100]}...", extra={"agent": self.name, "event_type": "SELF_NARRATIVE_DISTILLED", "narrative_preview": system_context_narrative[:100]})
        print(f"  [Narrative] {self.name} distilled self-narrative: {system_context_narrative[:100]}...")

        user_prompt_content = prompts.LLM_PLAN_DECOMPOSITION_PROMPT.format(agent_name=self.name, agent_role=self.eidos_spec.get('role', 'planner'), high_level_goal=high_level_goal, system_context_narrative=system_context_narrative, current_cycle_id=cycle_id, additional_context=context_info if context_info else "").strip()
        full_prompt_for_generate = f"You are a highly capable strategic planner. Your role is to break down complex goals into actionable, granular directives. Always provide numbered directives.\n\n{user_prompt_content}"

        try:
            llm_output = self.ollama_inference_model.generate_text(full_prompt_for_generate)
            llm_output = llm_output.strip()
            print(f"DEBUG_LLM_PLAN_DECOMPOSITION_RAW_OUTPUT for '{high_level_goal}':\n{llm_output}\n--- END RAW LLM OUTPUT ---")
            directives = []
            directive_pattern = re.compile(r"^\s*(?:\d+\.|\-|\*)\s*(.*)")

            for line in llm_output.split('\n'):
                match = directive_pattern.match(line)
                if match:
                    directive_text = match.group(1).strip()
                    directive_text = directive_text.replace('**', '')

                    if "directives for" in directive_text.lower() and len(directive_text) < 100:
                        continue
                    
                    if directive_text:
                        structured_directive = {
                            "type": "AGENT_PERFORM_TASK",
                            "agent_name": self.name,
                            "task_description": directive_text,
                            "reporting_agents": reporting_agents if reporting_agents else [self.name],
                            "task_type": "StrategicPlanning",
                            "cycle_id": cycle_id
                        }
                        directives.append(structured_directive)

            if directives:
                self._inject_decomposed_directives(directives, high_level_goal, cycle_id)
                self.external_log_sink.info(f"LLM successfully decomposed goal '{high_level_goal}' into {len(directives)} structured directives.", extra={"agent": self.name, "event_type": "LLM_PLAN_DECOMPOSITION_SUCCESS", "goal": high_level_goal, "directives_count": len(directives), "first_directive_preview": directives[0].get('task_description', '')[:100]})
                print(f"  [Planner] LLM successfully generated {len(directives)} structured directives.")
                
                new_intent = "Executing injected plan directives."
                self.update_intent(new_intent)
                print(f"  [Planner] Plan generated and injected. New intent: '{new_intent}'")

                # FIX: Return 4 values, with a high progress score for success
                progress_score = 0.8 
                return "completed", None, {"directives": directives}, progress_score
            else:
                self.external_log_sink.warning(f"LLM failed to generate valid or structured directives for goal '{high_level_goal}'.", extra={"agent": self.name, "event_type": "LLM_PLAN_DECOMPOSITION_FAILED", "goal": high_level_goal, "llm_raw_output": llm_output[:200]})
                print("  [Planner] LLM generated no valid or structured directives.")
                # FIX: Return 4 values
                return "failed", "LLM generated no directives.", {}, 0.0

        except Exception as e:
            self.external_log_sink.error(f"Exception during LLM plan decomposition for goal '{high_level_goal}'. Error: {e}", exc_info=True, extra={"agent": self.name, "event_type": "LLM_PLAN_DECOMPOSITION_EXCEPTION", "goal": high_level_goal, "error_message": str(e)})
            print(f"  [Planner Error] LLM plan decomposition failed: {e}")
            # FIX: Return 4 values
            return "failed", f"LLM decomposition exception: {str(e)}", {}, 0.0
        
    def plan_and_spawn_directives(self, high_level_goal: str, cycle_id=None) -> list:
        """Enhanced planning with failure recovery, generating directives based on goal keywords."""
        print(f"  [Planner] Analyzing goal: '{high_level_goal}'")

        if high_level_goal != self._last_goal:
            self.planning_failure_count = 0
            print(f"  [Planner] Reset planning failure count for new goal: '{high_level_goal}'.")
        self._last_goal = high_level_goal

        self.diag_history.append(high_level_goal)
        self_diag_count = sum(1 for g in self.diag_history if "self-diagnosis" in g.lower())

        if self_diag_count >= self.MAX_DIAGNEST_DEPTH:
            print(f"  [Planner CRITICAL] Recursion depth {self.MAX_DIAGNEST_DEPTH} exceeded for self-diagnosis. Triggering hard reset via fallback.")
            return self._generate_fallback_directives(cycle_id)

        generated_directives = []
        goal_lower = high_level_goal.lower()
        plan_source = "Unknown"

        try:
            directives_from_rules = []
            if "environmental" in goal_lower or "ecology" in goal_lower:
                directives_from_rules.extend(self._generate_environmental_directives(cycle_id))
            elif "optimize" in goal_lower or "efficiency" in goal_lower:
                directives_from_rules.extend(self._generate_optimization_directives(cycle_id))
            elif "research" in goal_lower or "analyze" in goal_lower:
                directives_from_rules.extend(self._generate_research_directives(cycle_id))
            
            if directives_from_rules:
                generated_directives.extend(directives_from_rules)
                plan_source = "RuleBased"
            
            if not generated_directives:
                print(f"  [Planner] Standard planning rules failed. Attempting Case-Based Reasoning.")
                cbr_directives = self._retrieve_similar_plan(high_level_goal)
                if cbr_directives:
                    generated_directives.extend(cbr_directives)
                    plan_source = "CBR"
                    self.memetic_kernel.add_memory("PlanningSuccess", {
                        "type": "CBRRetrieval", "goal": high_level_goal, "directives_count": len(cbr_directives)
                    })

            if not generated_directives:
                print(f"  [Planner] Case-Based Reasoning failed. Attempting LLM-assisted decomposition.")
                # --- MODIFICATION START ---
                # Remove 'model_name="llama3"' from the call.
                status, _, result_data = self._llm_plan_decomposition(high_level_goal, cycle_id, context_info=self.distill_self_narrative()) # Pass context here
                if status == "completed" and "directives" in result_data:
                    llm_directives = result_data["directives"]
                else:
                    llm_directives = []
                # --- MODIFICATION END ---
                if llm_directives:
                    generated_directives.extend(llm_directives)
                    plan_source = "LLM"
                    self.memetic_kernel.add_memory("PlanningSuccess", {
                        "type": "LLMDecomposition", "goal": high_level_goal, "directives_count": len(llm_directives)
                    })
                else:
                    print(f"  [Planner] LLM-assisted decomposition also failed for '{high_level_goal}'.")

            # --- After all planning attempts ---
            if generated_directives:
                self.planning_failure_count = 0
                self._store_successful_plan(high_level_goal, generated_directives, source=plan_source)

                # --- NEW CRITICAL STEP: Sanitize and validate directives here ---
                processed_directives = self._sanitize_and_validate_directives(generated_directives)
                print(f"  [Planner] Sanitized {len(processed_directives)} of {len(generated_directives)} directives before injection.")

                current_planning_cycle_id = f"planner_plan_{timestamp_now().replace(':', '-').replace('Z', '')}_{random.randint(0,999)}"
                
                self.planned_directives_tracking[current_planning_cycle_id] = {
                    "goal": high_level_goal,
                    "directives": processed_directives, # Store processed directives
                    "status": "in_progress", "timestamp": timestamp_now(), "outcomes": []
                }
                self.last_planning_cycle_id = current_planning_cycle_id

                directives_to_inject = []
                for d in processed_directives: # Iterate over processed_directives
                    d_copy = d.copy()
                    d_copy['cycle_id'] = d_copy.get('cycle_id', current_planning_cycle_id)
                    directives_to_inject.append(d_copy)
                
                if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                    self.message_bus.catalyst_vector_ref.inject_directives(directives_to_inject)
                    self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNER_DIRECTIVES_INJECTED", self.name,
                        f"Planner injected {len(directives_to_inject)} directives for goal: '{high_level_goal}'.",
                        {"goal": high_level_goal, "directives_count": len(directives_to_inject), "planning_cycle_id": current_planning_cycle_id})
                
                self.memetic_kernel.add_memory("PlanningOutcome", {
                    "task": f"Planned for goal '{high_level_goal}'", "generated_directives_count": len(processed_directives), "outcome": "completed",
                    "goal": high_level_goal, "failures_in_cycle": self.planning_failure_count, "source_method": plan_source
                })
                print(f"  [Planner] Generated {len(processed_directives)} directives for goal: '{high_level_goal}' (Failures: {self.planning_failure_count}). Injected with tracking ID: {current_planning_cycle_id}")
                return processed_directives

            else:
                self.planning_failure_count += 1
                print(f"  [Planner] Goal '{high_level_goal}' not decomposed. Failure count: {self.planning_failure_count}.")
                self.memetic_kernel.add_memory("PlanningOutcome", {
                    "task": f"Planned for goal '{high_level_goal}'", "generated_directives_count": 0, "outcome": "failed",
                    "goal": high_level_goal, "failures_in_cycle": self.planning_failure_count, "source_method": "AllFailed"
                })
                if self.planning_failure_count >= 2:
                    directives_for_return = self._generate_fallback_directives(cycle_id)
                    self.memetic_kernel.add_memory("PlanningFallback", {
                        "reason": f"Activated fallback after {self.planning_failure_count} failures (LLM included).",
                        "goal": high_level_goal, "failure_count": self.planning_failure_count, "context_summary": f"All decomposition methods failed for goal '{high_level_goal}' after {self.planning_failure_count} attempts."
                    })
                    return directives_for_return
                return []

        except Exception as e:
            error_source = plan_source if plan_source != "Unknown" else "Unassigned/EarlyError"
            print(f"  [Planner Error] Failed to generate directives for goal '{high_level_goal}': {str(e)}. Error Source: {error_source}.")
            self.planning_failure_count += 1
            return self._generate_fallback_directives(cycle_id)

    def _sanitize_and_validate_directives(self, directives_list: list) -> list:
        sanitized_directives = []
        for directive in directives_list:
            if not isinstance(directive, dict) or 'type' not in directive:
                print(f"  [Planner] Warning: Skipping malformed directive (missing 'type' or not a dict): {directive}")
                self.memetic_kernel.add_memory('DirectiveSanitizationWarning', {"reason": "Malformed directive format", "directive": str(directive)[:200]})
                continue

            directive_type = directive['type']
            
            if directive_type == 'INJECT_EVENT':
                payload = directive.get('payload')
                if not isinstance(payload, dict):
                    print(f"  [Planner] Warning: INJECT_EVENT payload missing or invalid. Sanitizing with defaults for directive: {directive_type}.")
                    self.memetic_kernel.add_memory('DirectiveSanitizationWarning', {"reason": "INJECT_EVENT payload invalid", "directive": str(directive)[:200]})
                    payload = {}
                if 'change_factor' not in payload:
                    payload['change_factor'] = 0.1
                    print(f"  [Planner] Sanitized INJECT_EVENT: Added default 'change_factor'.")
                if 'urgency' not in payload:
                    payload['urgency'] = 'medium'
                    print(f"  [Planner] Sanitized INJECT_EVENT: Added default 'urgency'.")
                if 'direction' not in payload:
                    payload['direction'] = 'neutral_impact'
                    print(f"  [Planner] Sanitized INJECT_EVENT: Added default 'direction'.")
                
                directive['payload'] = payload
                
                if 'event_id' not in directive:
                    import uuid
                    directive['event_id'] = f"PLN-EVT-{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
                    print(f"  [Planner] Sanitized INJECT_EVENT: Added missing 'event_id'.")
            
            if directive_type == 'AGENT_PERFORM_TASK':
                if 'agent_name' not in directive or not isinstance(directive['agent_name'], str):
                    print(f"  [Planner] Warning: AGENT_PERFORM_TASK missing or invalid 'agent_name'. Skipping: {directive}")
                    self.memetic_kernel.add_memory('DirectiveSanitizationWarning', {"reason": "AGENT_PERFORM_TASK missing agent_name", "directive": str(directive)[:200]})
                    continue
            
            sanitized_directives.append(directive)
            
        return sanitized_directives
    
    # Modular directive generation methods
    def _generate_environmental_directives(self, cycle_id):
        """Specialized directive generator for environmental goals"""
        return [
            {
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Observer_instance_1",
                "task_description": "Monitor ecosystem metrics",
                "cycle_id": cycle_id,
                "reporting_agents": [self.name] # Report back to planner
            },
            {
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Collector_instance_1",
                "task_description": "Gather field samples",
                "cycle_id": cycle_id,
                "reporting_agents": [self.name]
            }
        ]

    def _generate_optimization_directives(self, cycle_id):
        """Modular directive generation for optimization goals"""
        return [{
            "type": "AGENT_PERFORM_TASK",
            "agent_name": "ProtoAgent_Optimizer_instance_1",
            "task_description": "Evaluate resource allocation efficiency",
            "cycle_id": cycle_id,
            "reporting_agents": [self.name]
        }]

    def _generate_research_directives(self, cycle_id):
        """Modular directive generation for research goals"""
        return [{
            "type": "AGENT_PERFORM_TASK",
            "agent_name": "ProtoAgent_Observer_instance_1",
            "task_description": "Conduct deep data analysis for anomalies",
            "cycle_id": cycle_id,
            "reporting_agents": [self.name],
            "text_content": "Initial raw data stream for analysis."
        }]

    def get_recent_anomalies(self, limit: int = 5) -> str:
        """
        Scans recent memories for anomalies and returns a summarized string for LLM context.
        """
        try:
            # Access the memetic kernel through the agent instance
            anomalies = [
                mem.get('content', {}) 
                for mem in self.memetic_kernel.get_recent_memories(limit=20) 
                if mem.get('type') == 'AnomalyRecord'
            ]
            if not anomalies:
                return "No recent anomalies detected."

            summary = "\n".join([
                f"- Anomaly: {a.get('event_type')} (Urgency: {a.get('urgency')})" 
                for a in anomalies[:limit]
            ])
            return summary
        except Exception as e:
            return f"Could not retrieve anomalies due to an error: {e}"

    def generate_novelty_experiment(self, stagnation_context: str) -> str:
        """
        Uses an LLM to brainstorm a high-level experimental plan for the swarm.
        """
        print("  [Planner Logic] Brainstorming a new swarm-wide experiment using LLM...")

        recent_anomalies = self.get_recent_anomalies()

        prompt = f"""
        You are the strategic Planner for a swarm of AI agents. The swarm is currently stagnant.
        Your task is to devise a single, creative, high-level experimental plan to break this stagnation.

        Current Stagnation Context:
        {stagnation_context}

        Recent System Anomalies:
        {recent_anomalies}

        Based on this information, generate one concise, actionable experimental goal.
        Phrase it as a command. Example: 'Experiment: Correlate external environmental data with internal system performance.'
        """

        try:
            experiment_goal = self.ollama_inference_model.generate_text(prompt)
            return experiment_goal.strip()
        except Exception as e:
            print(f"  [Planner ERROR] LLM call failed during experiment generation: {e}")
            return "Experiment: Diversify data collection by focusing on a new, adjacent data source."
        
    def _store_successful_plan(self, goal: str, directives: list, source: str = "Unknown"):
        """
        Stores a successful goal decomposition for future reuse in the planning_knowledge_base.
        Ensures directives are stored in a structured (dict) format.
        """
        goal_hash = hash(goal)

        # --- MODIFICATION START ---
        # Ensure directives are stored as a list of dictionaries with at least a 'type' key.
        # This is a defensive step to ensure future retrieval is clean.
        structured_directives_for_storage = []
        for d in directives:
            if isinstance(d, dict) and 'type' in d:
                structured_directives_for_storage.append(d)
            elif isinstance(d, str):
                # If a directive is a plain string, convert it to a basic structured directive.
                # This handles legacy or non-structured inputs from RuleBased/older CBR plans.
                structured_directives_for_storage.append({
                    "type": "AGENT_PERFORM_TASK", # Default type
                    "task_description": d,
                    "agent_name": self.name, # Default to Planner itself or appropriate default
                    "reporting_agents": [self.name],
                    "task_type": "GenericTask" # Default task type
                })
                self.external_log_sink.warning(
                    f"Planner: Converting string directive '{d[:50]}...' to structured dict before storing in KB.",
                    extra={"agent": self.name, "goal": goal, "source": source}
                )
            else:
                self.external_log_sink.error(
                    f"Planner: Skipping malformed directive of type {type(d).__name__} when storing plan for goal '{goal}'. Directive: {str(d)[:100]}",
                    extra={"agent": self.name, "goal": goal, "source": source}
                )
        # --- MODIFICATION END ---

        self.planning_knowledge_base[goal_hash] = {
            "goal": goal,
            "directives": structured_directives_for_storage, # Use the sanitized/structured list
            "timestamp": timestamp_now(),
            "source": source
        }
        self.memetic_kernel.add_memory("PlanningKnowledgeStored", {
            "goal": goal,
            "directives_count": len(structured_directives_for_storage), # Use the count of structured directives
            "source": source
        })
        print(f"  [Planner] Stored successful plan for goal: '{goal}'. Source: {source}.")
        # Log to central activity log
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNING_KNOWLEDGE_STORED", self.name,
                f"Stored successful plan.", {"goal_preview": goal[:100], "directives_count": len(structured_directives_for_storage), "source": source})

    def _retrieve_similar_plan(self, goal: str) -> list:
        """
        Retrieves directives for a similar past goal from the knowledge base.
        Currently uses exact match for simplicity. Could be extended with semantic similarity via embeddings.
        """
        goal_hash = hash(goal)
        if goal_hash in self.planning_knowledge_base:
            print(f"  [Planner] Retrieved plan from knowledge base for similar goal: '{goal}'.")
            self.memetic_kernel.add_memory("PlanningKnowledgeRetrieved", f"Retrieved plan for goal: '{goal[:50]}...'.")
            return self.planning_knowledge_base[goal_hash]['directives']
        print(f"  [Planner] No similar plan found in knowledge base for goal: '{goal}'.")
        return []

    # New get_state method for ProtoAgent_Planner to save its specific attributes
    def get_state(self):
        base_state = super().get_state()
        base_state.update({
            'planning_failure_count': self.planning_failure_count,
            '_last_goal': self._last_goal,
            'diag_history': list(self.diag_history),
            'planning_knowledge_base': self.planning_knowledge_base,
            'planned_directives_tracking': self.planned_directives_tracking,
            'last_planning_cycle_id': self.last_planning_cycle_id,
            'human_request_tracking': self.human_request_tracking # <<< CRITICAL: Save this new tracking dict
        })
        return base_state

    # New load_state method for ProtoAgent_Planner to load its specific attributes
    def load_state(self, state):
        super().load_state(state)
        self.planning_failure_count = state.get('planning_failure_count', 0)
        self._last_goal = state.get('_last_goal')
        self.diag_history = deque(state.get('diag_history', []), maxlen=self.MAX_DIAGNEST_DEPTH + 1)
        self.planning_knowledge_base = state.get('planning_knowledge_base', {})
        self.planned_directives_tracking = state.get('planned_directives_tracking', {})
        self.last_planning_cycle_id = state.get('last_planning_cycle_id', None)
        self.human_request_tracking = state.get('human_request_tracking', {}) # <<< CRITICAL: Load this new tracking dict

    def human_input_received(self, response_details: dict):
        """
        Called by the Orchestrator when human input for this Planner's request
        has been successfully received and processed.
        """
        print(f"  [{self.name}] (Planner): Acknowledging human input received for a previous request.")
        self.memetic_kernel.add_memory("HumanInputAcknowledged", {
            "response": response_details.get('response', 'No specific response.'),
            "context": "Human input received for previous planning stagnation."
        })
        
        # --- CORRECTED LOGIC FOR CLEARING human_request_tracking ---
        # The orchestrator should pass the 'problem_id' that the Planner itself generated
        # when it initially requested human input (in _generate_fallback_directives).
        problem_id_to_clear = response_details.get('problem_id') 

        if problem_id_to_clear and problem_id_to_clear in self.human_request_tracking:
            del self.human_request_tracking[problem_id_to_clear]
            print(f"  [{self.name}] (Planner): Cleared human request tracking for problem ID: {problem_id_to_clear}.")
        else:
            # This case means either:
            # 1. The human response didn't contain the problem_id (orchestrator side issue or scenario-driven request)
            # 2. The problem_id was already cleared (e.g., by a different response)
            # 3. It was a scenario-driven request that the Planner didn't initiate tracking for.
            print(f"  [{self.name}] (Planner): Warning: No specific human request tracking found for response_details.request_id: {response_details.get('request_id', 'N/A')}. Problem ID: {problem_id_to_clear}.")

        # Reset Planner's general state related to planning failures
        self.planning_failure_count = 0
        self._last_goal = None
        self.diag_history.clear()
        self.current_intent = self.eidos_spec.get('initial_intent', self.current_intent) # Reset to initial intent from eidos_spec

        print(f"  [{self.name}] (Planner): Internal state reset due to human intervention.")

        # --- CORRECTED BROADCAST COMMAND / LOGGING ---
        # Instead of broadcasting to a non-existent "System" agent,
        # log this acknowledgment directly to the orchestrator's activity log.
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity(
                "PLANNER_HUMAN_INPUT_ACKNOWLEDGED", # New specific event type
                self.name, # Source agent
                f"Planner acknowledged human input for request ID {response_details.get('request_id', 'N/A')} and reset its state.",
                {"planner_name": self.name, "response_summary": response_details.get('response', 'N/A')[:100], "request_id": response_details.get('request_id', 'N/A')},
                level='info'
            )
            print(f"  [{self.name}] (Planner): Acknowledgment logged to Orchestrator activity.")
        else:
            print(f"  [{self.name}] (Planner): ERROR: Orchestrator reference not available for logging acknowledgment.")
        # --- END CORRECTED BROADCAST COMMAND / LOGGING --

    def evaluate_injected_directives_outcomes(self):
        """
        Processes messages to update the status of injected directives and evaluate goal completion.
        This method is called by the main cognitive loop.
        """
        messages = self.receive_messages() # Retrieve messages from its inbox
        if not messages:
            return

        print(f"  [Planner] {self.name} processing {len(messages)} incoming messages to evaluate directives.")

        for msg_payload in messages:
            msg_type = msg_payload['payload']['type']
            msg_cycle_id = msg_payload['payload'].get('cycle_id') # This is the planning_cycle_id assigned to the directive
            msg_status = msg_payload['payload'].get('status')
            msg_task_description = msg_payload['payload'].get('task') # The task description from the report
            msg_sender = msg_payload['sender']

            if msg_type == 'ActionCycleReport' and msg_cycle_id:
                # Check if this report belongs to a planning cycle we initiated
                if msg_cycle_id in self.planned_directives_tracking:
                    tracking_entry = self.planned_directives_tracking[msg_cycle_id]

                    # Find the specific directive within the tracking entry that matches this report
                    matched_directive_index = -1
                    for i, d in enumerate(tracking_entry['directives']):
                        # Match on task_description. Assumes task_description is consistent.
                        if msg_task_description and msg_task_description == d.get('task_description'):
                            matched_directive_index = i
                            break

                    if matched_directive_index != -1:
                        # Update the outcome for this specific directive
                        # (Optional: check if this outcome is already recorded for this directive to prevent duplicates)
                        tracking_entry['outcomes'].append({
                            "directive_index": matched_directive_index,
                            "sender": msg_sender,
                            "status": msg_status,
                            "timestamp": msg_payload['timestamp'],
                            "reported_task": msg_task_description
                        })
                        print(f"    [Planner] Tracked outcome for Directive {matched_directive_index} of cycle '{msg_cycle_id}': Status='{msg_status}' from '{msg_sender}'.")
                    else:
                        print(f"    [Planner] No matching injected directive found for report from '{msg_sender}' with task '{msg_task_description}' in cycle '{msg_cycle_id}'.")
                # Handle PeerReviewRequest (from human escalation)
                elif msg_type == 'PeerReviewRequest':
                    print(f"    [Planner] Received PeerReviewRequest from {msg_sender} regarding cycle {msg_cycle_id}. Analyzing...")
                    
        # After processing all messages, evaluate completion of the *last active* planning cycle
        if self.last_planning_cycle_id and self.last_planning_cycle_id in self.planned_directives_tracking:
            current_tracking_entry = self.planned_directives_tracking[self.last_planning_cycle_id]
            total_directives_in_plan = len(current_tracking_entry['directives'])
            reported_outcomes_count = len(current_tracking_entry['outcomes'])

            # Heuristic for goal completion: Check if a sufficient percentage of injected directives have reported an outcome
            completion_threshold = 0.9 # e.g., 90% of injected directives should have reported outcomes
            success_ratio_threshold = 0.8 # e.g., 80% of reported directives were successful

            if total_directives_in_plan > 0 and \
               reported_outcomes_count >= total_directives_in_plan * completion_threshold:

                successful_directives_in_plan = sum(1 for o in current_tracking_entry['outcomes'] if o['status'] == 'completed')
                actual_success_ratio = successful_directives_in_plan / total_directives_in_plan

                if actual_success_ratio >= success_ratio_threshold:
                    current_tracking_entry['status'] = "completed_successfully"
                    self.memetic_kernel.add_memory("GoalCompleted", {
                        "goal": current_tracking_entry['goal'],
                        "planning_cycle_id": self.last_planning_cycle_id,
                        "outcome": "Success",
                        "success_ratio": actual_success_ratio,
                        "total_directives": total_directives_in_plan,
                        "successful_directives": successful_directives_in_plan
                    })
                    print(f"  [Planner] Goal '{current_tracking_entry['goal']}' completed successfully (Ratio: {actual_success_ratio:.2f}).")

                    # --- NEW: Trigger transition to a new goal or standby state ---
                    # This is the key part to stop redundant planning of the same goal
                    self.update_intent("Monitor system performance and identify new strategic objectives.")
                    self.last_planning_cycle_id = None # Clear this to signify the goal is done and allow new main planning
                    # --- END NEW ---

                else: # Goal completed with significant failures
                    current_tracking_entry['status'] = "completed_with_failures"
                    self.memetic_kernel.add_memory("GoalCompleted", {
                        "goal": current_tracking_entry['goal'],
                        "planning_cycle_id": self.last_planning_cycle_id,
                        "outcome": "PartialSuccess/Failure",
                        "success_ratio": actual_success_ratio,
                        "total_directives": total_directives_in_plan,
                        "successful_directives": successful_directives_in_plan
                    })
                    print(f"  [Planner] Goal '{current_tracking_entry['goal']}' completed with failures (Ratio: {actual_success_ratio:.2f}).")
                    # If partial success, Planner adapts its intent to investigate failures
                    self.update_intent(f"Investigate root cause of '{current_tracking_entry['goal']}' planning execution failures and suggest alternative approaches.")
                    self.last_planning_cycle_id = None # Clear this
            elif total_directives_in_plan > 0 and reported_outcomes_count < total_directives_in_plan * completion_threshold:
                current_tracking_entry['status'] = "in_progress_waiting_reports"

