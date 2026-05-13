"""
SwarmProtocol — extracted from catalyst_vector_alpha.py for clarity.

Managed group of agents with a shared goal, memory, and sovereign gradient.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, TYPE_CHECKING

from shared_models import MemeticKernel, SovereignGradient

if TYPE_CHECKING:
    from catalyst_vector_alpha import CatalystVectorAlpha

logger = logging.getLogger("CatalystLogger")


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
                logger.debug(f"[SovereignGradient] Swarm task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
        
        self.memetic_kernel.add_memory("TaskCoordination", f"Swarm '{self.name}' coordinating task: '{final_task_description}' (Compliant: {gradient_compliant}) among {len(self.members)} members (conceptual).")
        logger.info(f"[SwarmProtocol] Swarm '{self.name}' coordinating task: '{final_task_description}' among {len(self.members)} members (conceptual).")
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
