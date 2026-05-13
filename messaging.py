"""
Messaging and Event Handling — extracted from shared_models.py for clarity.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional


@dataclass
class BusMessage:
    sender: str
    recipient: str
    message_type: str
    content: Any
    task_description: Optional[str]
    status: str
    cycle_id: str
    timestamp: str


class MessageBus:
    def __init__(self):
        self.messages = {}
        self.lock = Lock()
        self.catalyst_vector_ref = None

    def send_message(self, sender: str, recipient: str, message_type: str, content: Any, 
                     task_description: str = None, status: str = "pending", cycle_id: str = None):
        """Thread-safe message sending."""
        with self.lock:
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
    
    def get_messages_for(self, agent_name: str):
        """Thread-safe message retrieval - matches what agents expect."""
        with self.lock:
            msgs = self.messages.get(agent_name, [])
            self.messages[agent_name] = []
            return msgs
        
    def send_directive(self, directive):
        if self.catalyst_vector_ref:
            enqueue_fn = getattr(self.catalyst_vector_ref, "enqueue_directive", None)
            if enqueue_fn:
                enqueue_fn(directive)
            else:
                # Hard fail to avoid unsynchronized writes
                raise RuntimeError("enqueue_directive not available on catalyst_vector_ref")

class EventMonitor:
    def __init__(self):
        self.event_history = []
        self.agent_responses = defaultdict(list)
        self.current_cycle_id = None
        self._lock = Lock()

    def set_current_cycle(self, cycle_id: str):
        with self._lock:
            self.current_cycle_id = cycle_id

    def log_event(self, event_type: str, event_id: str, payload: dict):
        event_record = {
            'event_id': event_id,
            'type': event_type,
            'urgency': payload.get('urgency'),
            'change_factor': payload.get('change_factor'),
            'direction': payload.get('direction'),
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'cycle_id': self.current_cycle_id
        }
        with self._lock:
            self.event_history.append(event_record)
        print(f"  [EventMonitor] Logged event: {event_record['event_id'][:8]} ({event_record['type']})")

    def log_agent_response(self, agent_id: str, event_id: str, response_type: str, details: dict = None):
        response_record = {
            'event_id': event_id,
            'response_type': response_type,
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'details': details if details is not None else {},
            'cycle_id': self.current_cycle_id
        }
        with self._lock:
            self.agent_responses[agent_id].append(response_record)
        print(f"  [EventMonitor] Agent {agent_id} responded to {event_id[:8]} ({response_type})")

    def get_event_history(self, event_id: str = None):
        with self._lock:
            if event_id:
                return [e for e in self.event_history if e['event_id'] == event_id]
            return list(self.event_history)

    def get_agent_event_responses(self, agent_id: str, event_id: str = None):
        with self._lock:
            responses = list(self.agent_responses.get(agent_id, []))
        if event_id:
            return [r for r in responses if r['event_id'] == event_id]
        return responses

    def get_state(self):
        """Returns the current state of the EventMonitor for persistence."""
        with self._lock:
            serializable_agent_responses = {
                agent_id: list(responses) for agent_id, responses in self.agent_responses.items()
            }
            return {
                'event_history': list(self.event_history),
                'agent_responses': serializable_agent_responses,
                'current_cycle_id': self.current_cycle_id
            }

    def load_state(self, state):
        """Loads the state into the EventMonitor."""
        with self._lock:
            self.event_history = state.get('event_history', [])
            loaded_responses = state.get('agent_responses', {})
            self.agent_responses = defaultdict(list, {
                k: list(v) for k, v in loaded_responses.items()
            })
            self.current_cycle_id = state.get('current_cycle_id', None)
