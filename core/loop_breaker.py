# core/loop_breaker.py
from typing import List, Tuple, Dict, Any
from collections import deque
import time

# Use a deque to automatically keep track of the N most recent results
_activity_history: Dict[str, deque] = {}
HISTORY_LENGTH = 20 # Keep the last 20 results for any given activity

def should_continue_activity(activity_type: str, result: Any) -> Tuple[bool, str]:
    """
    Determines if a repetitive activity should continue or be stopped.
    Tracks the history of results for each activity type.

    Returns:
        (bool): True if the activity should continue.
        (str): The reason for the decision.
    """
    if activity_type not in _activity_history:
        _activity_history[activity_type] = deque(maxlen=HISTORY_LENGTH)

    # Add the latest result to this activity's history
    _activity_history[activity_type].append(result)

    history = list(_activity_history[activity_type])

    # Condition 1: Excessive repetition with no meaningful results
    # We define "no meaningful results" as tool skips or empty outputs
    if len(history) >= 10:
        non_meaningful_count = 0
        for res in history[-10:]: # Check the last 10
            if isinstance(res, dict) and res.get("status") == "skipped":
                non_meaningful_count += 1
            elif isinstance(res, str) and "No results for" in res:
                non_meaningful_count += 1

        if non_meaningful_count >= 8: # If 8 of the last 10 were useless
            return False, f"Excessive repetition ({non_meaningful_count}/10) without meaningful results."

    # Condition 2: Stable, consistent results (no new information)
    if len(history) >= 5:
        # Check if the last 5 results are all identical
        last_five = history[-5:]
        if all(res == last_five[0] for res in last_five):
            return False, "Results have stabilized; activity should be paused or changed."

    return True, "Continue activity."