# core/intent_guard.py

def enforce_intent(task: dict) -> bool:
    """
    Checks if a task directive has a valid, actionable intent.
    Returns False if the task should be dropped.
    """
    # Use .get() for dictionaries to safely access keys
    description = task.get('task_description', '').lower()
    
    # An INITIATE_PLANNING_CYCLE is always a valid intent
    if task.get('type') == 'INITIATE_PLANNING_CYCLE':
        return True
    
    # Check for idle phrases in the description of other tasks
    idle_phrases = [
        "no specific intent", "awaiting tasks", "diagnostic standby", 
        "standby mode", "no active objectives"
    ]
    if any(phrase in description for phrase in idle_phrases):
        return False
        
    return True