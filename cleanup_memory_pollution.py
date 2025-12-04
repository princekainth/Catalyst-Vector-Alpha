#!/usr/bin/env python3
"""Clean up memory-polluted task descriptions."""

from db_postgres import execute_query
import re

def clean_task_description(desc):
    """Extract original task from memory-polluted description."""
    # Pattern: everything after the last "No specific intent" or similar
    if "[Memory Recall]" in desc:
        # Find the actual task after all the memory context
        lines = desc.split('\n')
        # The real task is usually at the end
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('✓') and not line.startswith('⚠'):
                if 'Outcome:' not in line and 'success rate:' not in line:
                    return line
    return desc

# Get polluted tasks
polluted = execute_query('''
    SELECT task_id, task_description 
    FROM task_history 
    WHERE task_description LIKE '%[Memory Recall]%'
''', fetch=True)

print(f"Found {len(polluted)} polluted tasks. Cleaning...")

cleaned = 0
for task in polluted:
    original = clean_task_description(task['task_description'])
    if original != task['task_description']:
        execute_query('''
            UPDATE task_history 
            SET task_description = %s 
            WHERE task_id = %s
        ''', (original, task['task_id']))
        cleaned += 1

print(f"✓ Cleaned {cleaned} tasks!")
