#!/usr/bin/env python3
import json
import time
import os
import sys
from datetime import datetime

# Color mapping
COLORS = {
    "ANNOUNCEMENT": "\033[93m",      # Yellow
    "SYSTEM_EVOLUTION": "\033[95m",  # Purple
    "CURIOSITY_DISCOVERY": "\033[96m",# Cyan
    "CURIOSITY_GAP": "\033[91m",      # Red
    "MEMORY_RECALL": "\033[94m",      # Blue
    "META_EVOLUTION": "\033[92m",    # Green
    "RESET": "\033[0m"
}

ICONS = {
    "ANNOUNCEMENT": "📢",
    "SYSTEM_EVOLUTION": "🧬",
    "CURIOSITY_DISCOVERY": "🔭",
    "CURIOSITY_GAP": "⚠️",
    "MEMORY_RECALL": "🧠",
    "META_EVOLUTION": "⚙️"
}

LOG_FILE = "logs/catalyst.jsonl"

def print_pulse(event):
    etype = event.get("event_type")
    if not etype or etype not in COLORS:
        # Fallback for structured logs that nested their data
        try:
            msg = json.loads(event.get("message", "{}"))
            if isinstance(msg, dict) and "event_type" in msg:
                event = msg
                etype = event.get("event_type")
        except:
            pass
            
    if not etype or etype not in COLORS:
        return

    color = COLORS.get(etype, COLORS["RESET"])
    icon = ICONS.get(etype, "🔹")
    timestamp = event.get("timestamp")
    if isinstance(timestamp, (int, float)):
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    else:
        ts_str = str(timestamp)[11:19] if timestamp else "??:??:??"
        
    source = event.get("source", "System")
    desc = event.get("description") or event.get("message") or ""
    
    # Clean up desc if it's too long
    if len(desc) > 120:
        desc = desc[:117] + "..."

    print(f"[{ts_str}] {color}{icon} {etype:18} {COLORS['RESET']} | {color}{source:15}{COLORS['RESET']} | {desc}")

def main():
    print("\033[1m" + "="*80 + "\033[0m")
    print("\033[1;36m" + " CVA PULSE - High-Level System Feed ".center(80) + "\033[0m")
    print("\033[1m" + "="*80 + "\033[0m")
    print(f"Monitoring {LOG_FILE} for milestones...\n")

    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found. Start CVA first!")
        return

    # Seek to end of file
    with open(LOG_FILE, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            
            try:
                event = json.loads(line)
                print_pulse(event)
            except Exception:
                continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[1;33mPulse monitoring stopped.\033[0m")
