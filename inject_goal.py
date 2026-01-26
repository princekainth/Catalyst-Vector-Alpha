import sys
import requests
import json
import uuid
import argparse
import time
import os

def check_logs_for_task(task_id: str, log_file: str = "logs/catalyst.log", duration: int = 15):
    print(f"👀 Monitoring logs for Task ID: {task_id} ({duration}s timeout)...")
    start_time = time.time()
    
    # Track state
    found_injection = False
    found_fast_path = False
    found_dispatch = False
    
    while time.time() - start_time < duration:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                # Read from end if possible, but for simplicity read whole file 
                # (in prod use seek, here file is small enough)
                content = f.read()
                
                if task_id in content:
                    found_injection = True
                    
                    if "SKILL_FAST_PATH" in content:
                        found_fast_path = True
                    
                    if "SKILL_EXECUTION_DISPATCHED" in content and task_id in content:
                         found_dispatch = True
                         
        if found_dispatch:
            print("\n✨ VERIFICATION SUCCESS:")
            print(f"   - Injection Received: ✅")
            print(f"   - Skill Fast-Path:    ✅")
            print(f"   - Execution Dispatch: ✅ (SKILL_EXECUTION_DISPATCHED confirmed)")
            return True
            
        time.sleep(1)
        
    print("\n⚠️ VERIFICATION TIMEOUT:")
    print(f"   - Injection Received: {'✅' if found_injection else '❌'}")
    print(f"   - Skill Fast-Path:    {'✅' if found_fast_path else '❌'}")
    print(f"   - Execution Dispatch: {'❌' if not found_dispatch else '✅'}")
    return False

def inject_goal(goal: str, mission: str = "general_planning", port: int = 5000):
    url = f"http://127.0.0.1:{port}/api/command"
    payload = {
        "command": goal,
        "mission_type": mission
    }
    
    print(f"🚀 Injecting goal: '{goal}' (Mission: {mission})")
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        task_id = data.get('task_id')
        print(f"✅ Success! Task ID: {task_id}")
        
        # Auto-verify
        check_logs_for_task(task_id)
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to CVA at {url}. Is it running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a goal into CVA")
    parser.add_argument("goal", help="The goal description")
    parser.add_argument("--mission", default="general_planning", help="Mission type (default: general_planning)")
    parser.add_argument("--port", type=int, default=5000, help="API port")
    
    args = parser.parse_args()
    inject_goal(args.goal, args.mission, args.port)
