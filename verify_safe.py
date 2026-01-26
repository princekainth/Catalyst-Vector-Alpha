import time
import subprocess
import os

LOG_FILE = "logs/catalyst.log"

def follow_log(filepath):
    """Yield new lines from log file."""
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                yield None
            else:
                yield line

def wait_for_agent():
    print("Waiting for Planner agent initialization...")
    start_time = time.time()
    
    # We poll the file manually since it might not exist yet
    while time.time() - start_time < 300: # 5 min timeout
        if os.path.exists(LOG_FILE):
             with open(LOG_FILE, "r") as f:
                 content = f.read()
                 if "[SkillRegistry] Loaded" in content:
                     print("✅ SkillRegistry Loaded!")
                     time.sleep(5) # Extra buffer
                     return True
        time.sleep(1)
    
    print("❌ Timeout waiting for agent.")
    return False

def verify():
    # 1. Clean
    subprocess.run("fuser -k 5000/tcp || true", shell=True)
    subprocess.run("pkill -9 -f app.py || true", shell=True)
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    if os.path.exists("logs/catalyst.log"): os.remove("logs/catalyst.log")

    # 2. Start
    print("Starting server...")
    proc = subprocess.Popen("stdbuf -oL python3 app.py > live_verify.log 2>&1", shell=True)
    
    # 3. Wait for readiness
    if not wait_for_agent():
        proc.terminate()
        return

    # 4. Inject
    print("Injecting goal...")
    subprocess.run("python3 inject_goal.py --mission health_audit 'Check pod health and identify high CPU consumer pods'", shell=True)
    
    # 5. Monitor for Fast Path
    print("Monitoring for Fast Path Execution...")
    start_wait = time.time()
    found_debug = False
    found_dispatch = False
    
    while time.time() - start_wait < 120:
        if os.path.exists("logs/catalyst.log"):
            with open("logs/catalyst.log", "r") as f:
                content = f.read()
                if "[DEBUG_FAST]" in content:
                    found_debug = True
                if "PLAN_STEPS_INJECTED" in content:
                    # check count
                    for line in content.splitlines():
                        if "PLAN_STEPS_INJECTED" in line and '"count": 0' not in line and '"count": 4' in line:
                             found_dispatch = True
        
        if found_debug and found_dispatch:
            print("✅ SUCCESS: Fast Path Triggered and Dispatched steps!")
            break
        time.sleep(1)
        
    if not (found_debug and found_dispatch):
        print("❌ FAILED: Did not see expected logs.")
        subprocess.run("grep 'PLAN_STEPS_INJECTED' logs/catalyst.log", shell=True)
    
    proc.terminate()

if __name__ == "__main__":
    verify()
