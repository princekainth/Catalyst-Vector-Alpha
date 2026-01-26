import os
import time
import subprocess
import json
import signal

LOG_FILE = "logs/catalyst.jsonl"
OUT_FILE = "verify_truth.log"

def run_test():
    # 1. Cleanup
    subprocess.run("fuser -k 5000/tcp || true", shell=True)
    subprocess.run("pkill -9 -f app.py || true", shell=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    # 2. Start Server
    print("Starting server...")
    proc = subprocess.Popen("python3 app.py", shell=True, stdout=open(OUT_FILE, "w"), stderr=subprocess.STDOUT)
    
    # 3. Wait for stability
    print("Waiting for agents to rehydrate...")
    start_wait = time.time()
    ready = False
    while time.time() - start_wait < 120:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                content = f.read()
                if "Entering Continuous Cognitive Loop" in content:
                    ready = True
                    break
        time.sleep(5)
    
    if not ready:
        print("Server not ready after 120s. Check verify_truth.log")
        proc.terminate()
        return

    print("System READY. Injecting goal...")
    
    # 4. Inject
    cmd = "python3 inject_goal.py --mission health_audit 'Check pod health and identify high CPU consumer pods'"
    subprocess.run(cmd, shell=True)
    
    # 5. Monitor for completion
    print("Goal injected. Waiting for execution...")
    start_wait = time.time()
    success = False
    while time.time() - start_wait < 90:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                for line in f:
                    if "AGENT_TASK_PERFORMED" in line:
                        entry = json.loads(line)
                        if "report_content" in entry.get("details", {}):
                            report = entry["details"]["report_content"]
                            if report.get("fast_path"):
                                count = report.get("dispatched_count", 0)
                                print(f"FOUND TASK! FastPath: {report.get('fast_path')}, Dispatched: {count}")
                                if count > 0:
                                    success = True
                        elif "report_content" in entry: # check different structure
                             report = entry["report_content"]
                             if report.get("fast_path"):
                                count = report.get("dispatched_count", 0)
                                print(f"FOUND TASK! FastPath: {report.get('fast_path')}, Dispatched: {count}")
                                if count > 0:
                                    success = True
        if success: break
        time.sleep(2)
    
    # 6. Final Report
    print("\n=== FINAL VERIFICATION RESULTS ===")
    if success:
        print("✅ SUCCESS: Skill was matched and tool calls were dispatched!")
        # Grep for TOOL CALLS
        print("\n--- TOOL CALLS FOUND ---")
        subprocess.run(f"grep 'TOOL CALL' {LOG_FILE}", shell=True)
    else:
        print("❌ FAILURE: Skill might have matched but no tools were dispatched.")
        print("\n--- LOG TAIL ---")
        subprocess.run(f"tail -n 20 {LOG_FILE}", shell=True)

    os.kill(proc.pid, signal.SIGTERM)

if __name__ == "__main__":
    run_test()
