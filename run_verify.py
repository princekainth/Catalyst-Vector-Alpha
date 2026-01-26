import subprocess
import time
import sys

print("Starting server...")
with open("final_truth.log", "w") as out:
    proc = subprocess.Popen(["python3", "app.py"], stdout=out, stderr=subprocess.STDOUT, text=True)
    
    print("Waiting 90s for agents to rehydrate...")
    time.sleep(90)
    
    print("Injecting goal...")
    inj = subprocess.run(["python3", "inject_goal.py", "--mission", "health_audit", "Check pod health and identify high CPU consumer pods"], capture_output=True, text=True)
    print(inj.stdout)
    if inj.stderr: print(inj.stderr)
    
    print("Waiting 60s for cognitive cycle...")
    time.sleep(60)
    
    proc.terminate()
    print("Server stopped. checking logs...")
