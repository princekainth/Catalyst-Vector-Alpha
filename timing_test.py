#!/usr/bin/env python3
"""
End-to-End CVA Remediation Timing Test
Tests: Detection → Planning → Execution → Recovery
"""

import time
import subprocess
import json
from datetime import datetime

class TimingTest:
    def __init__(self):
        self.results = {
            'test_start': None,
            'chaos_injected': None,
            'detection_start': None,
            'detection_complete': None,
            'planning_start': None,
            'planning_complete': None,
            'execution_start': None,
            'execution_complete': None,
            'pod_running': None,
            'test_end': None,
            'total_time': None
        }
    
    def inject_chaos(self):
        """Create a broken pod"""
        print("\n🔥 INJECTING CHAOS...")
        self.results['chaos_injected'] = time.time()
        
        manifest = """
apiVersion: v1
kind: Pod
metadata:
  name: cva-timing-test
  labels:
    test: timing
spec:
  containers:
  - name: broken
    image: nonexistent-registry.io/fake-image:v1.0.0
"""
        
        with open('/tmp/broken-pod.yaml', 'w') as f:
            f.write(manifest)
        
        subprocess.run(['kubectl', 'delete', 'pod', 'cva-timing-test', '--ignore-not-found=true'])
        time.sleep(2)
        subprocess.run(['kubectl', 'apply', '-f', '/tmp/broken-pod.yaml'])
        print("✓ Broken pod deployed: cva-timing-test")
    
    def wait_for_imagepullbackoff(self, timeout=60):
        """Wait for pod to enter ImagePullBackOff state"""
        print("\n⏳ Waiting for ImagePullBackOff state...")
        start = time.time()
        
        while time.time() - start < timeout:
            result = subprocess.run(
                ['kubectl', 'get', 'pod', 'cva-timing-test', '-o', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pod = json.loads(result.stdout)
                status = pod.get('status', {})
                container_statuses = status.get('containerStatuses', [])
                
                if container_statuses:
                    waiting = container_statuses[0].get('state', {}).get('waiting', {})
                    reason = waiting.get('reason', '')
                    
                    if reason in ['ImagePullBackOff', 'ErrImagePull']:
                        print(f"✓ Pod in {reason} state")
                        return True
            
            time.sleep(2)
        
        print("✗ Timeout waiting for ImagePullBackOff")
        return False
    
    def monitor_cva_logs(self):
        """Monitor CVA logs for timing markers"""
        print("\n👁️  MONITORING CVA LOGS...")
        print("Watching for: Detection → Planning → Execution\n")
        
        # Tail CVA logs (adjust path to your CVA log file)
        try:
            proc = subprocess.Popen(
                ['tail', '-f', 'cva.log'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            for line in proc.stdout:
                print(f"  {line.strip()}")
                
                # Parse timing markers
                if 'Detection started' in line or 'K8sObserver' in line and 'ImagePullBackOff' in line:
                    if not self.results['detection_start']:
                        self.results['detection_start'] = time.time()
                        print(f"\n⏱️  DETECTION START: {self.elapsed('chaos_injected')}s from chaos")
                
                elif 'Planning started' in line or 'K8sPlanner' in line:
                    if not self.results['planning_start']:
                        self.results['planning_start'] = time.time()
                        print(f"⏱️  PLANNING START: {self.elapsed('detection_start')}s from detection")
                
                elif 'Execution started' in line or 'Remediation' in line or 'K8sWorker' in line:
                    if not self.results['execution_start']:
                        self.results['execution_start'] = time.time()
                        print(f"⏱️  EXECUTION START: {self.elapsed('planning_start')}s from planning")
                
                # Check if pod is fixed
                if self.check_pod_running():
                    if not self.results['pod_running']:
                        self.results['pod_running'] = time.time()
                        print(f"\n✅ POD RUNNING: {self.elapsed('chaos_injected')}s TOTAL\n")
                        proc.kill()
                        return True
        
        except KeyboardInterrupt:
            proc.kill()
            return False
    
    def check_pod_running(self):
        """Check if pod is in Running state"""
        result = subprocess.run(
            ['kubectl', 'get', 'pod', 'cva-timing-test', '-o', 'json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pod = json.loads(result.stdout)
            phase = pod.get('status', {}).get('phase', '')
            return phase == 'Running'
        
        return False
    
    def elapsed(self, from_key):
        """Calculate elapsed time from a previous timestamp"""
        if not self.results[from_key]:
            return 0
        return round(time.time() - self.results[from_key], 2)
    
    def print_results(self):
        """Print final timing breakdown"""
        print("\n" + "="*60)
        print("🏁 TIMING TEST RESULTS")
        print("="*60)
        
        if self.results['detection_start'] and self.results['chaos_injected']:
            detection_time = self.results['detection_start'] - self.results['chaos_injected']
            print(f"  Detection Time:    {detection_time:.2f}s")
        
        if self.results['planning_start'] and self.results['detection_start']:
            planning_time = self.results['planning_start'] - self.results['detection_start']
            print(f"  Planning Time:     {planning_time:.2f}s")
        
        if self.results['execution_start'] and self.results['planning_start']:
            execution_time = self.results['execution_start'] - self.results['planning_start']
            print(f"  Execution Time:    {execution_time:.2f}s")
        
        if self.results['pod_running'] and self.results['execution_start']:
            recovery_time = self.results['pod_running'] - self.results['execution_start']
            print(f"  Recovery Time:     {recovery_time:.2f}s")
        
        if self.results['pod_running'] and self.results['chaos_injected']:
            total_time = self.results['pod_running'] - self.results['chaos_injected']
            self.results['total_time'] = total_time
            
            print(f"\n  {'='*56}")
            print(f"  TOTAL TIME:        {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"  {'='*56}\n")
            
            # Verdict
            if total_time < 120:
                print("  ✅ READY TO SELL - Premium positioning possible")
            elif total_time < 300:
                print("  ✅ SELLABLE - Room for optimization")
            elif total_time < 600:
                print("  ⚠️  NEEDS OPTIMIZATION - Profile bottlenecks")
            else:
                print("  ❌ ARCHITECTURE ISSUE - Needs redesign")
        
        print("="*60 + "\n")
    
    def run(self):
        """Execute full timing test"""
        self.results['test_start'] = time.time()
        
        print("\n" + "="*60)
        print("🚀 CVA END-TO-END REMEDIATION TIMING TEST")
        print("="*60)
        
        # Step 1: Inject chaos
        self.inject_chaos()
        
        # Step 2: Wait for failure state
        if not self.wait_for_imagepullbackoff():
            print("Test failed - pod didn't reach ImagePullBackOff")
            return
        
        # Step 3: Monitor CVA remediation
        print("\n⚠️  START CVA NOW IN ANOTHER TERMINAL")
        print("Press ENTER when CVA is running...")
        input()
        
        self.monitor_cva_logs()
        
        # Step 4: Print results
        self.results['test_end'] = time.time()
        self.print_results()
        
        # Cleanup
        print("Cleaning up test pod...")
        subprocess.run(['kubectl', 'delete', 'pod', 'cva-timing-test'])

if __name__ == '__main__':
    test = TimingTest()
    test.run()