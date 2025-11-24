import subprocess
import tempfile
import os
import shlex

# The name of the Docker container we created earlier
CONTAINER_NAME = "cva_sandbox"

def execute_terminal_command(command: str, timeout: int = 30) -> dict:
    """
    Executes a shell command inside the sandbox container.
    This is the 'hands' of the agent.
    """
    print(f"📦 SANDBOX EXEC: {command}")
    
    # 1. Safety Check: Is the container actually running?
    try:
        check = subprocess.run(
            f"docker inspect -f '{{{{.State.Running}}}}' {CONTAINER_NAME}",
            shell=True, capture_output=True, text=True
        )
        if check.stdout.strip() != "true":
            return {"error": f"Container {CONTAINER_NAME} is not running. Please run ./setup_sandbox.sh"}
    except Exception:
        return {"error": "Docker check failed. Is Docker installed/running?"}

    # 2. Wrap command to run in bash inside docker
    # shlex.quote ensures that complex strings (like python scripts) don't break the bash command
    # We pass the command to /bin/bash -c inside the container
    docker_cmd = ["docker", "exec", CONTAINER_NAME, "/bin/bash", "-c", command]
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60 # 60s timeout prevents the agent from freezing forever
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout[:5000], # Limit output size to prevent log flooding
            "stderr": result.stderr[:5000],
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (60s limit)."}
    except Exception as e:
        return {"error": f"System error: {str(e)}"}

def write_sandbox_file(filepath: str, content: str) -> dict:
    """
    Writes code or text to a file inside the sandbox. 
    Uses 'docker cp' to avoid escaping issues with complex code.
    """
    print(f"📝 SANDBOX WRITE: {filepath}")
    
    tmp_path = None
    try:
        # 1. Create a temp file on your Host (Mac)
        # delete=False because we need to copy it before closing
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        # 2. Copy from Host -> Container
        subprocess.run(
            ["docker", "cp", tmp_path, f"{CONTAINER_NAME}:{filepath}"],
            check=True
        )
        
        # 3. Cleanup Host temp file
        os.remove(tmp_path)
        
        return {"status": "success", "message": f"Wrote {len(content)} bytes to {filepath}"}
        
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": f"Failed to write file: {str(e)}"}