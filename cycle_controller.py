import os
import json
import time
import random
import datetime

# --- Configuration ---
# Define the persistence directory where logs and control files are stored
PERSISTENCE_DIR = 'persistence_data'
# Define the path to the system pause flag file
SYSTEM_PAUSE_FILE = os.path.join(PERSISTENCE_DIR, 'system_pause.flag')
# Define the path to the main swarm activity log file
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'swarm_activity.jsonl') # Adjust to .jsonl if that's the actual filename

# How often (in seconds) the controller checks the logs for new requests
RESPONSE_INTERVAL_SECONDS = 5
# A small delay (in seconds) to simulate human processing time before writing a response
RESPONSE_DELAY_SECONDS = 1
# The possible responses the simulated human can provide
HUMAN_RESPONSE_OPTIONS = ["Approve planning", "Request change"]

# --- Global state for tracking ---
# Keeps track of the last position read in the log file to only process new entries
last_log_position = 0
# Stores cycle IDs that have already been responded to, to prevent duplicate responses
responded_cycle_ids = set() 

def timestamp_now():
    """
    Returns the current UTC timestamp in ISO 8601 format (e.g., '2025-07-19T12:34:56Z').
    This ensures consistency with the main system's timestamps.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def unpause_system():
    """
    Removes the system_pause.flag file, effectively unpausing the main system.
    This is called by the controller if it detects the system is paused.
    """
    try:
        if os.path.exists(SYSTEM_PAUSE_FILE):
            os.remove(SYSTEM_PAUSE_FILE)
            print(f"[Controller] --- SYSTEM UNPAUSED: '{SYSTEM_PAUSE_FILE}' removed. ---")
            return True
        return False
    except Exception as e:
        print(f"[Controller] ERROR: Failed to remove system pause file: {e}")
        return False

def generate_human_response_file(cycle_id: str, response_content: str):
    """
    Generates a mock human response JSON file in the persistence directory.
    The main system (CatalystVectorAlpha) will look for this file to process human input.

    Args:
        cycle_id (str): The unique identifier of the cognitive cycle that requested human input.
        response_content (str): The simulated human's response (e.g., "Approve planning").
    """
    response_file_name = f"control_human_response_{cycle_id}.json"
    response_file_path = os.path.join(PERSISTENCE_DIR, response_file_name)
    
    response_data = {
        "response": response_content,
        "timestamp": timestamp_now(),
        "generated_by": "CycleController" # Indicates this response came from the controller
    }

    try:
        with open(response_file_path, 'w') as f:
            json.dump(response_data, f, indent=2)
        print(f"[Controller] Generated human response for cycle {cycle_id}: '{response_content}' in {response_file_name}")
        return True
    except Exception as e:
        print(f"[Controller] ERROR: Failed to write human response file {response_file_name}: {e}")
        return False

def process_log_for_human_requests():
    """
    Reads new lines from the SWARM_ACTIVITY_LOG, identifies human input requests,
    and returns a list of unique requests that need a response.
    """
    global last_log_position
    requests_found = []

    try:
        # Check if the log file exists. If not, return empty.
        if not os.path.exists(SWARM_ACTIVITY_LOG):
            return []

        with open(SWARM_ACTIVITY_LOG, 'r') as f:
            # Seek to the last known position to read only new content
            f.seek(last_log_position) 
            new_lines = f.readlines()
            # Update the last_log_position for the next iteration
            last_log_position = f.tell() 

        for line in new_lines:
            try:
                log_entry = json.loads(line.strip())
                event_type = log_entry.get('event_type')
                details = log_entry.get('details', {})
                
                # Look for Level 1 or Level 2 human input requests from the main system
                if event_type in ["HUMAN_INPUT_REQUESTED_LEVEL1", "HUMAN_INPUT_PENDING_LEVEL2"]:
                    request_cycle_id = details.get('cycle_id')
                    # The 'human_request_counter' indicates the escalation level (0 for Level 1, 1 for Level 2)
                    human_request_counter = details.get('human_request_counter') 

                    # Only process requests that haven't been responded to yet
                    # And only respond to Level 1 or Level 2 requests (Level 3 leads to system pause)
                    if request_cycle_id and request_cycle_id not in responded_cycle_ids:
                        if human_request_counter in [0, 1]:
                            requests_found.append({
                                "cycle_id": request_cycle_id,
                                "human_request_counter": human_request_counter
                            })
                            # Add to responded_cycle_ids set to prevent duplicate responses for this cycle
                            responded_cycle_ids.add(request_cycle_id) 
            except json.JSONDecodeError:
                # Silently skip malformed JSON lines in the log (can happen if log is written partially)
                # print(f"[Controller] Warning: Malformed JSON line in log: {line.strip()[:100]}...")
                pass 
            except Exception as e:
                print(f"[Controller] Error processing log line: {e}")

    except FileNotFoundError:
        print(f"[Controller] Warning: Swarm activity log not found at {SWARM_ACTIVITY_LOG}")
    except Exception as e:
        print(f"[Controller] Error reading swarm activity log: {e}")
        
    return requests_found

def main_loop():
    """
    The main loop of the Cycle Controller.
    Continuously monitors the system log and responds to human input requests.
    """
    print(f"[Controller] Starting Cycle Controller. Monitoring '{SWARM_ACTIVITY_LOG}' for human input requests.")
    print(f"[Controller] Will generate responses every {RESPONSE_INTERVAL_SECONDS} seconds.")
    
    # Ensure the persistence directory exists for storing response files
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)

    # At startup, check if the system is already paused and attempt to unpause it
    # This is useful if the main system was paused in a previous run.
    if os.path.exists(SYSTEM_PAUSE_FILE):
        print("[Controller] Found existing system pause flag at startup. Attempting to unpause.")
        unpause_system()

    while True:
        # Process new log entries to find human input requests
        requests = process_log_for_human_requests()
        
        if requests:
            print(f"[Controller] Detected {len(requests)} new human input requests.")
            for req in requests:
                cycle_id = req['cycle_id']
                # Randomly choose a response from the predefined options
                response_choice = random.choice(HUMAN_RESPONSE_OPTIONS)
                
                # Introduce a small delay to simulate human reaction time
                time.sleep(RESPONSE_DELAY_SECONDS) 
                
                # Generate the response file. The main system's Level 2 logic
                # will detect this file and process the input, potentially unpausing itself.
                if generate_human_response_file(cycle_id, response_choice):
                    # No direct unpause from controller needed here, as the main system handles it
                    # after consuming the response file.
                    pass 
                
        # Wait for the next interval before checking logs again
        time.sleep(RESPONSE_INTERVAL_SECONDS)

if __name__ == "__main__":
    # This block executes when the script is run directly
    main_loop()


