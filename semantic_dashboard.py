import streamlit as st
import pandas as pd
import json
import os
import time
import logging 

# Configure logging to show info messages in the console/log file
# This will direct messages to the terminal where Streamlit is run.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Constants and Configurations ---
# Determine the project root directory.
# Since this script is now in the project root itself, PROJECT_ROOT_DIR is just its own directory.
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PERSISTENCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'persistence_data')

SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'swarm_activity.jsonl')
PAUSED_AGENTS_FILE = os.path.join(PERSISTENCE_DIR, 'paused_agents.json')

# Ensure the persistence directory exists (good practice for all components)
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

st.set_page_config(layout="wide") # Use wide layout for better visualization

# --- Helper functions for loading data (from persistence_data) ---
def load_all_agent_states():
    """Loads the latest state for all agents from their JSON files."""
    agent_states = {}
    logging.info(f"Dashboard: Starting load cycle. Looking in: {PERSISTENCE_DIR}") # Using logging
    if not os.path.exists(PERSISTENCE_DIR):
        logging.info(f"Dashboard: Persistence directory NOT found: {PERSISTENCE_DIR}. Returning empty.") # Using logging
        return {}
    
    agent_files = [f for f in os.listdir(PERSISTENCE_DIR) if f.startswith('agent_state_') and f.endswith('.json')]
    logging.info(f"Dashboard: Found {len(agent_files)} agent state files: {agent_files}") # Using logging
    
    if not agent_files:
        logging.info("Dashboard: No agent_state_*.json files found matching criteria in directory.") # Using logging
        
    for filename in agent_files:
        filepath = os.path.join(PERSISTENCE_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                agent_name = state.get('name')
                if agent_name: 
                    agent_states[agent_name] = state
                    logging.info(f"Dashboard: Loaded OK: '{agent_name}' from {filename}") # Using logging
                else:
                    logging.warning(f"Dashboard: SKIPPED: File {filename} has no 'name' key or it's empty.") # Using logging
        except (json.JSONDecodeError, FileNotFoundError) as e:
            st.warning(f"Dashboard: Could not load agent state from {filename}: {e}") # Keep st.warning for dashboard display
            logging.error(f"Dashboard: ERROR: JSON Decode Error/File Not Found for {filename}: {e}") # Using logging for console error
        except Exception as e:
            st.warning(f"Dashboard: Unexpected error loading agent state from {filename}: {e}")
            logging.error(f"Dashboard: ERROR: Unexpected error for {filename}: {e}") # Using logging for console error
    
    logging.info(f"Dashboard: Final count of agents loaded into dashboard: {len(agent_states)}") # Using logging
    return agent_states

def load_swarm_activity_logs():
    """Loads all logs from the swarm_activity.jsonl file."""
    logs = []
    if not os.path.exists(SWARM_ACTIVITY_LOG):
        return logs
    
    with open(SWARM_ACTIVITY_LOG, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue # Skip malformed lines
    return logs

# --- Helper functions for Pause/Resume logic ---
def load_paused_agents():
    """Loads the list of paused agents from persistence."""
    if os.path.exists(PAUSED_AGENTS_FILE):
        try:
            # --- CRITICAL FIX: Corrected typo here ---
            with open(PAUSED_AGENTS_FILE, 'r') as f: 
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted {PAUSED_AGENTS_FILE}. Starting with no agents paused.")
            return []
    return []

def save_paused_agents_list_from_dashboard(paused_agents_list):
    """Saves the list of paused agents to persistence."""
    try:
        with open(PAUSED_AGENTS_FILE, 'w') as f:
            json.dump(list(paused_agents_list), f, indent=2)
    except Exception as e:
        st.error(f"Dashboard: Error saving paused agents list: {e}")

def toggle_agent_pause_from_dashboard(agent_name, action):
    """
    Toggles the pause state of an agent by modifying a shared JSON file.
    'action' can be 'pause' or 'resume'.
    """
    # FIX: Call the correctly named function
    paused_agents = set(load_paused_agents()) 
    
    if action == 'pause':
        if agent_name in paused_agents:
            st.info(f"Agent '{agent_name}' is already paused.")
        else:
            paused_agents.add(agent_name)
            save_paused_agents_list_from_dashboard(paused_agents)
            st.success(f"Agent '{agent_name}' paused successfully.")
            st.rerun()
    elif action == 'resume':
        if agent_name not in paused_agents:
            st.info(f"Agent '{agent_name}' is not currently paused.")
        else:
            paused_agents.remove(agent_name)
            save_paused_agents_list_from_dashboard(paused_agents)
            st.success(f"Agent '{agent_name}' resumed successfully.")
            st.rerun()
    else:
        st.error("Invalid action. Use 'pause' or 'resume'.")

# --- Dashboard Layout ---
st.title("🌌 The First Semantic OS: Live Dashboard")
st.markdown("---")

# Auto-refresh mechanism (adjust interval as needed)
refresh_interval_seconds = 2 # Refresh every 2 seconds
time_placeholder = st.empty() # Placeholder for current time display

st.sidebar.header("Controls")
if st.sidebar.button("Manual Refresh"):
    st.rerun() # Force a rerun

st.sidebar.text(f"Auto-refresh: {refresh_interval_seconds}s")

# Load agent states and logs globally for the current rerun cycle
agent_states = load_all_agent_states()
paused_agents = load_paused_agents() # Load paused agents for display


# Add a session state variable to store the selected agent's full details for display
if 'selected_agent_details' not in st.session_state:
    st.session_state.selected_agent_details = None


# Main content columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Active Agents Overview")
    
    # --- Dynamic Agent Table with Pause/Resume Buttons ---
    # Define columns for the custom table header - SIMPLIFIED FOR READABILITY
    col_name_h, col_role_h, col_intent_h, col_status_h, col_pause_h, col_resume_h, col_info_h = st.columns([1.5, 0.7, 3.0, 0.7, 0.25, 0.25, 0.4])
    
    # Display header row
    with col_name_h: st.write("**Agent Name**")
    with col_role_h: st.write("**Role**")
    with col_intent_h: st.write("**Current Intent**")
    with col_status_h: st.write("**Status**")
    with col_pause_h: st.write("**P**") # Abbreviated header for Pause button column
    with col_resume_h: st.write("**R**") # Abbreviated header for Resume button column
    with col_info_h: st.write("**Info**") # Header for Info button

    st.markdown("---") # Separator below the header

    if agent_states:
        # Sort agents by name for consistent display
        sorted_agent_names = sorted(agent_states.keys())
        for name in sorted_agent_names:
            state = agent_states[name]
            
            # Extract agent details
            current_intent_full = state.get('current_intent', 'N/A')
            # TRUNCATE CURRENT INTENT FOR DISPLAY IN TABLE - MORE AGGRESSIVE
            current_intent_display = current_intent_full
            if len(current_intent_display) > 50: # Truncate to 50 chars for table display
                current_intent_display = current_intent_display[:47] + "..."
            
            # Location, Gradient, Last Memory are removed from this table for readability
            # but their full versions will be available via the Info button.
            
            # Determine agent status and button disabled states
            status_text = "PAUSED" if name in paused_agents else "ACTIVE"
            is_paused_status = (name in paused_agents)

            # Create columns for the current agent's row (using adjusted widths)
            col_name, col_role, col_intent, col_status, col_pause, col_resume, col_info = st.columns([1.5, 0.7, 3.0, 0.7, 0.25, 0.25, 0.4])

            with col_name:
                st.write(name)
            with col_role:
                st.write(state.get('eidos_spec', {}).get('role', 'N/A'))
            with col_intent:
                st.write(current_intent_display) # Use display version
            with col_status:
                st.write(status_text)
            with col_pause:
                st.button(
                    "⏸️",
                    key=f"pause_{name}", # Unique key for each button
                    on_click=toggle_agent_pause_from_dashboard,
                    args=(name, 'pause',), # Pass agent name and action to the callback
                    disabled=is_paused_status # Disable if agent is already paused
                )
            with col_resume:
                st.button(
                    "▶️",
                    key=f"resume_{name}", # Unique key for each button
                    on_click=toggle_agent_pause_from_dashboard,
                    args=(name, 'resume',), # Pass agent name and action to the callback
                    disabled=not is_paused_status # Disable if agent is already active
                )
            with col_info:
                # Store full details in session_state when info button is clicked
                if st.button("ℹ️", key=f"info_{name}"):
                    # Get full details including those not in main table
                    full_details = {
                        "Name": name,
                        "Role": state.get('eidos_spec', {}).get('role', 'N/A'),
                        "Full Current Intent": current_intent_full, # Store full intent
                        "Location": state.get('location', 'N/A'),
                        "Gradient": state.get('sovereign_gradient'), # This is the full dict for Gradient
                        "Last Memory": state.get('memetic_kernel', {}).get('memories', [])[-1].get('content', 'No memories yet.') if state.get('memetic_kernel', {}).get('memories') else 'No memories yet.'
                    }
                    st.session_state.selected_agent_details = full_details
                    st.rerun() # Re-run to display details below

        st.markdown("---") # Separator at the bottom of the table
    else:
        st.info("No active agents found. Run 'catalyst_vector_alpha.py' to spawn agents.")

    # --- Display Selected Agent Details Section ---
    if st.session_state.selected_agent_details:
        st.subheader(f"Details for {st.session_state.selected_agent_details['Name']}")
        
        details = st.session_state.selected_agent_details
        
        # Format gradient for better display in the details section
        gradient_display = "None"
        if details.get('Gradient'):
            g = details['Gradient']
            gradient_display = (
                f"**Vector:** {g.get('autonomy_vector', 'N/A')}\n"
                f"**Ethical Constraints:** {', '.join(g.get('ethical_constraints', []))}\n"
                f"**Self-Correction Protocol:** {g.get('self_correction_protocol', 'N/A')}\n"
                f"**Override Threshold:** {g.get('override_threshold', 'N/A')}"
            )

        st.write(f"**Role:** {details.get('Role')}")
        st.write(f"**Location:** {details.get('Location')}")
        st.write(f"**Full Current Intent:**")
        st.info(details.get('Full Current Intent')) # Use st.info for a visually distinct block
        st.write(f"**Last Memory:**")
        st.info(details.get('Last Memory')) # Use st.info for a visually distinct block
        st.write(f"**Sovereign Gradient:**")
        st.info(gradient_display) # Use st.info for a visually distinct block
        
        st.markdown("---")
        if st.button("Clear Details"):
            st.session_state.selected_agent_details = None
            st.rerun()

# ... (rest of the semantic_dashboard.py code for logs, conflict monitor, etc., remains the same) ...