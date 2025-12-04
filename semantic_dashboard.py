import streamlit as st
import pandas as pd
import json
import os
import time
import logging 
import requests
import psycopg2
import pandas as pd

# Configure logging to show info messages in the console/log file
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Constants and Configurations ---
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PERSISTENCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'persistence_data')
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'logs', 'swarm_activity.jsonl') # Updated path based on structure
# Fallback if log is in root persistence
if not os.path.exists(SWARM_ACTIVITY_LOG):
    SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'swarm_activity.jsonl')

PAUSED_AGENTS_FILE = os.path.join(PERSISTENCE_DIR, 'paused_agents.json')
ALERTS_FILE = os.path.join(PERSISTENCE_DIR, 'alerts.json')

# Ensure the persistence directory exists
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="CVA Dashboard") 

# --- Helper functions for loading data ---

def load_all_agent_states():
    """Loads the latest state for all agents from their JSON files."""
    agent_states = {}
    if not os.path.exists(PERSISTENCE_DIR):
        return {}
    
    agent_files = [f for f in os.listdir(PERSISTENCE_DIR) if f.startswith('agent_state_') and f.endswith('.json')]
        
    for filename in agent_files:
        filepath = os.path.join(PERSISTENCE_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                agent_name = state.get('name')
                if agent_name: 
                    agent_states[agent_name] = state
        except Exception as e:
            logging.error(f"Dashboard: Error loading {filename}: {e}")
    
    return agent_states

def load_swarm_activity_logs(limit=50):
    """Loads recent logs from the swarm_activity.jsonl file."""
    logs = []
    if not os.path.exists(SWARM_ACTIVITY_LOG):
        return logs
    
    try:
        with open(SWARM_ACTIVITY_LOG, 'r') as f:
            # Read lines efficiently (simulating tail)
            lines = f.readlines()
            for line in reversed(lines[-limit:]): # Last N lines, reversed
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue 
    except Exception as e:
        logging.error(f"Error reading logs: {e}")
    return logs

def load_alerts():
    """Loads active alerts from the persistence file."""
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, 'r') as f:
            content = f.read().strip()
            if not content: return []
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# --- Helper functions for Pause/Resume logic ---
def load_paused_agents():
    """Loads the list of paused agents from persistence."""
    if os.path.exists(PAUSED_AGENTS_FILE):
        try:
            with open(PAUSED_AGENTS_FILE, 'r') as f: 
                return json.load(f)
        except json.JSONDecodeError:
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
    """Toggles the pause state of an agent."""
    paused_agents = set(load_paused_agents()) 
    
    if action == 'pause':
        if agent_name not in paused_agents:
            paused_agents.add(agent_name)
            save_paused_agents_list_from_dashboard(paused_agents)
            st.toast(f"⏸️ Paused {agent_name}")
    elif action == 'resume':
        if agent_name in paused_agents:
            paused_agents.remove(agent_name)
            save_paused_agents_list_from_dashboard(paused_agents)
            st.toast(f"▶️ Resumed {agent_name}")
    
    # We don't strictly need st.rerun() here if using st.toast and auto-refresh, 
    # but it makes UI snappy.
    # st.rerun() 

# --- Dashboard Layout ---
st.title("🌌 Catalyst Vector Alpha: Control Center")
st.markdown("---")

# Auto-refresh
refresh_interval = 2
if st.checkbox("Auto-refresh", value=True):
    time.sleep(refresh_interval)
    st.rerun()

# Load Data
agent_states = load_all_agent_states()
paused_agents = load_paused_agents()
alerts = load_alerts()
logs = load_swarm_activity_logs()

# Initializing Session State for 'Info' button
if 'selected_agent_details' not in st.session_state:
    st.session_state.selected_agent_details = None

# --- Derived telemetry helpers ---
def _agent_latencies():
    """Approximate agent responsiveness via agent_state file mtimes."""
    latencies = {}
    now = time.time()
    if not os.path.exists(PERSISTENCE_DIR):
        return latencies
    for filename in os.listdir(PERSISTENCE_DIR):
        if filename.startswith("agent_state_") and filename.endswith(".json"):
            path = os.path.join(PERSISTENCE_DIR, filename)
            try:
                mtime = os.path.getmtime(path)
                age = max(0, now - mtime)
                name = filename.replace("agent_state_", "").replace(".json", "")
                latencies[name] = age
            except Exception:
                continue
    return latencies

def _fetch_tool_breakers():
    url = "http://localhost:5000/api/tool_breakers"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def _stability_counter():
    """Count restart events from metrics if available; fallback to logs."""
    # Prefer DB metrics
    try:
        from db_postgres import DB_CONFIG
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM metrics 
            WHERE metric_type IN ('circuit_breaker_reset','circuit_breaker_trip','supervisor_restart')
              AND timestamp > NOW() - INTERVAL '24 hours'
        """)
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0] or 0
    except Exception:
        pass
    # Fallback to log scan
    count = 0
    if os.path.exists(SWARM_ACTIVITY_LOG):
        try:
            with open(SWARM_ACTIVITY_LOG, "r") as f:
                for line in f.readlines()[-500:]:
                    if "RESTART" in line.upper():
                        count += 1
        except Exception:
            pass
    return count

agent_latencies = _agent_latencies()
breaker_status = _fetch_tool_breakers()
stability_count = _stability_counter()


# --- MAIN UI COLUMNS ---
col1, col2 = st.columns([2, 1.2])

# --- LEFT COLUMN: AGENTS ---
with col1:
    st.subheader(f"🤖 Active Agents ({len(agent_states)})")
    
    # Custom Table Layout using Columns
    # Header
    h1, h2, h3, h4, h5 = st.columns([1.5, 3, 0.8, 1, 0.5])
    h1.markdown("**Name**")
    h2.markdown("**Current Intent**")
    h3.markdown("**Status**")
    h4.markdown("**Controls**")
    h5.markdown("**Info**")
    st.markdown("---")

    if agent_states:
        sorted_names = sorted(agent_states.keys())
        for name in sorted_names:
            state = agent_states[name]
            
            # Data prep
            full_intent = state.get('current_intent', 'Idle')
            display_intent = (full_intent[:50] + '...') if len(full_intent) > 50 else full_intent
            
            is_paused = name in paused_agents
            status_color = "🔴 Paused" if is_paused else "🟢 Active"
            
            # Row
            c1, c2, c3, c4, c5 = st.columns([1.5, 3, 0.8, 1, 0.5])
            
            c1.code(name.replace("ProtoAgent_", "").replace("_instance_1", ""), language="text")
            c2.write(display_intent)
            c3.write(status_color)
            
            # Controls
            with c4:
                b_col1, b_col2 = st.columns(2)
                if is_paused:
                    b_col2.button("▶️", key=f"res_{name}", on_click=toggle_agent_pause_from_dashboard, args=(name, 'resume'))
                else:
                    b_col1.button("⏸️", key=f"pau_{name}", on_click=toggle_agent_pause_from_dashboard, args=(name, 'pause'))
            
            # Info
            with c5:
                if st.button("ℹ️", key=f"inf_{name}"):
                    # Prepare details for the bottom section
                    memories = state.get('memetic_kernel', {}).get('memories', [])
                    last_mem = memories[-1].get('content') if memories else "No memories"
                    
                    st.session_state.selected_agent_details = {
                        "Name": name,
                        "Full Intent": full_intent,
                        "Gradient": state.get('sovereign_gradient', {}),
                        "Last Memory": last_mem,
                        "Config": state.get('eidos_spec', {})
                    }

    else:
        st.info("No agents detected. Is CVA running?")

    # --- Selected Agent Details (Bottom of Left Col) ---
    if st.session_state.selected_agent_details:
        st.markdown("---")
        d = st.session_state.selected_agent_details
        st.subheader(f"🔍 Details: {d['Name']}")
        st.info(f"**Intent:** {d['Full Intent']}")
        
        with st.expander("🧠 Last Memory", expanded=True):
            st.write(d['Last Memory'])
            
        with st.expander("⚙️ Sovereign Gradient"):
            st.json(d['Gradient'])
            
        if st.button("Close Details"):
            st.session_state.selected_agent_details = None
            st.rerun()

# --- RIGHT COLUMN: ALERTS & LOGS ---
with col2:
    # --- SYSTEM ALERTS SECTION ---
    st.subheader("🔥 Active Alerts")
    
    if alerts:
        # Check if we have critical flight updates
        flight_alerts = [a for a in alerts if a.get('type') == 'flight_update']
        if flight_alerts:
            st.error(f"✈️ {len(flight_alerts)} Flight Update(s) Detected!")
            
        for alert in reversed(alerts[-5:]): # Show last 5
            icon = "🚨"
            atype = alert.get('type', 'unknown')
            
            if atype == 'flight_update': icon = "✈️"
            elif atype == 'invoice': icon = "💰"
            elif 'cpu' in atype: icon = "💻"
            
            with st.expander(f"{icon} {atype.upper()}", expanded=True):
                st.caption(f"Source: {alert.get('source')}")
                st.write(f"**Subject:** {alert.get('subject', 'N/A')}")
                
                details = alert.get('details', {})
                if details:
                    st.json(details)
                
                ts = alert.get('timestamp')
                if ts:
                    st.caption(f"Time: {time.ctime(ts)}")
    else:
        st.success("System Clear. No active alerts.")

    st.markdown("---")

    # --- LIVE LOGS SECTION ---
    st.subheader("📜 Live Activity Log")
    
    log_container = st.container(height=400) # Scrollable container
    with log_container:
        for log in logs:
            # Format log entry
            event = log.get('event_type', 'UNKNOWN')
            desc = log.get('description', '')
            src = log.get('source', '').replace("ProtoAgent_", "").replace("_instance_1", "")
            
            # Color coding for log types
            emoji = "🔹"
            if "ALERT" in event: emoji = "🚨"
            elif "TOOL" in event: emoji = "🛠️"
            elif "PLAN" in event: emoji = "🧠"
            elif "ERROR" in event or "failed" in desc.lower(): emoji = "❌"
            elif "SUCCESS" in event or "completed" in desc.lower(): emoji = "✅"
            elif "CONFLICT" in desc: emoji = "⚠️"

            st.markdown(f"**{emoji} {src}**: `{event}`")
            st.caption(desc)
            st.markdown("---")

# --- SECOND ROW: Telemetry & Health ---
st.markdown("---")
tele_col1, tele_col2, tele_col3 = st.columns([1.2, 1.2, 1])

with tele_col1:
    st.subheader("⏱️ Agent Speedometer")
    if agent_latencies:
        planner_latency = agent_latencies.get("ProtoAgent_Planner_instance_1")
        worker_latency = agent_latencies.get("ProtoAgent_Worker_instance_1")
        st.metric("Planner freshness (s)", f"{planner_latency:.1f}" if planner_latency is not None else "N/A")
        st.metric("Worker freshness (s)", f"{worker_latency:.1f}" if worker_latency is not None else "N/A")
        st.caption("Lower is fresher (time since last state write).")
    else:
        st.info("No agent state files found.")

with tele_col2:
    st.subheader("🛠️ Tool Health Board")
    if breaker_status and breaker_status.get("status") == "ok":
        tools = breaker_status.get("data", {}).get("tools", [])
        healthy = [t for t in tools if t.get("status") == "healthy"]
        broken = [t for t in tools if t.get("status") == "broken"]
        expired = [t for t in tools if t.get("status") == "expired"]
        st.success(f"Healthy: {len(healthy)}")
        if broken:
            st.error(f"Broken: {len(broken)}")
            for t in broken[:6]:
                st.write(f"🔴 {t.get('tool')} (until {int(t.get('broken_until',0))})")
        if expired:
            st.warning(f"Expired: {len(expired)} (awaiting reset)")
    else:
        st.info("Tool breaker status unavailable (is API up?).")

with tele_col3:
    st.subheader("🧭 Stability Counter")
    st.metric("Events (24h)", stability_count)
    st.caption("Restart / breaker events (metrics preferred, logs fallback).")
