import sqlite3
import json
import time
import os

DB_PATH = "persistence_data/cva.db"

def get_db():
    return sqlite3.connect(DB_PATH)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def view_brain():
    while True:
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("SELECT value_json FROM system_state WHERE key='swarm_state'")
            row = cursor.fetchone()
            
            clear_screen()
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║         CVA CEREBRAL MONITOR (SQLite Live View)              ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            
            if row:
                state = json.loads(row[0])
                cycle = state.get('current_action_cycle_id', 'Unknown')
                print(f"⏱️  Cycle: {cycle}")
                print(f"🔄 Running: {state.get('is_running', False)} | Paused: {state.get('is_paused', False)}")
                print("-" * 62)
                
                agents = state.get('agent_instances', {})
                print(f"{'AGENT':<32} | {'ROLE':<25}")
                print("-" * 62)
                
                for name, data in agents.items():
                    role = data.get('eidos_spec', {}).get('role', 'Unknown') if isinstance(data, dict) else 'Active'
                    print(f"{name:<32} | {role:<25}")
            else:
                print("⚠️  No Swarm State found.")
            
            print(f"\n🛠️  RECENT TOOL USAGE")
            print("-" * 62)
            cursor.execute("SELECT tool_name, success, execution_time_seconds FROM tool_usage ORDER BY id DESC LIMIT 5")
            tools = cursor.fetchall()
            for t in tools:
                status = "✅" if t[1] else "❌"
                print(f"{status} {t[0]:<45} ({t[2]:.2f}s)")
            
            print(f"\n📊 STATS")
            print("-" * 62)
            cursor.execute("SELECT COUNT(*), SUM(success) FROM tool_usage")
            total, success = cursor.fetchone()
            total = total or 0
            success = success or 0
            rate = (success/total*100) if total > 0 else 0
            print(f"Total: {total} | Success: {rate:.1f}%")
            
            print("-" * 62)
            print("Ctrl+C to exit")
            conn.close()
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    view_brain()
