import psycopg2
from config_manager import config
db_cfg = config.get('database')
print(f"Connecting to {db_cfg.get('host')}:{db_cfg.get('port')} as {db_cfg.get('user')}")
try:
    conn = psycopg2.connect(
        dbname=db_cfg.get('name'), 
        user=db_cfg.get('user'), 
        password=db_cfg.get('password'), 
        host=db_cfg.get('host'), 
        port=db_cfg.get('port')
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        # Check if vector extension is created
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("PgVector extension ensured.")
        
        # Check table
        cur.execute("ALTER TABLE task_history ADD COLUMN IF NOT EXISTS task_embedding vector(1024);")
        print("task_embedding column added.")
        
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'task_history';")
        print([row[0] for row in cur.fetchall()])
except Exception as e:
    print(f"Error: {e}")
