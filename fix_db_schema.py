from database import cva_db
from db_postgres import execute_query

print("Checking db connection...")
try:
    # We must be careful - some postgres roles are not superusers and can't create extensions
    # execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Executing ALTER TABLE to add task_embedding...")
    execute_query("ALTER TABLE task_history ADD COLUMN IF NOT EXISTS task_embedding vector(1024);")
    print("Success! task_embedding added.")
except Exception as e:
    print(f"Failed to alter table: {e}")
