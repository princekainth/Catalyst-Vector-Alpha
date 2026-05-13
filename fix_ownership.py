import psycopg2

try:
    conn = psycopg2.connect(dbname="cva_db", user="postgres", host="127.0.0.1")
    conn.autocommit = True
    with conn.cursor() as cur:
        # Give cva_user ownership of the entire database and all tables
        cur.execute("ALTER DATABASE cva_db OWNER TO cva_user;")
        cur.execute("ALTER TABLE task_history OWNER TO cva_user;")
        cur.execute("ALTER TABLE task_history ADD COLUMN IF NOT EXISTS task_embedding vector(1024);")
        print("Success! Ownership transferred and column added.")
except Exception as e:
    print(f"Error: {e}")
