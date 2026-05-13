import yaml
import subprocess
import os

with open('config/dev.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_cfg = config.get('database', {})
user = db_cfg.get('user', 'cva_user')
password = db_cfg.get('password', 'cva_pass')
db = db_cfg.get('name', 'cva_db')
host = db_cfg.get('host', '127.0.0.1')
port = str(db_cfg.get('port', 5432))

env = os.environ.copy()
env['PGPASSWORD'] = password

print("Adding task_embedding column directly...")
result = subprocess.run(
    ['psql', '-U', user, '-h', host, '-p', port, '-d', db, '-c', "ALTER TABLE task_history ADD COLUMN task_embedding vector(1024);"],
    env=env,
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error:\n{result.stderr}")
else:
    print(f"Success:\n{result.stdout}")
