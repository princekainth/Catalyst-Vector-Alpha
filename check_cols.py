import subprocess
import os
import yaml

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

result = subprocess.run(
    ['psql', '-U', user, '-h', host, '-p', port, '-d', db, '-c', "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'task_history';"],
    env=env,
    capture_output=True,
    text=True
)
print(result.stdout)
if result.stderr: print("ERR:", result.stderr)
