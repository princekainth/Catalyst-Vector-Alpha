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

print(f"Running psql migration for {user}@{host}:{port}/{db}...")
result = subprocess.run(
    ['psql', '-U', user, '-h', host, '-p', port, '-d', db, '-f', 'migrate_to_postgres.sql'],
    env=env,
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error:\n{result.stderr}")
else:
    print(f"Success:\n{result.stdout}")
