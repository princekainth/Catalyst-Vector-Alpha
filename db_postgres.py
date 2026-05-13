import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor, Json
from contextlib import contextmanager
import os
import json
from config_manager import get_config

_cfg = get_config()
_db_cfg = (_cfg.get("database") or {})

DB_CONFIG = {
    'host': _db_cfg.get('host', 'localhost'),
    'database': _db_cfg.get('database', 'cva_db'),
    'user': _db_cfg.get('user', 'cva_user'),
    'password': _db_cfg.get('password', 'your_secure_password_here'),
    'port': _db_cfg.get('port', 5432)
}

pool = None
_POOL_ERR = None
# Allow CI/test environments to skip Postgres
if os.getenv("CVA_NO_POSTGRES") or os.getenv("GITHUB_ACTIONS") or os.getenv("SKIP_POSTGRES"):
    _POOL_ERR = "Postgres disabled via env"
else:
    try:
        pool = ThreadedConnectionPool(10, 20, **DB_CONFIG)
    except Exception as e:
        _POOL_ERR = e

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    if pool is None:
        raise RuntimeError(f"Postgres pool unavailable: {_POOL_ERR}")
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        pool.putconn(conn)

def execute_query(query, params=None, fetch=False):
    """Execute a query with automatic connection handling"""
    if pool is None:
        raise RuntimeError(f"Postgres pool unavailable: {_POOL_ERR}")
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return cursor.rowcount

def health_check():
    """Check database connectivity"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False

def close_pool():
    """Shut down the database connection pool."""
    global pool
    if pool:
        try:
            pool.closeall()
            print("Database connection pool closed.")
        except Exception as e:
            print(f"Error closing database pool: {e}")
        pool = None
