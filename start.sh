#!/usr/bin/env bash
# ==============================================================================
# Catalyst Vector Alpha - Unified Startup Script (v4 - Improved)
# - OS-aware Docker start
# - PRESERVES logs with timestamps (doesn't delete)
# - Safer bash options, traps
# - Metrics-server enable for kubectl top
# - Single app entrypoint
# ==============================================================================

set -Eeuo pipefail

# --- Pretty output ------------------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
say()  { echo -e "${GREEN}$*${NC}"; }
warn() { echo -e "${YELLOW}$*${NC}"; }
fail() { echo -e "${RED}$*${NC}" >&2; }

trap 'fail "\n[ERROR] Line $LINENO failed. Exiting."' ERR

# --- Paths & env --------------------------------------------------------------
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$PROJECT_DIR/venv"
PY="$VENV_DIR/bin/python"
export PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

say "=== Starting Catalyst Vector Alpha System ==="

# --- 1) Dependency checks -----------------------------------------------------
warn "\n[1/6] Checking for required tools..."
check_command() {
  if ! command -v "$1" &>/dev/null; then
    fail "Error: Command '$1' not found. Install it and ensure it's in PATH."
    exit 1
  fi
}

check_command docker
check_command minikube
check_command kubectl

# --- 1a) Ensure Docker daemon is running (OS-aware) --------------------------
docker_running=true
if ! docker info &>/dev/null; then
  docker_running=false
  warn "Docker daemon is not running. Attempting to start..."

  UNAME="$(uname -s || true)"
  case "$UNAME" in
    Linux*)
      if command -v systemctl &>/dev/null; then
        sudo systemctl start docker || true
      elif command -v service &>/dev/null; then
        sudo service docker start || true
      fi
      ;;
    Darwin*)
      # macOS: require Docker Desktop
      if ! pgrep -f "Docker.app" &>/dev/null; then
        open -a Docker || true
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      warn "Windows detected. Please ensure Docker Desktop is running."
      ;;
  esac

  # Wait up to ~20s for Docker
  for i in {1..20}; do
    if docker info &>/dev/null; then docker_running=true; break; fi
    sleep 1
  done
fi

if [ "$docker_running" != true ]; then
  fail "Failed to start Docker daemon. Start it manually, then rerun."
  exit 1
fi
say "All required tools found and Docker is running."

# --- 2) Cleanup & Log Archival ------------------------------------------------
warn "\n[2/6] Preparing directories and archiving old logs..."
mkdir -p "$PROJECT_DIR/persistence_data" "$PROJECT_DIR/logs" "$PROJECT_DIR/logs/archive"

# Archive existing logs with timestamp (don't delete!)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "$PROJECT_DIR/logs/catalyst.log" ]; then
    mv "$PROJECT_DIR/logs/catalyst.log" "$PROJECT_DIR/logs/archive/catalyst_${TIMESTAMP}.log"
    say "Archived previous log to logs/archive/catalyst_${TIMESTAMP}.log"
fi

# Clean up __pycache__ only
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Optional: Clean old persistence data (commented out by default)
# rm -f "$PROJECT_DIR/persistence_data/"*.json || true

# Keep only last 10 archived logs (optional cleanup)
if ls "$PROJECT_DIR/logs/archive/"*.log 1> /dev/null 2>&1; then
    LOG_COUNT=$(ls -1 "$PROJECT_DIR/logs/archive/"*.log | wc -l)
    if [ "$LOG_COUNT" -gt 10 ]; then
        warn "Found $LOG_COUNT archived logs. Keeping only the 10 most recent..."
        ls -1t "$PROJECT_DIR/logs/archive/"*.log | tail -n +11 | xargs rm -f
    fi
fi

say "Cleanup complete. Fresh logs will be written to logs/catalyst.log"

# --- 3) Environment setup -----------------------------------------------------
warn "\n[3/6] Activating Python virtual environment..."
if [ -x "$PY" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  say "Virtual environment activated. Prometheus URL = $PROMETHEUS_URL"
else
  fail "Virtual environment not found at '$VENV_DIR'. Create it first:"
  echo "  python3 -m venv \"$VENV_DIR\" && source \"$VENV_DIR/bin/activate\" && pip install -r requirements.txt"
  exit 1
fi

# --- 4) Start/ensure Minikube -------------------------------------------------
warn "\n[4/6] Starting Minikube Kubernetes cluster..."
if ! minikube status &>/dev/null; then
  # let minikube pick a driver if not specified; allow override via env
  MINIKUBE_DRIVER_FLAG=""
  if [ "${MINIKUBE_DRIVER:-}" != "" ]; then
    MINIKUBE_DRIVER_FLAG="--driver=${MINIKUBE_DRIVER}"
  fi
  warn "Minikube is not running. Starting cluster..."
  minikube start $MINIKUBE_DRIVER_FLAG
else
  say "Minikube is already running."
fi

# Ensure metrics-server for `kubectl top`
if ! kubectl get deployment metrics-server -n kube-system &>/dev/null; then
  warn "Enabling metrics-server addon (needed for kubectl top)..."
  minikube addons enable metrics-server
  # give metrics-server a moment to come up
  sleep 5
fi
say "Kubernetes cluster is ready."

# --- 5) Preflight port check (optional but helpful) ---------------------------
warn "\n[5/6] Checking if port 5000 is free..."
if command -v lsof &>/dev/null; then
  if lsof -i :5000 -sTCP:LISTEN &>/dev/null; then
    warn "Port 5000 appears in use. If a previous run is stuck, you may need to stop it."
  fi
fi

# --- 6) Ensure sandbox container is running -----------------------------------
warn "\n[5.5/6] Checking CVA sandbox container..."
if ! docker ps | grep -q cva_sandbox; then
  if docker ps -a | grep -q cva_sandbox; then
    warn "Sandbox container exists but is stopped. Starting..."
    docker start cva_sandbox
  else
    warn "Sandbox container doesn't exist. Creating..."
    if [ -f "Dockerfile.cva-sandbox" ]; then
      docker build -t cva-sandbox -f Dockerfile.cva-sandbox .
      docker run -d --name cva_sandbox --memory="2g" --cpus="2" cva-sandbox
    else
      warn "Dockerfile.cva-sandbox not found. Sandbox features may not work."
    fi
  fi
else
  say "CVA sandbox container is running."
fi

# --- 7) Launch the application (single entrypoint) ----------------------------
warn "\n[6/6] Launching Catalyst Vector Alpha (app.py)..."
echo ""
say "=========================================="
say "  CVA Dashboard: http://127.0.0.1:5000"
say "  Logs: tail -f logs/catalyst.log"
say "  Press Ctrl+C to stop"
say "=========================================="
echo ""

# Use exec to replace the shell with the Python process (cleaner signals)
exec "$PY" "$PROJECT_DIR/app.py"
