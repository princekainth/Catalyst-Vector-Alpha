#!/bin/bash

# ==============================================================================
# Catalyst Vector Alpha - Shutdown Script
# ==============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}--- Shutting Down Catalyst Vector Alpha Services ---${NC}"

# Stop the Minikube cluster
echo "Stopping Minikube cluster..."
minikube stop

# Optional: Find and kill the python app.py process if it's still running
# This is a safety net in case you ran it in the background.
echo "Checking for running app.py process..."
P_PID=$(pgrep -f "python3.*app.py")
if [ -n "$P_PID" ]; then
    echo "Stopping Catalyst app (PID: $P_PID)..."
    kill $P_PID
fi

echo -e "${GREEN}System shutdown complete.${NC}"