# Quickstart: CVA Cloud Beta

Get up and running with Catalyst Vector Alpha (CVA) in 3 minutes.

## 1. Environment Setup

Ensure you have a Kubernetes cluster (Minikube/KIND) and Ollama running locally.

```bash
# Clone the repository
git clone https://github.com/catalyst-vector/cva-alpha.git
cd cva-alpha

# Install dependencies
pip install -r requirements.txt

# Start the dashboard backend
./start.sh
```

## 2. Configuration

CVA uses environment variables for primary safety toggles:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CVA_APPROVAL_MODE` | `human` | Set to `auto` for fully autonomous remediation (Unsafe!) |
| `CVA_ALLOWED_SERVICES` | `""` | List of systemd services the agent can restart |
| `CVA_DEMO_MODE` | `0` | Set to `1` to clean up log output for demos |

## 3. Launching the Agent Swarm

Launch the cognitive loop in a separate terminal:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 catalyst_vector_alpha.py --mode production
```

## 4. Run your first Incident Detection

CVA will automatically begin monitoring your default Kubernetes namespace. You can verify detection logic using the integrated benchmark:

```bash
./demo.sh
```

---
**Next Steps:**
- Read [DEMO.md](DEMO.md) for walkthrough instructions.
- Read [SAFETY_MODEL.md](SAFETY_MODEL.md) to understand how CVA prevents outages.
