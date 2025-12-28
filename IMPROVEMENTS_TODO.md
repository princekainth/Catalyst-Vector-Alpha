# CVA Future Improvements

## ✅ COMPLETED THIS SESSION

### Event-Driven Architecture
- [x] `_should_be_idle()` gate - checks K8s health
- [x] All 5 agents respect idle mode
- [x] System sleeps when healthy, activates on problems

### Performance
- [x] OllamaLLMIntegration singleton (11 → 1 connection)
- [x] SharedMemory singleton (3 → 1 init)
- [x] AgentFactory singleton (2 → 1 init)
- [x] Tool embeddings cached to disk
- [x] Memory consultation skipped for idle tasks

### Code Quality
- [x] ~400 print statements → proper logging
- [x] DEBUG spam removed
- [x] Duplicate log messages fixed
- [x] Bare `except:` → `except Exception:`

### Bug Fixes
- [x] `cva_db` → `db` attribute
- [x] Idle outcome logged as INFO not ERROR
- [x] Memory context pollution fixed
- [x] GmailAgent duplicate logs fixed

---

## 🔮 FUTURE IMPROVEMENTS

### Config Migration (Low Effort, High Value)
Replace hardcoded values with `config.py`:
```python
# Before
timeout = 60
# After
from config import config
timeout = config.LLM_TIMEOUT
```

Files to update:
- [ ] agents.py - timeouts, cooldowns, intervals
- [ ] catalyst_vector_alpha.py - loop intervals
- [ ] curiosity_loop.py - exploration interval
- [ ] students/k8s_agent.py - remediation timeouts

### Resource Cleanup (Medium Effort)
Add `finally` blocks for proper cleanup:
- [ ] Database connections
- [ ] File handles
- [ ] Network connections

### Connection Pooling (Medium Effort)
- [ ] Ollama connection pool for parallel requests
- [ ] Database connection pool

### Testing (High Effort, High Value)
- [ ] Unit tests for idle gates
- [ ] Integration tests for remediation flow
- [ ] Load tests for multi-agent cycles

### Monitoring (Medium Effort)
- [ ] Prometheus metrics integration
- [ ] Grafana dashboard templates
- [ ] Alert rules for agent failures

---

## 📊 METRICS ACHIEVED

| Metric | Before | After |
|--------|--------|-------|
| Ollama connections | 11+ | 1 |
| Embedding calls/20s | 40+ | 4 |
| Agent execution (idle) | 21s | 0.02s |
| Print statements | ~450 | ~10 |
| DEBUG spam | 50+ | 0 |
| Log lines/20s | 500+ | ~100 |
