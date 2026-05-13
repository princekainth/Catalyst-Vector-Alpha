import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('cva_token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) window.location.href = '/login';
    return Promise.reject(error);
  }
);

// ─── Health & System ───────────────────────────────
export const getSystemHealth = () => api.get('/health/detailed').then(r => r.data);
export const getMetrics = () => api.get('/metrics/stats').then(r => r.data);
export const getMetricsTrends = () => api.get('/metrics/trends').then(r => r.data);
export const getSystemMetrics = () => api.get('/system_metrics').then(r => r.data);
export const getDashboardSummary = () => api.get('/dashboard_summary').then(r => r.data);
export const getDiagnostics = () => api.get('/diagnostics').then(r => r.data);
export const getToolBreakers = () => api.get('/tool_breakers').then(r => r.data);
export const pauseSystem = () => api.post('/pause').then(r => r.data);
export const unpauseSystem = () => api.post('/unpause').then(r => r.data);
export const runSelfTest = () => api.post('/self_test').then(r => r.data);

// ─── Agents ────────────────────────────────────────
export const getAgentsStatus = () => api.get('/agents').then(r => r.data);
export const getAgent = (name) => api.get(`/agents/${name}`).then(r => r.data);
export const restartAgent = (name) => api.post(`/agents/${name}/restart`).then(r => r.data);
export const getFactoryStatus = () => api.get('/agents/factory').then(r => r.data);
export const spawnAgent = (purpose, context = {}, ttl_hours = 24) =>
  api.post('/agents/spawn', { purpose, context, ttl_hours }).then(r => r.data);
export const killAgent = (agent_id) => api.post(`/agents/kill/${agent_id}`).then(r => r.data);
export const assignAgentTask = (agent_id, task) =>
  api.post('/agents/task', { agent_id, task }).then(r => r.data);
export const getSemanticToolSuggestions = (purpose) =>
  api.post('/agents/semantic-tools', { purpose }).then(r => r.data);

// ─── Tasks ─────────────────────────────────────────
export const getTaskHistory = () => api.get('/task_history').then(r => r.data);
export const getTaskStatus = (task_id) => api.get(`/catalyst/task_status/${task_id}`).then(r => r.data);
export const executeCommand = (command) => api.post('/command', { command }).then(r => r.data);
export const getTasks = () => api.get('/tasks').then(r => r.data);
export const getAuditLogs = () => api.get('/audit/logs').then(r => r.data);

// ─── Evolution ─────────────────────────────────────
export const getEvolutionStatus = () => api.get('/evolution/status').then(r => r.data);
export const approveEvolution = (tool_name, notes = '') =>
  api.post('/evolution/approve', { tool_name, notes }).then(r => r.data);
export const reportCapabilityGap = (gap_description, failed_task) =>
  api.post('/evolution/report-gap', { gap_description, failed_task }).then(r => r.data);

// ─── Tools ─────────────────────────────────────────
export const executeTool = (tool_name, tool_args = {}, agent_id = 'dashboard') =>
  api.post('/tools/execute', { tool_name, tool_args, agent_id }).then(r => r.data);

// ─── Plans & Approvals ─────────────────────────────
export const getPendingPlans = () => api.get('/catalyst/plans').then(r => r.data);
export const approvePlan = (planData) => api.post('/approve', planData).then(r => r.data);
export const getPending = () => api.get('/pending').then(r => r.data);

// ─── Organism / Prometheus ─────────────────────────
export const getOrganismStatus = () => api.get('/organism/status').then(r => r.data);
export const promQuery = (query, time) =>
  api.get('/prom/query', { params: { query, time } }).then(r => r.data);

// ─── Incidents ─────────────────────────────────────
export const getIncidents = () => api.get('/incidents').then(r => r.data);
export const getIncident = (id) => api.get(`/incidents/${id}`).then(r => r.data);
export const resolveIncident = (id, reason = 'manual') => 
  api.post(`/incidents/${id}/resolve`, { reason }).then(r => r.data);

// ─── Misc ──────────────────────────────────────────
export const getLatestInsight = () => api.get('/latest_insight').then(r => r.data);

export default api;