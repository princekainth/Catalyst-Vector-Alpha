import React, { useState, useEffect, useCallback } from 'react';
import {
    Box, Typography, Card, CardContent, Grid, Chip, Button, Alert,
    CircularProgress, Dialog, DialogTitle, DialogContent, DialogActions,
    TextField, List, ListItem, ListItemText, ListItemSecondaryAction,
    IconButton, Tooltip, Divider, LinearProgress, Slider, FormControlLabel,
    Switch
} from '@mui/material';
import {
    AddCircle, Stop, Memory, SmartToy, Refresh, Send, Build,
    AccessTime, Speed
} from '@mui/icons-material';
import { getFactoryStatus, spawnAgent, killAgent, assignAgentTask, getSemanticToolSuggestions } from '../api';

function AgentFactoryPage() {
    const [factoryData, setFactoryData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [msg, setMsg] = useState('');
    const [spawnDialog, setSpawnDialog] = useState(false);
    const [taskDialog, setTaskDialog] = useState({ open: false, agentId: null, agentName: '' });
    const [purpose, setPurpose] = useState('');
    const [ttlHours, setTtlHours] = useState(24);
    const [taskText, setTaskText] = useState('');
    const [semanticSuggestions, setSemanticSuggestions] = useState([]);
    const [suggestionLoading, setSuggestionLoading] = useState(false);
    const [autoKill, setAutoKill] = useState(false);

    const fetchData = useCallback(async () => {
        try {
            const data = await getFactoryStatus();
            setFactoryData(data);
            setError(null);
        } catch (e) {
            setError('Failed to load agent factory data.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        const t = setInterval(fetchData, 10000);
        return () => clearInterval(t);
    }, [fetchData]);

    const handleSpawn = async () => {
        try {
            const result = await spawnAgent(purpose, {}, ttlHours);
            if (result.success === false) {
                setMsg(`❌ ${result.error}\n💡 Try: ${result.suggestions?.[0] || 'Be more specific.'}`);
            } else {
                setMsg(`✅ Agent "${result.name || result.agent?.name}" spawned! ID: ${result.agent_id || result.agent?.agent_id}`);
            }
            setSpawnDialog(false);
            setPurpose('');
            fetchData();
        } catch (e) {
            setMsg(`❌ Spawn failed: ${e.message}`);
        }
    };

    const handleKill = async (agentId, agentName) => {
        try {
            await killAgent(agentId);
            setMsg(`✅ Agent "${agentName}" terminated.`);
            fetchData();
        } catch (e) {
            setMsg(`❌ Kill failed: ${e.message}`);
        }
    };

    const handleAssignTask = async () => {
        try {
            const result = await assignAgentTask(taskDialog.agentId, taskText);
            setMsg(`✅ Task assigned to "${taskDialog.agentName}": ${JSON.stringify(result).substring(0, 80)}...`);
            setTaskDialog({ open: false, agentId: null, agentName: '' });
            setTaskText('');
        } catch (e) {
            setMsg(`❌ Task assignment failed: ${e.message}`);
        }
    };

    const handleGetSuggestions = async () => {
        if (!purpose) return;
        setSuggestionLoading(true);
        try {
            const data = await getSemanticToolSuggestions(purpose);
            setSemanticSuggestions(data.matches || []);
        } catch (e) {
            setSemanticSuggestions([]);
        } finally {
            setSuggestionLoading(false);
        }
    };

    if (loading) return <Box display="flex" justifyContent="center" pt={6}><CircularProgress size={60} /></Box>;

    const activeAgents = factoryData?.active_agents || factoryData?.agents || [];

    return (
        <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Box display="flex" alignItems="center" gap={1}>
                    <SmartToy color="secondary" sx={{ fontSize: 32 }} />
                    <Typography variant="h4">Dynamic Agent Factory</Typography>
                </Box>
                <Box display="flex" gap={2}>
                    <Button
                        variant="contained"
                        color="secondary"
                        startIcon={<AddCircle />}
                        onClick={() => setSpawnDialog(true)}
                    >
                        Spawn Agent
                    </Button>
                    <Button variant="outlined" startIcon={<Refresh />} onClick={fetchData}>Refresh</Button>
                </Box>
            </Box>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {msg && (
                <Alert
                    severity={msg.startsWith('✅') ? 'success' : 'error'}
                    sx={{ mb: 2, whiteSpace: 'pre-line' }}
                    onClose={() => setMsg('')}
                >
                    {msg}
                </Alert>
            )}

            {/* Stats */}
            <Grid container spacing={2} mb={3}>
                <Grid item xs={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #7c3aed44' }}>
                        <CardContent>
                            <Typography variant="h3" fontWeight={700} color="secondary.main">{activeAgents.length}</Typography>
                            <Typography variant="body2" color="text.secondary">Active Agents</Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #0891b244' }}>
                        <CardContent>
                            <Typography variant="h3" fontWeight={700} color="info.main">
                                {factoryData?.total_spawned || activeAgents.length}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Total Spawned</Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #10b98144' }}>
                        <CardContent>
                            <Typography variant="h3" fontWeight={700} color="success.main">
                                {factoryData?.semantic_enabled ? 'ON' : 'OFF'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Semantic Matching</Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: '1px solid #f59e0b44' }}>
                        <CardContent>
                            <Typography variant="h3" fontWeight={700} color="warning.main">
                                {factoryData?.tool_embedding_count || 'N/A'}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">Tools Embedded</Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Active Agents List */}
            <Typography variant="h5" mb={2}>Active Dynamic Agents</Typography>

            {activeAgents.length === 0 ? (
                <Alert severity="info" sx={{ mb: 3 }}>
                    No dynamic agents running. Spawn one from a task or use the button above.
                </Alert>
            ) : (
                <Grid container spacing={2} mb={3}>
                    {activeAgents.map((agent) => {
                        const id = agent.agent_id || agent.id;
                        const name = agent.name;
                        const expires = agent.expires_at ? new Date(agent.expires_at).toLocaleTimeString() : 'N/A';
                        return (
                            <Grid item xs={12} md={6} key={id}>
                                <Card sx={{ border: '1px solid #7c3aed33' }}>
                                    <CardContent>
                                        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                                            <Box>
                                                <Typography variant="h6" display="flex" alignItems="center" gap={1}>
                                                    <Memory fontSize="small" color="secondary" /> {name}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">{id}</Typography>
                                            </Box>
                                            <Box display="flex" gap={1}>
                                                <Tooltip title="Assign a task">
                                                    <IconButton
                                                        size="small"
                                                        color="primary"
                                                        onClick={() => setTaskDialog({ open: true, agentId: id, agentName: name })}
                                                    >
                                                        <Send />
                                                    </IconButton>
                                                </Tooltip>
                                                <Tooltip title="Terminate agent">
                                                    <IconButton size="small" color="error" onClick={() => handleKill(id, name)}>
                                                        <Stop />
                                                    </IconButton>
                                                </Tooltip>
                                            </Box>
                                        </Box>

                                        <Typography variant="body2" color="text.secondary" mb={1}>
                                            {agent.purpose || 'General purpose agent'}
                                        </Typography>

                                        <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                                            <AccessTime fontSize="small" sx={{ color: 'text.secondary', fontSize: 14 }} />
                                            <Typography variant="caption" color="text.secondary">Expires: {expires}</Typography>
                                        </Box>

                                        <Divider sx={{ mb: 1.5 }} />

                                        <Typography variant="caption" color="text.secondary">Tools:</Typography>
                                        <Box display="flex" flexWrap="wrap" gap={0.5} mt={0.5}>
                                            {(agent.tools || []).length > 0 ? (
                                                agent.tools.map((t) => (
                                                    <Chip key={t} label={t} size="small" variant="outlined" sx={{ fontSize: '0.65rem' }} />
                                                ))
                                            ) : (
                                                <Typography variant="caption">None assigned</Typography>
                                            )}
                                        </Box>
                                    </CardContent>
                                </Card>
                            </Grid>
                        );
                    })}
                </Grid>
            )}

            {/* Spawn Dialog */}
            <Dialog open={spawnDialog} onClose={() => setSpawnDialog(false)} maxWidth="sm" fullWidth>
                <DialogTitle display="flex" alignItems="center" gap={1}>
                    <AddCircle color="secondary" /> Spawn New Agent
                </DialogTitle>
                <DialogContent>
                    <Alert severity="info" sx={{ mb: 2 }}>
                        Be specific. Use action verbs: "Scan", "Monitor", "Track", "Analyze", "Search".
                    </Alert>
                    <TextField
                        label="Purpose"
                        fullWidth
                        multiline
                        rows={2}
                        value={purpose}
                        onChange={(e) => setPurpose(e.target.value)}
                        placeholder="e.g. Scan bill.spicetv.cc for open ports and report results"
                        sx={{ mb: 2 }}
                    />
                    <Typography variant="body2" gutterBottom>TTL: {ttlHours} hours</Typography>
                    <Slider
                        value={ttlHours}
                        onChange={(_, v) => setTtlHours(v)}
                        min={1} max={168} marks
                        step={1}
                        valueLabelDisplay="auto"
                        sx={{ mb: 2 }}
                    />
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<Build />}
                        onClick={handleGetSuggestions}
                        disabled={!purpose || suggestionLoading}
                        sx={{ mb: 1 }}
                    >
                        Preview Tool Matches
                    </Button>
                    {suggestionLoading && <LinearProgress sx={{ mb: 1 }} />}
                    {semanticSuggestions.length > 0 && (
                        <Box mb={1}>
                            <Typography variant="caption" color="text.secondary">Semantic tool matches:</Typography>
                            <Box display="flex" flexWrap="wrap" gap={0.5} mt={0.5}>
                                {semanticSuggestions.map((m) => (
                                    <Chip
                                        key={m.tool}
                                        label={`${m.tool} (${(m.confidence * 100).toFixed(0)}%)`}
                                        size="small"
                                        color="secondary"
                                        variant="outlined"
                                    />
                                ))}
                            </Box>
                        </Box>
                    )}
                    <FormControlLabel
                        control={<Switch checked={autoKill} onChange={(e) => setAutoKill(e.target.checked)} />}
                        label="Auto-terminate when task completes"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => { setSpawnDialog(false); setSemanticSuggestions([]); }}>Cancel</Button>
                    <Button variant="contained" color="secondary" disabled={!purpose} onClick={handleSpawn}>
                        Spawn
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Task Dialog */}
            <Dialog open={taskDialog.open} onClose={() => setTaskDialog({ open: false, agentId: null, agentName: '' })} maxWidth="sm" fullWidth>
                <DialogTitle>Assign Task to {taskDialog.agentName}</DialogTitle>
                <DialogContent>
                    <TextField
                        label="Task"
                        fullWidth
                        multiline
                        rows={3}
                        value={taskText}
                        onChange={(e) => setTaskText(e.target.value)}
                        placeholder="e.g. Scan all ports on 192.168.1.1 and return open ports"
                        sx={{ mt: 1 }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setTaskDialog({ open: false, agentId: null, agentName: '' })}>Cancel</Button>
                    <Button variant="contained" disabled={!taskText} onClick={handleAssignTask}>Assign</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default AgentFactoryPage;
