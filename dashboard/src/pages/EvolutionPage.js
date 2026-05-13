import React, { useState, useEffect, useCallback } from 'react';
import {
    Box, Typography, Card, CardContent, Grid, Chip, Button, Alert,
    CircularProgress, Divider, Dialog, DialogTitle, DialogContent,
    DialogActions, TextField, LinearProgress, List, ListItem, ListItemText,
    ListItemSecondaryAction, IconButton, Tooltip, Badge
} from '@mui/material';
import {
    AutoFixHigh, CheckCircle, WarningAmber, Schedule, Code,
    ThumbUp, InfoOutlined, Refresh, ReportProblem, Science
} from '@mui/icons-material';
import { getEvolutionStatus, approveEvolution, reportCapabilityGap } from '../api';

const STATUS_COLORS = {
    active: 'success', idle: 'default', quarantined: 'warning',
    promoted: 'success', failed: 'error', testing: 'info'
};

function EvolutionPage() {
    const [evoData, setEvoData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [approveDialog, setApproveDialog] = useState({ open: false, tool: null });
    const [gapDialog, setGapDialog] = useState(false);
    const [gapDesc, setGapDesc] = useState('');
    const [failedTask, setFailedTask] = useState('');
    const [actionMsg, setActionMsg] = useState('');

    const fetchData = useCallback(async () => {
        try {
            const data = await getEvolutionStatus();
            setEvoData(data);
            setError(null);
        } catch (e) {
            setError('Failed to load evolution status. Is the backend running?');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        const t = setInterval(fetchData, 15000);
        return () => clearInterval(t);
    }, [fetchData]);

    const handleApprove = async (toolName) => {
        try {
            await approveEvolution(toolName, 'Approved from dashboard');
            setActionMsg(`✅ Tool "${toolName}" approved and promoted to active!`);
            setApproveDialog({ open: false, tool: null });
            fetchData();
        } catch (e) {
            setActionMsg(`❌ Approval failed: ${e.message}`);
        }
    };

    const handleReportGap = async () => {
        try {
            await reportCapabilityGap(gapDesc, failedTask);
            setActionMsg('✅ Capability gap reported – Evolution Agent will research a new tool!');
            setGapDialog(false);
            setGapDesc(''); setFailedTask('');
            fetchData();
        } catch (e) {
            setActionMsg(`❌ Failed: ${e.message}`);
        }
    };

    if (loading) return <Box display="flex" justifyContent="center" pt={6}><CircularProgress size={60} /></Box>;

    const quarantined = evoData?.quarantined_tools || [];
    const evolved = evoData?.evolved_tools || [];
    const stats = evoData?.statistics || {};

    return (
        <Box>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Box display="flex" alignItems="center" gap={1}>
                    <AutoFixHigh color="primary" sx={{ fontSize: 32 }} />
                    <Typography variant="h4">Evolution Agent</Typography>
                </Box>
                <Box display="flex" gap={2}>
                    <Button variant="outlined" startIcon={<ReportProblem />} onClick={() => setGapDialog(true)}>
                        Report Capability Gap
                    </Button>
                    <Button variant="outlined" startIcon={<Refresh />} onClick={fetchData}>Refresh</Button>
                </Box>
            </Box>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {actionMsg && <Alert severity="info" sx={{ mb: 2 }} onClose={() => setActionMsg('')}>{actionMsg}</Alert>}

            {/* Stats Row */}
            <Grid container spacing={2} mb={3}>
                {[
                    { label: 'Tools Generated', value: stats.tools_generated ?? evolved.length, icon: <Science />, color: '#4fc3f7' },
                    { label: 'Quarantined', value: quarantined.length, icon: <WarningAmber />, color: '#ffb74d' },
                    { label: 'Promoted', value: stats.tools_promoted ?? 0, icon: <CheckCircle />, color: '#81c784' },
                    { label: 'Failed Tests', value: stats.tools_failed ?? 0, icon: <Schedule />, color: '#e57373' },
                ].map(({ label, value, icon, color }) => (
                    <Grid item xs={6} md={3} key={label}>
                        <Card sx={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', border: `1px solid ${color}22` }}>
                            <CardContent>
                                <Box display="flex" alignItems="center" gap={1} mb={1}>
                                    <Box color={color}>{icon}</Box>
                                    <Typography variant="h3" fontWeight={700} color={color}>{value}</Typography>
                                </Box>
                                <Typography variant="body2" color="text.secondary">{label}</Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            {/* Quarantined Tools – Awaiting Approval */}
            <Typography variant="h5" mb={2} display="flex" alignItems="center" gap={1}>
                <Badge badgeContent={quarantined.length} color="warning">
                    <WarningAmber color="warning" />
                </Badge>
                &nbsp;Quarantined Tools – Awaiting Approval
            </Typography>

            {quarantined.length === 0 ? (
                <Alert severity="success" sx={{ mb: 3 }}>No tools pending approval. All clear!</Alert>
            ) : (
                <Grid container spacing={2} mb={3}>
                    {quarantined.map((tool) => (
                        <Grid item xs={12} md={6} key={tool.name || tool}>
                            <Card sx={{ border: '1px solid #ffb74d44' }}>
                                <CardContent>
                                    <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                                        <Box>
                                            <Typography variant="h6" display="flex" alignItems="center" gap={1}>
                                                <Code fontSize="small" />
                                                {tool.name || tool}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 2 }}>
                                                {tool.description || 'Autonomously generated tool waiting for human review.'}
                                            </Typography>
                                            <Chip label={tool.status || 'quarantined'} color={STATUS_COLORS[tool.status] || 'warning'} size="small" />
                                            {tool.test_result && (
                                                <Chip label={tool.test_result} color="info" size="small" sx={{ ml: 1 }} />
                                            )}
                                        </Box>
                                        <Tooltip title="Promote to active tools">
                                            <Button
                                                variant="contained"
                                                color="success"
                                                size="small"
                                                startIcon={<ThumbUp />}
                                                onClick={() => setApproveDialog({ open: true, tool: tool.name || tool })}
                                            >
                                                Approve
                                            </Button>
                                        </Tooltip>
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}

            {/* Active Evolved Tools */}
            <Typography variant="h5" mb={2} display="flex" alignItems="center" gap={1}>
                <CheckCircle color="success" /> Active Evolved Tools
            </Typography>
            {evolved.length === 0 ? (
                <Alert severity="info" sx={{ mb: 3 }}>No evolved tools promoted yet.</Alert>
            ) : (
                <List sx={{ mb: 3 }}>
                    {evolved.map((tool) => (
                        <React.Fragment key={tool.name || tool}>
                            <ListItem>
                                <ListItemText
                                    primary={<Typography fontWeight={600}>{tool.name || tool}</Typography>}
                                    secondary={tool.description || 'Autonomously generated and promoted tool.'}
                                />
                                <ListItemSecondaryAction>
                                    <Chip label="ACTIVE" color="success" size="small" />
                                </ListItemSecondaryAction>
                            </ListItem>
                            <Divider />
                        </React.Fragment>
                    ))}
                </List>
            )}

            {/* Agent Status */}
            <Card sx={{ background: 'linear-gradient(135deg, #0d1117 0%, #1a1a2e 100%)' }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>Evolution Agent Status</Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="text.secondary">Status</Typography>
                            <Chip
                                label={evoData?.agent_status || evoData?.status || 'unknown'}
                                color={evoData?.agent_status === 'idle' ? 'default' : 'success'}
                                sx={{ mt: 0.5 }}
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="text.secondary">Last Activity</Typography>
                            <Typography variant="body2">{evoData?.last_activity || evoData?.last_evolution || 'N/A'}</Typography>
                        </Grid>
                        {evoData?.current_task && (
                            <Grid item xs={12}>
                                <Typography variant="body2" color="text.secondary">Current Task</Typography>
                                <Typography variant="body2">{evoData.current_task}</Typography>
                                <LinearProgress color="secondary" sx={{ mt: 1 }} />
                            </Grid>
                        )}
                    </Grid>
                </CardContent>
            </Card>

            {/* Approve Dialog */}
            <Dialog open={approveDialog.open} onClose={() => setApproveDialog({ open: false, tool: null })}>
                <DialogTitle>Approve Tool: {approveDialog.tool}</DialogTitle>
                <DialogContent>
                    <Typography>
                        This will promote <strong>{approveDialog.tool}</strong> from quarantine to the active tool library.
                        All agents will immediately have access to it.
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setApproveDialog({ open: false, tool: null })}>Cancel</Button>
                    <Button variant="contained" color="success" onClick={() => handleApprove(approveDialog.tool)}>
                        Approve & Promote
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Report Gap Dialog */}
            <Dialog open={gapDialog} onClose={() => setGapDialog(false)} maxWidth="sm" fullWidth>
                <DialogTitle display="flex" alignItems="center" gap={1}>
                    <ReportProblem color="warning" /> Report Capability Gap
                </DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="text.secondary" mb={2}>
                        Describe what the agents can't do. The Evolution Agent will autonomously generate a new tool.
                    </Typography>
                    <TextField
                        label="Gap Description"
                        fullWidth
                        multiline
                        rows={3}
                        value={gapDesc}
                        onChange={(e) => setGapDesc(e.target.value)}
                        placeholder="e.g. Agents cannot resolve DNS lookups or check if a domain is live"
                        sx={{ mb: 2 }}
                    />
                    <TextField
                        label="Related Failed Task (optional)"
                        fullWidth
                        value={failedTask}
                        onChange={(e) => setFailedTask(e.target.value)}
                        placeholder="e.g. Check if bill.spicetv.cc is responding on port 443"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setGapDialog(false)}>Cancel</Button>
                    <Button variant="contained" disabled={!gapDesc} onClick={handleReportGap}>
                        Submit Gap
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default EvolutionPage;
