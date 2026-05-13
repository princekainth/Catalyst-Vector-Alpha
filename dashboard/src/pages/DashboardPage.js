import React, { useState, useEffect } from 'react';
import { Grid, Typography, Box, Button, CircularProgress, Alert, Snackbar } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import SystemStatusCard from '../components/SystemStatusCard';
import AgentSummary from '../components/AgentSummary';
import RecentTasks from '../components/RecentTasks';
import MetricsOverview from '../components/MetricsOverview';
import PendingApprovals from '../components/PendingApprovals';
import CommandInput from '../components/CommandInput';
import IncidentCapabilityPanel from '../components/IncidentCapabilityPanel';
import { getSystemHealth, getAgentsStatus, getMetrics, getTaskHistory, getPendingPlans, executeCommand } from '../api';

import IncidentFeed from '../components/IncidentFeed';

function DashboardPage() {
  // ... existing state ...
  const [systemHealth, setSystemHealth] = useState(null);
  const [agentsStatus, setAgentsStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [taskHistory, setTaskHistory] = useState([]);
  const [pendingPlans, setPendingPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [submittingCommand, setSubmittingCommand] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const fetchData = async () => {
    try {
      setRefreshing(true);
      setError(null);

      const [healthRes, agentsRes, metricsRes, tasksRes, plansRes] = await Promise.all([
        getSystemHealth(),
        getAgentsStatus(),
        getMetrics(),
        getTaskHistory(),
        getPendingPlans()
      ]);

      setSystemHealth(healthRes.data || healthRes);
      setAgentsStatus(agentsRes.agents || agentsRes.data || {});
      setMetrics(metricsRes.data || metricsRes);
      setTaskHistory(tasksRes.data || tasksRes.recent || []);
      setPendingPlans(plansRes.data || plansRes);

    } catch (err) {
      setError(err.message || 'Failed to fetch dashboard data');
      setSnackbarMessage('Error loading dashboard data');
      setSnackbarOpen(true);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();

    const interval = setInterval(() => {
      fetchData();
    }, 30000); // Refresh every 30s

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchData();
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  const handleExecuteCommand = async (commandText) => {
    setSubmittingCommand(true);
    try {
      const result = await executeCommand(commandText);
      setSnackbarMessage(`Command submitted successfully (Task ID: ${result.task_id.substring(0, 8)}...)`);
      setSnackbarOpen(true);
      setTimeout(fetchData, 1000);
    } catch (err) {
      setSnackbarMessage(`Error submitting command: ${err.message}`);
      setSnackbarOpen(true);
    } finally {
      setSubmittingCommand(false);
    }
  };

  if (loading && !systemHealth) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold">CVA Cloud Dashboard</Typography>
          <Typography variant="subtitle2" color="text.secondary">AI SRE Agent with Approval-Gated Remediation</Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={refreshing}
          sx={{ borderRadius: 2 }}
        >
          {refreshing ? 'Syncing...' : 'Sync Now'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* System Status */}
        <Grid item xs={12} md={6} lg={4}>
          <SystemStatusCard
            health={systemHealth}
            agents={agentsStatus}
            metrics={metrics}
          />
        </Grid>

        {/* Live Incident Feed */}
        <Grid item xs={12} md={6} lg={4}>
          <IncidentFeed />
        </Grid>

        {/* Metrics Overview */}
        <Grid item xs={12} md={12} lg={4}>
          <MetricsOverview metrics={metrics} />
        </Grid>

        {/* Command Center */}
        <Grid item xs={12}>
           <CommandInput
            onExecuteCommand={handleExecuteCommand}
            isSubmitting={submittingCommand}
          />
        </Grid>

        {/* Remediation Gate */}
        <Grid item xs={12} md={6}>
          <PendingApprovals
            plans={pendingPlans}
            onRefresh={fetchData}
            setSnackbarMessage={setSnackbarMessage}
            setSnackbarOpen={setSnackbarOpen}
          />
        </Grid>

        {/* Recent Tasks */}
        <Grid item xs={12} md={6}>
          <RecentTasks tasks={taskHistory} />
        </Grid>
        
        {/* Intelligence Coverage / Supported Incidents */}
        <Grid item xs={12} md={6}>
          <IncidentCapabilityPanel />
        </Grid>

        {/* Agent Inventory */}
        <Grid item xs={12} md={6}>
          <AgentSummary agents={agentsStatus} />
        </Grid>
      </Grid>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        message={snackbarMessage}
      />
    </Box>
  );
}

export default DashboardPage;