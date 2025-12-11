import React, { useState, useEffect } from 'react';
import { Grid, Typography, Box, Button, CircularProgress, Alert, Snackbar } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import SystemStatusCard from '../components/SystemStatusCard';
import AgentSummary from '../components/AgentSummary';
import RecentTasks from '../components/RecentTasks';
import MetricsOverview from '../components/MetricsOverview';
import PendingApprovals from '../components/PendingApprovals';
import { getSystemHealth, getAgentsStatus, getMetrics, getTaskHistory, getPendingPlans } from '../api';

function DashboardPage() {
  const [systemHealth, setSystemHealth] = useState(null);
  const [agentsStatus, setAgentsStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [taskHistory, setTaskHistory] = useState([]);
  const [pendingPlans, setPendingPlans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const fetchData = async () => {
    try {
      setLoading(true);
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
    }, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
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
        <Typography variant="h4" gutterBottom>
          Catalyst Vector Alpha Dashboard
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
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

        {/* Agent Summary */}
        <Grid item xs={12} md={6} lg={4}>
          <AgentSummary agents={agentsStatus} />
        </Grid>

        {/* Metrics Overview */}
        <Grid item xs={12} md={12} lg={4}>
          <MetricsOverview metrics={metrics} />
        </Grid>

        {/* Recent Tasks */}
        <Grid item xs={12} md={6}>
          <RecentTasks tasks={taskHistory} />
        </Grid>

        {/* Pending Approvals */}
        <Grid item xs={12} md={6}>
          <PendingApprovals 
            plans={pendingPlans} 
            onRefresh={fetchData}
            setSnackbarMessage={setSnackbarMessage}
            setSnackbarOpen={setSnackbarOpen}
          />
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