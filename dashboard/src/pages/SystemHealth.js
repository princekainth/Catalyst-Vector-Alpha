import React, { useState, useEffect } from 'react';
import { Typography, Box, Card, CardContent, CircularProgress, Alert, Grid, Chip } from '@mui/material';
import { getSystemHealth } from '../api';
import { CheckCircle, Warning, Error, Memory, Computer as Cpu, Speed, Timeline } from '@mui/icons-material';

function SystemHealth() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        const data = await getSystemHealth();
        setHealthData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthData();

    const interval = setInterval(fetchHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Error loading system health: {error}
      </Alert>
    );
  }

  const getStatusChip = (status) => {
    let color, icon, label;
    switch (status) {
      case 'healthy':
        color = 'success';
        icon = <CheckCircle fontSize="small" />;
        label = 'Healthy';
        break;
      case 'warning':
        color = 'warning';
        icon = <Warning fontSize="small" />;
        label = 'Warning';
        break;
      case 'error':
        color = 'error';
        icon = <Error fontSize="small" />;
        label = 'Error';
        break;
      default:
        color = 'info';
        icon = <Timeline fontSize="small" />;
        label = 'Unknown';
    }
    return <Chip icon={icon} label={label} color={color} size="small" />;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Health Monitor
      </Typography>

      <Grid container spacing={3} sx={{ mt: 1 }}>
        {/* Overall Status */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Overall System Status
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Typography variant="subtitle1">Status:</Typography>
                {getStatusChip(healthData?.status)}
              </Box>

              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
                <StatusItem icon={<Cpu />} label="System Running" value={healthData?.running ? 'Yes' : 'No'} />
                <StatusItem icon={<Memory />} label="System Paused" value={healthData?.paused ? 'Yes' : 'No'} />
                <StatusItem icon={<Speed />} label="Current Cycle" value={healthData?.cycle || 'N/A'} />
                <StatusItem icon={<Timeline />} label="Agent Count" value={healthData?.agent_count || 0} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Tool Statistics */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tool Statistics
              </Typography>
              
              {healthData?.tool_stats ? (
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                  <StatItem label="Total Tools" value={healthData.tool_stats.total_tools || 0} />
                  <StatItem label="Successful Calls" value={healthData.tool_stats.successful_calls || 0} />
                  <StatItem label="Failed Calls" value={healthData.tool_stats.failed_calls || 0} />
                  <StatItem label="Average Latency" value={`${healthData.tool_stats.avg_latency || 0}ms`} />
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No tool statistics available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Task Statistics */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Task Statistics
              </Typography>
              
              {healthData?.task_stats ? (
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                  <StatItem label="Total Tasks" value={healthData.task_stats.total_tasks || 0} />
                  <StatItem label="Completed" value={healthData.task_stats.completed || 0} />
                  <StatItem label="Failed" value={healthData.task_stats.failed || 0} />
                  <StatItem label="Pending" value={healthData.task_stats.pending || 0} />
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No task statistics available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Agents List */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Agents
              </Typography>
              
              {healthData?.agents && healthData.agents.length > 0 ? (
                <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                  {healthData.agents.map((agent, index) => (
                    <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', 
                        backgroundColor: agent.paused ? 'warning.main' : 'success.main' }} />
                      <Typography variant="body2" noWrap>
                        {agent}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No active agents
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

function StatusItem({ icon, label, value }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
      {React.cloneElement(icon, { color: 'primary', fontSize: 'small' })}
      <Box>
        <Typography variant="caption">{label}</Typography>
        <Typography variant="body2">{value}</Typography>
      </Box>
    </Box>
  );
}

function StatItem({ label, value }) {
  return (
    <Box sx={{ py: 1 }}>
      <Typography variant="caption">{label}</Typography>
      <Typography variant="body2" fontWeight="medium">{value}</Typography>
    </Box>
  );
}

export default SystemHealth;