import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress, Divider } from '@mui/material';
import { CheckCircle, Warning, Error, Info, Memory, Computer as Cpu, People as AgentsIcon } from '@mui/icons-material';

function SystemStatusCard({ health, agents, metrics }) {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'info';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'warning': return <Warning color="warning" />;
      case 'error': return <Error color="error" />;
      default: return <Info color="info" />;
    }
  };

  // Calculate overall health score (0-100)
  const calculateHealthScore = () => {
    if (!health) return 50;
    
    let score = 70; // Base score
    
    if (health.status === 'healthy') score += 15;
    if (health.running) score += 10;
    if (!health.paused) score += 5;
    
    // Adjust based on agent health
    const agentCount = agents ? Object.keys(agents).length : 0;
    if (agentCount > 0) score += 5;
    
    return Math.min(100, score);
  };

  const healthScore = calculateHealthScore();

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {getStatusIcon(health?.status || 'healthy')}
          System Status
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Overall Health Score
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={healthScore}
            color={healthScore > 70 ? 'success' : healthScore > 40 ? 'warning' : 'error'}
            sx={{ height: 8, borderRadius: 4, mt: 0.5 }}
          />
          <Typography variant="body2" sx={{ textAlign: 'right', mt: 0.5 }}>
            {healthScore}/100
          </Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }}>
          <StatusItem 
            icon={<Cpu />}
            label="System Status"
            value={health?.status || 'Unknown'}
            color={getStatusColor(health?.status)}
          />
          <StatusItem 
            icon={<Memory />}
            label="Agents Active"
            value={agents ? Object.keys(agents).length : 0}
            color="primary"
          />
          <StatusItem 
            icon={<AgentsIcon />}
            label="Running"
            value={health?.running ? 'Yes' : 'No'}
            color={health?.running ? 'success' : 'error'}
          />
          <StatusItem 
            icon={health?.paused ? <Warning /> : <CheckCircle />}
            label="Paused"
            value={health?.paused ? 'Yes' : 'No'}
            color={health?.paused ? 'warning' : 'success'}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          System Metrics
        </Typography>

        {metrics ? (
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
            <MetricItem label="CPU Usage" value={`${metrics.cpu || 0}%`} />
            <MetricItem label="Memory Usage" value={`${metrics.memory || 0}%`} />
            <MetricItem label="Planner Latency" value={`${metrics.planner_latency || 0}ms`} />
            <MetricItem label="Worker Latency" value={`${metrics.worker_latency || 0}ms`} />
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Metrics loading...
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

function StatusItem({ icon, label, value, color }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {React.cloneElement(icon, { color: color, fontSize: 'small' })}
      <Box>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="body2">{value}</Typography>
      </Box>
    </Box>
  );
}

function MetricItem({ label, value }) {
  return (
    <Box>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="body2" fontWeight="medium">{value}</Typography>
    </Box>
  );
}

export default SystemStatusCard;