import React from 'react';
import { Card, CardContent, Typography, Box, Divider } from '@mui/material';
import { Memory, Computer as Cpu, Speed, Timeline, CheckCircle, Warning } from '@mui/icons-material';

function MetricsOverview({ metrics }) {
  if (!metrics) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Metrics Overview
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No metrics data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Timeline color="secondary" />
          Metrics Overview
        </Typography>

        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
          <MetricItem 
            icon={<Cpu color="primary" />}
            label="CPU Usage"
            value={`${metrics.cpu || 0}%`}
            trend={metrics.cpu > 70 ? 'warning' : 'normal'}
          />
          <MetricItem 
            icon={<Memory color="secondary" />}
            label="Memory Usage"
            value={`${metrics.memory || 0}%`}
            trend={metrics.memory > 80 ? 'warning' : 'normal'}
          />
          <MetricItem 
            icon={<Speed color="success" />}
            label="Planner Latency"
            value={`${metrics.planner_latency || 0}ms`}
            trend={metrics.planner_latency > 1000 ? 'warning' : 'normal'}
          />
          <MetricItem 
            icon={<Speed color="info" />}
            label="Worker Latency"
            value={`${metrics.worker_latency || 0}ms`}
            trend={metrics.worker_latency > 500 ? 'warning' : 'normal'}
          />
          <MetricItem 
            icon={<CheckCircle color="success" />}
            label="Breaker Trips"
            value={metrics.breaker_trips || 0}
            trend={metrics.breaker_trips > 5 ? 'warning' : 'normal'}
          />
          <MetricItem 
            icon={<Timeline color="primary" />}
            label="Uptime"
            value="Calculating..."
            trend="normal"
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        <Typography variant="caption" color="text.secondary">
          Last updated: {new Date().toLocaleTimeString()}
        </Typography>
      </CardContent>
    </Card>
  );
}

function MetricItem({ icon, label, value, trend }) {
  const getTrendIcon = () => {
    if (trend === 'warning') {
      return <Warning color="warning" fontSize="small" />;
    }
    return <CheckCircle color="success" fontSize="small" />;
  };

  return (
    <Box sx={{ py: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {icon}
        <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
          {label}
        </Typography>
        {getTrendIcon()}
      </Box>
      <Typography variant="body2" fontWeight="medium" sx={{ pl: 3 }}>
        {value}
      </Typography>
    </Box>
  );
}

export default MetricsOverview;