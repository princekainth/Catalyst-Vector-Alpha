import React, { useState, useEffect } from 'react';
import { Typography, Box, Card, CardContent, Alert, CircularProgress, Grid, Chip, Divider } from '@mui/material';
import { CheckCircle, Warning, Error, Info } from '@mui/icons-material';
import api from '../api';

function AgentsMonitor() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        const response = await api.get('/agents');
        if (response.data && response.data.agents) {
          const agentsData = Object.entries(response.data.agents).map(([name, info]) => ({
            name,
            status: info.status || 'unknown',
            role: info.role || 'Agent',
            tools: info.state?.tools || [],
            lastActivity: info.last_activity || 'N/A',
            health: info.health || 'healthy'
          }));
          setAgents(agentsData);
        }
      } catch (err) {
        console.error('Failed to fetch agents:', err);
        setError('Failed to load agent data');
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
    const interval = setInterval(fetchAgents, 10000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'warning': return <Warning color="warning" />;
      case 'error': return <Error color="error" />;
      default: return <Info color="info" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'info';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Agents Monitor
        </Typography>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Agents Monitor
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Real-time status of {agents.length} agents in the swarm
      </Alert>

      <Grid container spacing={3}>
        {agents.length > 0 ? (
          agents.map((agent) => (
            <Grid item xs={12} md={6} lg={4} key={agent.name}>
              <Card elevation={3} sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6" gutterBottom>
                      {agent.name}
                    </Typography>
                    {getStatusIcon(agent.status)}
                  </Box>
                  
                  <Chip 
                    label={agent.status}
                    color={getStatusColor(agent.status)}
                    size="small"
                    sx={{ mb: 2 }}
                  />
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Role: {agent.role}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Last Activity: {agent.lastActivity}
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Available Tools:
                  </Typography>
                  
                  <Box display="flex" flexWrap="wrap" gap={1}>
                    {agent.tools.length > 0 ? (
                      agent.tools.map((tool) => (
                        <Chip key={tool} label={tool} size="small" variant="outlined" />
                      ))
                    ) : (
                      <Typography variant="caption">No tools configured</Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))
        ) : (
          <Grid item xs={12}>
            <Alert severity="info">No agents found in the system</Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default AgentsMonitor;