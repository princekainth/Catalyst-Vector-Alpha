import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress, Avatar, Divider, Chip } from '@mui/material';
import { People as AgentsIcon, CheckCircle, Warning, Memory, Speed } from '@mui/icons-material';

function AgentSummary({ agents }) {
  if (!agents || Object.keys(agents).length === 0) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Agent Summary
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No agents available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Calculate agent statistics
  const agentCount = Object.keys(agents).length;
  const activeAgents = Object.values(agents).filter(agent => !agent.paused).length;
  const pausedAgents = agentCount - activeAgents;
  const activePercentage = Math.round((activeAgents / agentCount) * 100);

  // Get agent roles
  const roles = {};
  Object.values(agents).forEach(agent => {
    const role = agent.role || 'Unknown';
    roles[role] = (roles[role] || 0) + 1;
  });

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AgentsIcon color="primary" />
          Agent Summary
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Agent Activity
          </Typography>
          <LinearProgress 
            variant="determinate"
            value={activePercentage}
            color={activePercentage > 70 ? 'success' : 'warning'}
            sx={{ height: 8, borderRadius: 4, mt: 0.5 }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="body2">
              {activeAgents} of {agentCount} active
            </Typography>
            <Typography variant="body2">
              {activePercentage}%
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }}>
          <StatusItem 
            icon={<CheckCircle color="success" />}
            label="Active Agents"
            value={activeAgents}
          />
          <StatusItem 
            icon={<Warning color="warning" />}
            label="Paused Agents"
            value={pausedAgents}
          />
          <StatusItem 
            icon={<Memory color="primary" />}
            label="Total Agents"
            value={agentCount}
          />
          <StatusItem 
            icon={<Speed color="secondary" />}
            label="Agent Roles"
            value={Object.keys(roles).length}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          Agent Roles Distribution
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
          {Object.entries(roles).map(([role, count]) => (
            <Chip 
              key={role}
              label={`${role}: ${count}`}
              size="small"
              avatar={<Avatar sx={{ bgcolor: getRoleColor(role), width: 24, height: 24 }}>{count}</Avatar>}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
}

function StatusItem({ icon, label, value }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {icon}
      <Box>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="body2">{value}</Typography>
      </Box>
    </Box>
  );
}

function getRoleColor(role) {
  const colors = [
    '#90caf9', '#f48fb1', '#a5d6a7', '#fff59d', '#b39ddb', '#80cbc4'
  ];
  const index = role.charCodeAt(0) % colors.length;
  return colors[index];
}

export default AgentSummary;