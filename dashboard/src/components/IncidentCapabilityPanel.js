import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box } from '@mui/material';
import { CheckCircle, Info } from '@mui/icons-material';

const IncidentCapabilityPanel = () => {
  const capabilities = [
    { type: 'CrashLoopBackOff', desc: 'Auto-detects loops & restarts only if transient.' },
    { type: 'ImagePullBackOff', desc: 'Identifies bad tags & proposes image patches.' },
    { type: 'OOMKilled', desc: 'Detects memory limit breaches & proposes scaling.' },
    { type: 'Failed Probes', desc: 'Diagnoses readiness/liveness failures.' },
    { type: 'Bad Rollouts', desc: 'Detects unhealthy rollouts & proposes rollbacks.' },
  ];

  return (
    <Card elevation={3} sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Info color="primary" />
          Intelligence Coverage
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
          CVA Cloud currently supports these incident classes:
        </Typography>
        <List dense>
          {capabilities.map((cap, i) => (
            <ListItem key={i} disableGutters>
              <ListItemIcon sx={{ minWidth: 32 }}>
                <CheckCircle color="success" fontSize="small" />
              </ListItemIcon>
              <ListItemText 
                primary={<Typography variant="body2" fontWeight="bold">{cap.type}</Typography>}
                secondary={<Typography variant="caption">{cap.desc}</Typography>}
              />
            </ListItem>
          ))}
        </List>
        <Box sx={{ mt: 2, p: 1, bgcolor: '#f5f5f5', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            <b>TODO:</b> Live incident aggregation endpoint needed after runtime incident store is implemented.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default IncidentCapabilityPanel;
