import React from 'react';
import { Typography, Box, Grid } from '@mui/material';
import AuditTimeline from '../components/AuditTimeline';

function Security() {
  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">Security & Audit Center</Typography>
        <Typography variant="subtitle2" color="text.secondary">Immutable trail of AI-driven remediation actions</Typography>
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <AuditTimeline />
        </Grid>
      </Grid>
    </Box>
  );
}

export default Security;