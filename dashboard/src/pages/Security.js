import React, { useState, useEffect } from 'react';
import { Typography, Box, Card, CardContent, Alert, CircularProgress, Grid, List, ListItem, ListItemText, Chip, Divider } from '@mui/material';
import { Shield, Lock, Warning, CheckCircle, Error } from '@mui/icons-material';
import api from '../api';

function Security() {
  const [securityData, setSecurityData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSecurity = async () => {
      try {
        setLoading(true);
        // Try to fetch security data from API
        const response = await api.get('/security/status');
        if (response.data && response.data.status === 'ok') {
          setSecurityData(response.data.data);
        }
      } catch (err) {
        console.error('Failed to fetch security data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchSecurity();
  }, []);

  // Generate sample security data
  const generateSampleData = () => ({
    threats: [
      { id: 1, type: 'Suspicious Activity', severity: 'medium', status: 'investigating', timestamp: '2023-12-10T10:30:00' },
      { id: 2, type: 'Unauthorized Access Attempt', severity: 'high', status: 'resolved', timestamp: '2023-12-09T14:15:00' }
    ],
    accessLogs: [
      { user: 'system', action: 'API Access', timestamp: '2023-12-10T11:00:00', status: 'success' },
      { user: 'admin', action: 'Dashboard Login', timestamp: '2023-12-10T10:45:00', status: 'success' }
    ],
    compliance: {
      status: 'compliant',
      checksPassed: 18,
      checksFailed: 2,
      lastScan: '2023-12-10T09:30:00'
    },
    systemStatus: 'secure'
  });

  const data = securityData || generateSampleData();

  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return <CheckCircle color="success" fontSize="small" />;
      case 'failed': return <Error color="error" fontSize="small" />;
      case 'investigating': return <Warning color="warning" fontSize="small" />;
      default: return <Security color="primary" fontSize="small" />;
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Security Center
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        {securityData ? 'Active security monitoring' : 'Security monitoring (sample data shown)'}
      </Alert>

      <Grid container spacing={3}>
        {/* System Security Status */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Shield color="primary" sx={{ mr: 1 }} />
                System Security Status
              </Typography>
              
              <Box display="flex" alignItems="center" mb={2}>
                <Typography variant="h4" mr={2}>
                  {data.systemStatus === 'secure' ? 
                    <CheckCircle color="success" fontSize="large" /> :
                    <Warning color="warning" fontSize="large" />}
                </Typography>
                <Typography variant="h5">
                  {data.systemStatus.toUpperCase()}
                </Typography>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle1" gutterBottom>
                Compliance Status
              </Typography>
              
              <Box display="flex" alignItems="center" mb={1}>
                <Typography variant="body2" mr={1}>Overall:</Typography>
                <Chip 
                  label={data.compliance.status}
                  color={data.compliance.status === 'compliant' ? 'success' : 'error'}
                  size="small"
                />
              </Box>
              
              <Typography variant="body2">
                {data.compliance.checksPassed} passed, {data.compliance.checksFailed} failed
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Last scan: {new Date(data.compliance.lastScan).toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Threats */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Warning color="warning" sx={{ mr: 1 }} />
                Recent Security Events
              </Typography>
              
              {data.threats && data.threats.length > 0 ? (
                <List dense>
                  {data.threats.map((threat) => (
                    <ListItem key={threat.id} disableGutters>
                      <ListItemText
                        primary={threat.type}
                        secondary={new Date(threat.timestamp).toLocaleString()}
                      />
                      <Box display="flex" alignItems="center" ml={2}>
                        <Chip 
                          label={threat.severity}
                          color={getSeverityColor(threat.severity)}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Chip 
                          label={threat.status}
                          color={threat.status === 'resolved' ? 'success' : 'warning'}
                          size="small"
                        />
                      </Box>
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No recent security events
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Access Logs */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Lock color="primary" sx={{ mr: 1 }} />
                Recent Access Activity
              </Typography>
              
              {data.accessLogs && data.accessLogs.length > 0 ? (
                <List dense>
                  {data.accessLogs.map((log, index) => (
                    <ListItem key={index} disableGutters>
                      <ListItemText
                        primary={log.action}
                        secondary={
                          <>
                            <Typography component="span" variant="body2" color="text.primary">
                              {log.user}
                            </Typography>
                            {` - ${new Date(log.timestamp).toLocaleString()}`}
                          </>
                        }
                      />
                      {getStatusIcon(log.status)}
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No recent access activity
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Security Recommendations */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Security color="primary" sx={{ mr: 1 }} />
                Security Recommendations
              </Typography>
              
              <Alert severity="info" sx={{ mb: 2 }}>
                Your system appears to be secure. Consider these best practices:
              </Alert>
              
              <List dense>
                <ListItem>
                  <ListItemText primary="Enable two-factor authentication for all users" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Regularly review access logs and permissions" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Keep all system components updated" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Monitor for unusual activity patterns" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Security;