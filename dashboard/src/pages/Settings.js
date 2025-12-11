import React, { useState } from 'react';
import { Typography, Box, Card, CardContent, Alert, Grid, TextField, Button, Switch, FormControlLabel, Divider, Snackbar, List, ListItem, ListItemText } from '@mui/material';
import { Settings as SettingsIcon, Save, Tune, Notifications, Group, Security } from '@mui/icons-material';

function Settings() {
  const [settings, setSettings] = useState({
    agentTimeout: 300,
    maxConcurrentTasks: 10,
    autoRetryFailedTasks: true,
    enableNotifications: false,
    notificationEmail: '',
    maxAgents: 20,
    debugMode: false,
    apiRateLimit: 100
  });

  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');

  const handleSettingChange = (name, value) => {
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = () => {
    // In a real app, this would save to the backend
    console.log('Saving settings:', settings);
    setSnackbarMessage('Settings saved successfully!');
    setSnackbarSeverity('success');
    setSnackbarOpen(true);
  };

  const handleReset = () => {
    // Reset to default values
    setSettings({
      agentTimeout: 300,
      maxConcurrentTasks: 10,
      autoRetryFailedTasks: true,
      enableNotifications: false,
      notificationEmail: '',
      maxAgents: 20,
      debugMode: false,
      apiRateLimit: 100
    });
    setSnackbarMessage('Settings reset to defaults');
    setSnackbarSeverity('info');
    setSnackbarOpen(true);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Settings
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure and manage your Catalyst Vector Alpha system
      </Alert>

      <Grid container spacing={3}>
        {/* Agent Configuration */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Tune color="primary" sx={{ mr: 1 }} />
                Agent Configuration
              </Typography>
              
              <TextField
                label="Agent Timeout (seconds)"
                type="number"
                value={settings.agentTimeout}
                onChange={(e) => handleSettingChange('agentTimeout', parseInt(e.target.value) || 0)}
                fullWidth
                margin="normal"
                helperText="Maximum time an agent can run before timeout"
              />
              
              <TextField
                label="Max Concurrent Tasks"
                type="number"
                value={settings.maxConcurrentTasks}
                onChange={(e) => handleSettingChange('maxConcurrentTasks', parseInt(e.target.value) || 0)}
                fullWidth
                margin="normal"
                helperText="Maximum number of tasks that can run simultaneously"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.autoRetryFailedTasks}
                    onChange={(e) => handleSettingChange('autoRetryFailedTasks', e.target.checked)}
                    color="primary"
                  />
                }
                label="Auto-retry failed tasks"
                sx={{ mt: 1 }}
              />
              
              <TextField
                label="Max Agents"
                type="number"
                value={settings.maxAgents}
                onChange={(e) => handleSettingChange('maxAgents', parseInt(e.target.value) || 0)}
                fullWidth
                margin="normal"
                helperText="Maximum number of agents that can be active"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Notifications color="primary" sx={{ mr: 1 }} />
                Notification Settings
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.enableNotifications}
                    onChange={(e) => handleSettingChange('enableNotifications', e.target.checked)}
                    color="primary"
                  />
                }
                label="Enable notifications"
              />
              
              {settings.enableNotifications && (
                <TextField
                  label="Notification Email"
                  type="email"
                  value={settings.notificationEmail}
                  onChange={(e) => handleSettingChange('notificationEmail', e.target.value)}
                  fullWidth
                  margin="normal"
                  helperText="Email address for system notifications"
                />
              )}
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Notification types:
              </Typography>
              <List dense>
                <ListItem disableGutters>
                  <ListItemText primary="Agent status changes" secondary="✓ Enabled" />
                </ListItem>
                <ListItem disableGutters>
                  <ListItemText primary="Task completion/failure" secondary="✓ Enabled" />
                </ListItem>
                <ListItem disableGutters>
                  <ListItemText primary="System health alerts" secondary="✓ Enabled" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* System Settings */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <SettingsIcon color="primary" sx={{ mr: 1 }} />
                System Settings
              </Typography>
              
              <TextField
                label="API Rate Limit (requests/min)"
                type="number"
                value={settings.apiRateLimit}
                onChange={(e) => handleSettingChange('apiRateLimit', parseInt(e.target.value) || 0)}
                fullWidth
                margin="normal"
                helperText="Maximum API requests per minute"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.debugMode}
                    onChange={(e) => handleSettingChange('debugMode', e.target.checked)}
                    color="primary"
                  />
                }
                label="Debug mode"
                sx={{ mt: 2 }}
              />
              
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                Enable debug mode for detailed logging and diagnostics
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* User Management */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom display="flex" alignItems="center">
                <Group color="primary" sx={{ mr: 1 }} />
                User Management
              </Typography>
              
              <Typography variant="body2" color="text.secondary" paragraph>
                User management is currently configured through the system administration interface.
              </Typography>
              
              <Button 
                variant="outlined"
                color="primary"
                startIcon={<Security />}
                disabled
              >
                Manage Users (Admin Only)
              </Button>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="body2" paragraph>
                Current session information:
              </Typography>
              
              <List dense>
                <ListItem disableGutters>
                  <ListItemText primary="User Role" secondary="Administrator" />
                </ListItem>
                <ListItem disableGutters>
                  <ListItemText primary="Session Duration" secondary="Active" />
                </ListItem>
                <ListItem disableGutters>
                  <ListItemText primary="Last Activity" secondary="Recently active" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Actions */}
        <Grid item xs={12}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Configuration Actions
              </Typography>
              
              <Box display="flex" gap={2}>
                <Button 
                  variant="contained"
                  color="primary"
                  startIcon={<Save />}
                  onClick={handleSave}
                  size="large"
                >
                  Save Settings
                </Button>
                
                <Button 
                  variant="outlined"
                  color="secondary"
                  onClick={handleReset}
                  size="large"
                >
                  Reset to Defaults
                </Button>
              </Box>
              
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 2 }}>
                Note: Some settings may require system restart to take effect
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSnackbarOpen(false)}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default Settings;