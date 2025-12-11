import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CssBaseline, Drawer, AppBar, Toolbar, List, ListItem, ListItemIcon, ListItemText, Typography, Divider, IconButton } from '@mui/material';
import { Dashboard as DashboardIcon, MonitorHeart as MonitorHeartIcon, People as AgentsIcon, Settings as SettingsIcon, Menu as MenuIcon, BarChart as BarChartIcon, Security as SecurityIcon } from '@mui/icons-material';
import DashboardPage from './pages/DashboardPage';
import SystemHealth from './pages/SystemHealth';
import AgentsMonitor from './pages/AgentsMonitor';
import Metrics from './pages/Metrics';
import Security from './pages/Security';
import Settings from './pages/Settings';
import api from './api';

const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Check system connection status
    const checkConnection = async () => {
      try {
        const response = await api.get('/health');
        if (response.data.status === 'ok') {
          setIsConnected(true);
        }
      } catch (error) {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '#/' },
    { text: 'System Health', icon: <MonitorHeartIcon />, path: '#/health' },
    { text: 'Agents', icon: <AgentsIcon />, path: '#/agents' },
    { text: 'Metrics', icon: <BarChartIcon />, path: '#/metrics' },
    { text: 'Security', icon: <SecurityIcon />, path: '#/security' },
    { text: 'Settings', icon: <SettingsIcon />, path: '#/settings' },
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Catalyst Vector Alpha
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ 
              width: 12, 
              height: 12, 
              borderRadius: '50%',
              backgroundColor: isConnected ? 'success.main' : 'error.main',
              boxShadow: isConnected ? '0 0 8px rgba(0, 255, 0, 0.5)' : '0 0 8px rgba(255, 0, 0, 0.5)'
            }} />
            <Typography variant="caption">
              {isConnected ? 'Connected' : 'Disconnected'}
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{ 
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          <Sidebar menuItems={menuItems} />
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{ 
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          <Sidebar menuItems={menuItems} />
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{ 
          flexGrow: 1, 
          p: 3, 
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8
        }}
      >
        <Routes>
          <Route path="" element={<DashboardPage />} />
          <Route path="health" element={<SystemHealth />} />
          <Route path="agents" element={<AgentsMonitor />} />
          <Route path="metrics" element={<Metrics />} />
          <Route path="security" element={<Security />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="" replace />} />
        </Routes>
      </Box>
    </Box>
  );
}

function Sidebar({ menuItems }) {
  return (
    <div>
      <Toolbar />
      <Divider />
      <List>
        {menuItems.map((item, index) => (
          <ListItem button key={item.text} component="a" href={item.path} sx={{ py: 1.5 }}>
            <ListItemIcon>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </div>
  );
}

export default App;