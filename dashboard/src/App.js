import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import {
  Box, CssBaseline, Drawer, AppBar, Toolbar, List, ListItem,
  ListItemIcon, ListItemText, Typography, Divider, IconButton, Chip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  MonitorHeart as MonitorHeartIcon,
  People as AgentsIcon,
  Settings as SettingsIcon,
  Menu as MenuIcon,
  BarChart as BarChartIcon,
  Security as SecurityIcon,
  AutoFixHigh as EvoIcon,
  SmartToy as FactoryIcon,
} from '@mui/icons-material';

import DashboardPage from './pages/DashboardPage';
import SystemHealth from './pages/SystemHealth';
import AgentsMonitor from './pages/AgentsMonitor';
import Metrics from './pages/Metrics';
import Security from './pages/Security';
import Settings from './pages/Settings';
import EvolutionPage from './pages/EvolutionPage';
import AgentFactoryPage from './pages/AgentFactoryPage';
import api from './api';

const drawerWidth = 250;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [swarmStatus, setSwarmStatus] = useState('unknown');

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await api.get('/health/detailed');
        const d = response?.data;
        const connected = d?.status === 'ok' || d?.status === 'online';
        setIsConnected(connected);
        setSwarmStatus(d?.runtime_state || d?.status || 'unknown');
      } catch {
        setIsConnected(false);
        setSwarmStatus('offline');
      }
    };
    checkConnection();
    const iv = setInterval(checkConnection, 30000);
    return () => clearInterval(iv);
  }, []);

  const menuItems = [
    { text: 'Overview', icon: <DashboardIcon />, path: '#/' },
    { text: 'System Health', icon: <MonitorHeartIcon />, path: '#/health' },
    { text: 'Agents', icon: <AgentsIcon />, path: '#/agents' },
    { text: 'Agent Factory', icon: <FactoryIcon />, path: '#/factory', badge: 'NEW' },
    { text: 'Evolution', icon: <EvoIcon />, path: '#/evolution', badge: 'NEW' },
    { text: 'Metrics', icon: <BarChartIcon />, path: '#/metrics' },
    { text: 'Security', icon: <SecurityIcon />, path: '#/security' },
    { text: 'Settings', icon: <SettingsIcon />, path: '#/settings' },
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          background: 'linear-gradient(90deg, #0f0c29, #302b63, #24243e)',
          borderBottom: '1px solid rgba(124, 58, 237, 0.3)',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={() => setMobileOpen(!mobileOpen)}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: 1 }}>
            ⚡ Catalyst Vector Alpha
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <Chip
              size="small"
              label={swarmStatus.toUpperCase()}
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
              sx={{ fontWeight: 700, fontSize: '0.7rem' }}
            />
            <Box sx={{
              width: 10, height: 10, borderRadius: '50%',
              backgroundColor: isConnected ? '#4ade80' : '#f87171',
              boxShadow: isConnected
                ? '0 0 8px #4ade80, 0 0 16px rgba(74, 222, 128, 0.4)'
                : '0 0 8px #f87171',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }} />
          </Box>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}>
        {[{ variant: 'temporary', open: mobileOpen, onClose: () => setMobileOpen(false), display: { xs: 'block', sm: 'none' } },
        { variant: 'permanent', open: true, display: { xs: 'none', sm: 'block' } }
        ].map((props, i) => (
          <Drawer
            key={i}
            variant={props.variant}
            open={props.open}
            onClose={props.onClose}
            ModalProps={{ keepMounted: true }}
            sx={{
              display: props.display,
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box', width: drawerWidth,
                background: 'linear-gradient(180deg, #0d1117 0%, #0f0c29 100%)',
                borderRight: '1px solid rgba(124, 58, 237, 0.2)',
              }
            }}
          >
            <Toolbar />
            <Box sx={{ px: 2, py: 1.5 }}>
              <Typography variant="caption" color="rgba(255,255,255,0.3)" fontWeight={600} letterSpacing={2}>
                NAVIGATION
              </Typography>
            </Box>
            <Divider sx={{ borderColor: 'rgba(124,58,237,0.2)' }} />
            <List>
              {menuItems.map((item) => (
                <ListItem
                  button
                  key={item.text}
                  component="a"
                  href={item.path}
                  sx={{
                    py: 1.2, px: 2,
                    '&:hover': { background: 'rgba(124, 58, 237, 0.15)', borderRadius: 1 },
                    '& .MuiListItemIcon-root': { minWidth: 40, color: 'rgba(255,255,255,0.6)' },
                    '& .MuiListItemText-primary': { fontSize: '0.9rem', color: 'rgba(255,255,255,0.85)' },
                  }}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                  {item.badge && (
                    <Chip label={item.badge} size="small" color="secondary" sx={{ fontSize: '0.6rem', height: 18 }} />
                  )}
                </ListItem>
              ))}
            </List>
            <Divider sx={{ borderColor: 'rgba(124,58,237,0.2)', mt: 'auto' }} />
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="caption" color="rgba(255,255,255,0.2)">
                CVA v3.0 • {new Date().getFullYear()}
              </Typography>
            </Box>
          </Drawer>
        ))}
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1, p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          background: '#070714',
          minHeight: '100vh',
        }}
      >
        <Routes>
          <Route path="" element={<DashboardPage />} />
          <Route path="health" element={<SystemHealth />} />
          <Route path="agents" element={<AgentsMonitor />} />
          <Route path="factory" element={<AgentFactoryPage />} />
          <Route path="evolution" element={<EvolutionPage />} />
          <Route path="metrics" element={<Metrics />} />
          <Route path="security" element={<Security />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="" replace />} />
        </Routes>
      </Box>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </Box>
  );
}

export default App;
