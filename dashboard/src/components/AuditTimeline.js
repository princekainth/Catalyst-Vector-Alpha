import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, Chip, CircularProgress, Divider } from '@mui/material';
import { History, Lock, CheckCircle, Warning, Gavel } from '@mui/icons-material';
import { getAuditLogs } from '../api';

const AuditTimeline = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await getAuditLogs();
        if (res.ok) setLogs(res.logs || []);
      } catch (err) {
        console.error("Audit log fetch failed:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchLogs();
    const int = setInterval(fetchLogs, 5000);
    return () => clearInterval(int);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'approval_required': return <Gavel fontSize="small" color="warning" />;
      case 'allow': return <CheckCircle fontSize="small" color="success" />;
      case 'start': return <History fontSize="small" color="info" />;
      case 'ok': return <CheckCircle fontSize="small" color="success" />;
      case 'error': return <Warning fontSize="small" color="error" />;
      default: return <History fontSize="small" />;
    }
  };

  if (loading) return <CircularProgress size={20} />;

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <History color="primary" />
          Remediation Audit Timeline
        </Typography>
        <List dense sx={{ maxHeight: 400, overflow: 'auto' }}>
          {logs.length === 0 ? (
            <Typography variant="body2" color="text.secondary">No remediation actions recorded.</Typography>
          ) : (
            logs.reverse().map((log, i) => (
              <React.Fragment key={i}>
                <ListItem alignItems="flex-start" sx={{ px: 0 }}>
                  <Box sx={{ mr: 2, mt: 0.5 }}>{getStatusIcon(log.status)}</Box>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" fontWeight="bold">
                          {log.tool || 'System Action'}
                        </Typography>
                        <Chip label={log.status.toUpperCase()} size="small" variant="outlined" sx={{ height: 18, fontSize: '0.65rem' }} />
                      </Box>
                    }
                    secondary={
                      <Box component="span">
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                          {new Date(log.timestamp).toLocaleTimeString()} | Trace: {log.trace_id?.substring(0, 8)}...
                        </Typography>
                        {log.args && (
                          <Box sx={{ mt: 0.5, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Lock fontSize="inherit" color="action" />
                            <Typography variant="caption" color="text.secondary">Args Redacted (Security Gated)</Typography>
                          </Box>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
                {i < logs.length - 1 && <Divider component="li" />}
              </React.Fragment>
            ))
          )}
        </List>
      </CardContent>
    </Card>
  );
};

export default AuditTimeline;
