import React from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, ListItemAvatar, Avatar, Divider, Chip } from '@mui/material';
import { Timeline, CheckCircle, Error, AccessTime } from '@mui/icons-material';

function RecentTasks({ tasks }) {
  if (!tasks || tasks.length === 0) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Tasks
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No recent tasks available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const getStatusChip = (status) => {
    let color, label;
    switch (status) {
      case 'completed':
        color = 'success';
        label = 'Completed';
        break;
      case 'failed':
        color = 'error';
        label = 'Failed';
        break;
      case 'processing':
        color = 'info';
        label = 'Processing';
        break;
      default:
        color = 'default';
        label = status || 'Unknown';
    }
    return <Chip label={label} color={color} size="small" />;
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle color="success" />;
      case 'failed': return <Error color="error" />;
      case 'processing': return <AccessTime color="info" />;
      default: return <Timeline color="action" />;
    }
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Timeline color="primary" />
          Recent Tasks
        </Typography>

        <Typography variant="caption" color="text.secondary" gutterBottom>
          Last {Math.min(tasks.length, 10)} tasks
        </Typography>

        <Divider sx={{ my: 1 }} />

        <List sx={{ maxHeight: 300, overflow: 'auto' }}>
          {tasks.slice(0, 10).map((task, index) => (
            <React.Fragment key={task.task_id || index}>
              <ListItem alignItems="flex-start">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: getAvatarColor(task.status) }}>
                    {getStatusIcon(task.status)}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={task.summary || 'Untitled Task'}
                  secondary={
                    <React.Fragment>
                      <Typography component="span" variant="body2" color="text.primary">
                        {task.task_id}
                      </Typography>
                      {task.details && task.details.agent && (
                        <Typography component="span" variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                          Agent: {task.details.agent}
                        </Typography>
                      )}
                    </React.Fragment>
                  }
                />
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                  {getStatusChip(task.status)}
                  {task.timestamp && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                      {new Date(task.timestamp).toLocaleTimeString()}
                    </Typography>
                  )}
                </Box>
              </ListItem>
              {index < tasks.slice(0, 10).length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))}
        </List>
      </CardContent>
    </Card>
  );
}

function getAvatarColor(status) {
  switch (status) {
    case 'completed': return '#4caf50';
    case 'failed': return '#f44336';
    case 'processing': return '#2196f3';
    default: return '#9e9e9e';
  }
}

export default RecentTasks;