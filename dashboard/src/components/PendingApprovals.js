import React, { useState } from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, Button, Divider, Chip, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material';
import { CheckCircle, Close, Info, Gavel } from '@mui/icons-material';
import { approvePlan } from '../api';

function PendingApprovals({ plans, onRefresh, setSnackbarMessage, setSnackbarOpen }) {
  const [approveDialogOpen, setApproveDialogOpen] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState(null);
  const [approvalToken, setApprovalToken] = useState('');
  const [isApproving, setIsApproving] = useState(false);

  if (!plans || plans.length === 0) {
    return (
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Gavel color="primary" />
            Pending Approvals
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No pending approvals
          </Typography>
          <Chip label="All systems operational" color="success" size="small" sx={{ mt: 1 }} />
        </CardContent>
      </Card>
    );
  }

  const handleApproveClick = (plan) => {
    setSelectedPlan(plan);
    setApprovalToken(plan.approval_token || '');
    setApproveDialogOpen(true);
  };

  const handleApprove = async () => {
    if (!selectedPlan || isApproving) return;

    setIsApproving(true);
    try {
      const result = await approvePlan({
        task_id: selectedPlan.task_id,
        approval_token: approvalToken
      });

      if (result.ok) {
        setSnackbarMessage(`Approved: ${selectedPlan.action} - ${selectedPlan.deployment || selectedPlan.namespace}`);
        setSnackbarOpen(true);
        onRefresh();
      } else {
        setSnackbarMessage(`Approval failed: ${result.error || 'Unknown error'}`);
        setSnackbarOpen(true);
      }
    } catch (error) {
      setSnackbarMessage(`Approval error: ${error.message}`);
      setSnackbarOpen(true);
    } finally {
      setIsApproving(false);
      setApproveDialogOpen(false);
    }
  };

  const getActionIcon = (action) => {
    switch (action) {
      case 'k8s_scale': return '📊';
      case 'k8s_restart': return '🔄';
      default: return '⚙️';
    }
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Gavel color="warning" />
          Pending Approvals
        </Typography>

        <Typography variant="caption" color="text.secondary" gutterBottom>
          {plans.length} action{plans.length !== 1 ? 's' : ''} requiring approval
        </Typography>

        <Divider sx={{ my: 1 }} />

        <List sx={{ maxHeight: 300, overflow: 'auto' }}>
          {plans.map((plan, index) => (
            <React.Fragment key={plan.task_id || index}>
              <ListItem 
                secondaryAction={
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button 
                      size="small" 
                      variant="outlined" 
                      color="success" 
                      startIcon={<CheckCircle />}
                      onClick={() => handleApproveClick(plan)}
                    >
                      Approve
                    </Button>
                  </Box>
                }
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <span style={{ fontSize: '1.2em' }}>{getActionIcon(plan.action)}</span>
                      <Typography variant="body1" fontWeight="medium">
                        {plan.action.replace('_', ' ').toUpperCase()}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <React.Fragment>
                      {plan.deployment && (
                        <Typography component="span" variant="body2">
                          {plan.deployment}
                        </Typography>
                      )}
                      {plan.namespace && (
                        <Typography component="span" variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                          Namespace: {plan.namespace}
                        </Typography>
                      )}
                      {plan.replicas && (
                        <Typography component="span" variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                          Replicas: {plan.replicas}
                        </Typography>
                      )}
                      {plan.rationale && (
                        <Typography component="span" variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                          <Info fontSize="small" color="info" /> {plan.rationale}
                        </Typography>
                      )}
                    </React.Fragment>
                  }
                />
              </ListItem>
              {index < plans.length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))}
        </List>

        <ApproveDialog
          open={approveDialogOpen}
          onClose={() => setApproveDialogOpen(false)}
          plan={selectedPlan}
          approvalToken={approvalToken}
          setApprovalToken={setApprovalToken}
          onApprove={handleApprove}
          isApproving={isApproving}
        />
      </CardContent>
    </Card>
  );
}

function ApproveDialog({ open, onClose, plan, approvalToken, setApprovalToken, onApprove, isApproving }) {
  if (!plan) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Gavel color="warning" />
          Confirm Approval
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            {plan.action.replace('_', ' ').toUpperCase()}
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 2, my: 2 }}>
            {plan.deployment && (
              <React.Fragment>
                <Typography variant="body2" color="text.secondary">Deployment:</Typography>
                <Typography variant="body2">{plan.deployment}</Typography>
              </React.Fragment>
            )}
            {plan.namespace && (
              <React.Fragment>
                <Typography variant="body2" color="text.secondary">Namespace:</Typography>
                <Typography variant="body2">{plan.namespace}</Typography>
              </React.Fragment>
            )}
            {plan.replicas && (
              <React.Fragment>
                <Typography variant="body2" color="text.secondary">Replicas:</Typography>
                <Typography variant="body2">{plan.replicas}</Typography>
              </React.Fragment>
            )}
          </Box>

          {plan.rationale && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Rationale:</Typography>
              <Typography variant="body2" color="text.secondary">
                {plan.rationale}
              </Typography>
            </Box>
          )}

          <TextField
            label="Approval Token"
            value={approvalToken}
            onChange={(e) => setApprovalToken(e.target.value)}
            fullWidth
            margin="normal"
            variant="outlined"
            disabled={isApproving}
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isApproving} startIcon={<Close />}>Cancel</Button>
        <Button 
          onClick={onApprove} 
          disabled={isApproving || !approvalToken}
          startIcon={<CheckCircle />}
          variant="contained"
          color="success"
        >
          {isApproving ? 'Approving...' : 'Approve'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default PendingApprovals;