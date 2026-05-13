import React, { useState } from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, Button, Divider, Chip, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material';
import { CheckCircle, Close, Info, Gavel } from '@mui/icons-material';
import { approvePlan } from '../api';
import RiskBadge from './RiskBadge';
import EvidencePreview from './EvidencePreview';

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
        setSnackbarMessage(`Approved: ${selectedPlan.action}`);
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
    if (action.includes('patch')) return '🔐';
    if (action.includes('rollout')) return '🔄';
    if (action.includes('scale')) return '📊';
    return '⚙️';
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Gavel color="warning" />
          Remediation Gate
        </Typography>

        <Typography variant="caption" color="text.secondary" gutterBottom>
          {plans.length} action{plans.length !== 1 ? 's' : ''} blocked by safety policy
        </Typography>

        <Divider sx={{ my: 1 }} />

        <List sx={{ maxHeight: 400, overflow: 'auto' }}>
          {plans.map((plan, index) => (
            <React.Fragment key={plan.task_id || index}>
              <ListItem 
                alignItems="flex-start"
                secondaryAction={
                  <Button 
                    size="small" 
                    variant="contained" 
                    color="primary" 
                    startIcon={<CheckCircle />}
                    onClick={() => handleApproveClick(plan)}
                  >
                    Review
                  </Button>
                }
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <span style={{ fontSize: '1.1em' }}>{getActionIcon(plan.action)}</span>
                      <Typography variant="body2" fontWeight="bold">
                        {plan.action.replace('k8s_', '').toUpperCase()}
                      </Typography>
                      <RiskBadge risk={plan.action.includes('patch') || plan.action.includes('undo') ? 'DESTRUCTIVE' : 'SAFE'} />
                    </Box>
                  }
                  secondary={
                    <React.Fragment>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        Trace: {plan.task_id.substring(0, 12)}...
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', fontWeight: 'bold' }}>
                        Target: {plan.deployment || 'Cluster'} ({plan.namespace || 'global'})
                      </Typography>
                      {plan.rationale && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic' }}>
                          "{plan.rationale}"
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

  const isDestructive = plan.action.includes('patch') || plan.action.includes('undo');

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ bgcolor: isDestructive ? '#fff5f5' : 'inherit' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Gavel color={isDestructive ? 'error' : 'warning'} />
          Verify Remediation Plan
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ mt: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">{plan.action.toUpperCase()}</Typography>
            <RiskBadge risk={isDestructive ? 'DESTRUCTIVE' : 'SAFE'} />
          </Box>

          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2 }}>
            <Typography variant="caption" color="text.secondary">TARGET RESOURCES</Typography>
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              {plan.namespace}/{plan.deployment}
            </Typography>
            {plan.replicas && <Typography variant="body2">Scaling to: {plan.replicas} replicas</Typography>}
          </Box>

          {plan.rationale && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Info fontSize="small" color="primary" /> Rationale
              </Typography>
              <Typography variant="body2" sx={{ pl: 3 }}>{plan.rationale}</Typography>
            </Box>
          )}

          <EvidencePreview evidence={plan.evidence || "Trace context: Pod logs and events analyzed by Intelligence Layer."} />

          <TextField
            label="Security Approval Token"
            value={approvalToken}
            onChange={(e) => setApprovalToken(e.target.value)}
            fullWidth
            margin="normal"
            placeholder="Enter token from CLI or secure channel"
            variant="outlined"
            disabled={isApproving}
            autoFocus
          />
          <Typography variant="caption" color="error" sx={{ display: 'block', mt: 1 }}>
            {isDestructive ? '⚠️ WARNING: This action modifies cluster state and requires strict verification.' : ''}
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isApproving}>Cancel</Button>
        <Button 
          onClick={onApprove} 
          disabled={isApproving || !approvalToken}
          variant="contained"
          color={isDestructive ? 'error' : 'primary'}
        >
          {isApproving ? 'Executing...' : 'Authorize Action'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default PendingApprovals;