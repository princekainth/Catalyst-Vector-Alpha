import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Description } from '@mui/icons-material';

const EvidencePreview = ({ evidence }) => {
  if (!evidence) return null;

  return (
    <Box sx={{ mt: 1, mb: 1 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <Description fontSize="inherit" /> Evidence Preview
      </Typography>
      <Paper 
        variant="outlined" 
        sx={{ 
          p: 1, 
          bgcolor: '#1e1e1e', 
          color: '#d4d4d4', 
          fontFamily: 'monospace', 
          fontSize: '0.75rem',
          maxHeight: '150px',
          overflow: 'auto',
          mt: 0.5
        }}
      >
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          {typeof evidence === 'string' ? evidence : JSON.stringify(evidence, null, 2)}
        </pre>
      </Paper>
    </Box>
  );
};

export default EvidencePreview;
