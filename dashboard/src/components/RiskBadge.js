import React from 'react';
import { Chip } from '@mui/material';
import { Shield, Warning, Gavel } from '@mui/icons-material';

const RiskBadge = ({ risk }) => {
  const r = risk?.toUpperCase() || 'SAFE';
  
  if (r === 'DESTRUCTIVE') {
    return (
      <Chip 
        icon={<Gavel style={{ color: 'white' }} />} 
        label="DESTRUCTIVE" 
        size="small" 
        sx={{ bgcolor: '#d32f2f', color: 'white', fontWeight: 'bold' }} 
      />
    );
  }
  
  if (r === 'CAUTION') {
    return (
      <Chip 
        icon={<Warning style={{ color: 'white' }} />} 
        label="CAUTION" 
        size="small" 
        sx={{ bgcolor: '#ed6c02', color: 'white', fontWeight: 'bold' }} 
      />
    );
  }

  return (
    <Chip 
      icon={<Shield style={{ color: 'white' }} />} 
      label="SAFE" 
      size="small" 
      sx={{ bgcolor: '#2e7d32', color: 'white', fontWeight: 'bold' }} 
    />
  );
};

export default RiskBadge;
