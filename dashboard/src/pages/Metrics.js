import React, { useState, useEffect } from 'react';
import { Typography, Box, Card, CardContent, Alert, CircularProgress, Grid } from '@mui/material';
import { BarChart, LineChart, PieChart } from '@mui/x-charts';
import api from '../api';

function Metrics() {
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await api.get('/metrics/stats');
        if (response.data && response.data.status === 'ok') {
          setMetricsData(response.data.data);
        }
      } catch (err) {
        console.error('Failed to fetch metrics:', err);
        setError('Failed to load metrics data');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 20000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          System Metrics
        </Typography>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      </Box>
    );
  }

  // Generate sample data if no real data is available
  const generateSampleData = () => ({
    cpuUsage: [25, 35, 45, 30, 50, 60, 40],
    memoryUsage: [40, 45, 50, 55, 60, 50, 45],
    taskTimes: [120, 180, 90, 210, 150, 190, 130],
    toolUsage: [
      { id: 0, value: 45, label: 'Web Search' },
      { id: 1, value: 30, label: 'File System' },
      { id: 2, value: 20, label: 'Database' },
      { id: 3, value: 5, label: 'Other' }
    ],
    successRate: 92.5
  });

  const data = metricsData || generateSampleData();

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Metrics
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        {metricsData ? 'Real-time performance metrics' : 'Sample metrics data (connect API for real data)'}
      </Alert>

      <Grid container spacing={3}>
        {/* CPU and Memory Usage Chart */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Resource Usage Over Time
              </Typography>
              <LineChart
                xAxis={[{ data: ['0s', '10s', '20s', '30s', '40s', '50s', '60s'] }]}
                series={[
                  { 
                    data: data.cpuUsage, 
                    label: 'CPU Usage (%)',
                    color: '#90caf9'
                  },
                  { 
                    data: data.memoryUsage, 
                    label: 'Memory Usage (%)',
                    color: '#f48fb1'
                  }
                ]}
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Task Performance Chart */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Task Execution Times (ms)
              </Typography>
              <BarChart
                xAxis={[{ 
                  data: ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7'],
                  scaleType: 'band'
                }]}
                series={[{ data: data.taskTimes, color: '#4caf50' }]}
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Tool Usage Chart */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tool Usage Distribution
              </Typography>
              <PieChart
                series={[{ 
                  data: data.toolUsage,
                  innerRadius: 30,
                  outerRadius: 100,
                  paddingAngle: 5,
                  cornerRadius: 5
                }]}
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Metrics Summary */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Summary
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Success Rate
                      </Typography>
                      <Typography variant="h5" color="success.main">
                        {data.successRate}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Avg CPU
                      </Typography>
                      <Typography variant="h5">
                        {Math.round(data.cpuUsage.reduce((a, b) => a + b, 0) / data.cpuUsage.length)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Avg Memory
                      </Typography>
                      <Typography variant="h5">
                        {Math.round(data.memoryUsage.reduce((a, b) => a + b, 0) / data.memoryUsage.length)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Avg Task Time
                      </Typography>
                      <Typography variant="h5">
                        {Math.round(data.taskTimes.reduce((a, b) => a + b, 0) / data.taskTimes.length)}ms
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Metrics;