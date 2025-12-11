import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('cva_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Handle specific error statuses
      if (error.response.status === 401) {
        // Handle unauthorized access
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const getSystemHealth = async () => {
  try {
    const response = await api.get('/health/detailed');
    return response.data;
  } catch (error) {
    console.error('Error fetching system health:', error);
    throw error;
  }
};

export const getAgentsStatus = async () => {
  try {
    const response = await api.get('/agents');
    return response.data;
  } catch (error) {
    console.error('Error fetching agents status:', error);
    throw error;
  }
};

export const getMetrics = async () => {
  try {
    const response = await api.get('/metrics/stats');
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    throw error;
  }
};

export const getTaskHistory = async () => {
  try {
    const response = await api.get('/task_history');
    return response.data;
  } catch (error) {
    console.error('Error fetching task history:', error);
    throw error;
  }
};

export const executeCommand = async (command) => {
  try {
    const response = await api.post('/command', { command });
    return response.data;
  } catch (error) {
    console.error('Error executing command:', error);
    throw error;
  }
};

export const spawnAgent = async (purpose, context = {}) => {
  try {
    const response = await api.post('/agents/spawn', { purpose, context });
    return response.data;
  } catch (error) {
    console.error('Error spawning agent:', error);
    throw error;
  }
};

export const getPendingPlans = async () => {
  try {
    const response = await api.get('/catalyst/plans');
    return response.data;
  } catch (error) {
    console.error('Error fetching pending plans:', error);
    throw error;
  }
};

export const approvePlan = async (planData) => {
  try {
    const response = await api.post('/approve', planData);
    return response.data;
  } catch (error) {
    console.error('Error approving plan:', error);
    throw error;
  }
};

export default api;