import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 45000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const getHealth = () => api.get('/health').then(r => r.data);
export const getStates = () => api.get('/states').then(r => r.data);
export const getDistricts = (state) => api.get(`/districts?state=${encodeURIComponent(state)}`).then(r => r.data);
export const getYears = () => api.get('/years').then(r => r.data);
export const submitPrediction = (data) => api.post('/predictions', data).then(r => r.data);
export const getPredictionById = (id) => api.get(`/predictions/${id}`).then(r => r.data);
export const getHistory = (params) => api.get('/predictions/history', { params }).then(r => r.data);
export const getModelMetadata = () => api.get('/model/metadata').then(r => r.data);
export const getModelPerformance = () => api.get('/model/performance').then(r => r.data);
export const getDistrictHistorical = (state, district) => 
  api.get(`/districts/historical?state=${encodeURIComponent(state)}&district=${encodeURIComponent(district)}`).then(r => r.data);

export const exportHistoryCsvUrl = `${API_BASE}/predictions/history/export`;

export default api;
