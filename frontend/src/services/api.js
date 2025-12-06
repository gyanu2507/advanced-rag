import axios from 'axios';
import API_URL from '../config/api.js';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth API
export const authAPI = {
  // Google OAuth
  googleAuth: async (token) => {
    const response = await api.post('/auth/google', { token });
    return response.data;
  },

  // Email/Password
  signup: async (email, password) => {
    const response = await api.post('/auth/email/signup', { email, password });
    return response.data;
  },

  login: async (email, password) => {
    const response = await api.post('/auth/email/login', { email, password });
    return response.data;
  },

  // Phone OTP
  sendOTP: async (phone) => {
    const response = await api.post('/auth/phone/send-otp', { phone });
    return response.data;
  },

  verifyOTP: async (phone, code) => {
    const response = await api.post('/auth/phone/verify', { phone, code });
    return response.data;
  },

  // Token verification
  verifyToken: async (token) => {
    const formData = new FormData();
    formData.append('token', token);
    const response = await api.post('/auth/verify-token', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Google OAuth config
  getGoogleConfig: async () => {
    const response = await api.get('/auth/google/config');
    return response.data;
  },
};

// Document API
export const documentAPI = {
  upload: async (file, userId, onUploadProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);

    console.log('Upload API call:', {
      filename: file.name,
      fileSize: file.size,
      fileType: file.type,
      userId
    });

    const response = await api.post('/upload', formData, {
      // Let axios automatically set Content-Type with boundary for FormData
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onUploadProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onUploadProgress(percentCompleted);
        }
      },
    });
    console.log('Upload response:', response.data);
    return response.data;
  },

  getDocuments: async (userId) => {
    const response = await api.get(`/users/${userId}/documents`);
    // Backend returns { user_id, count, documents: [...] }
    // Extract the documents array
    const data = response.data;

    // If data has a documents property, use it
    if (data && Array.isArray(data.documents)) {
      return data.documents;
    }

    // If data itself is an array, use it
    if (Array.isArray(data)) {
      return data;
    }

    // Otherwise return empty array
    console.warn('Unexpected API response format:', data);
    return [];
  },

  deleteDocument: async (userId, documentId) => {
    const response = await api.delete(`/users/${userId}/documents/${documentId}`);
    return response.data;
  },

  purgeOldData: async (userId, days = 7) => {
    const response = await api.post(`/users/${userId}/purge?days=${days}`);
    return response.data;
  },
};

// Query API
export const queryAPI = {
  query: async (question, userId, documentIds = null) => {
    const response = await api.post('/query', {
      question,
      user_id: userId,
      document_ids: documentIds,
    });
    return response.data;
  },

  // Enhanced query with RAGAS metrics
  enhancedQuery: async (question, userId, documentIds = null, options = {}) => {
    const response = await api.post('/query/enhanced', {
      question,
      user_id: userId,
      document_ids: documentIds,
      use_hyde: options.useHyde ?? true,
      use_rrf: options.useRrf ?? true,
      use_compression: options.useCompression ?? false,
      include_evaluation: options.includeEvaluation ?? true,
    });
    return response.data;
  },

  getHistory: async (userId) => {
    const response = await api.get(`/users/${userId}/queries`);
    return response.data;
  },
};

// Stats API
export const statsAPI = {
  getStats: async (userId) => {
    const response = await api.get(`/users/${userId}/stats`);
    return response.data;
  },
};

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;

