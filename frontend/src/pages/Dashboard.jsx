import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, MessageSquare, FileText, LogOut, User, Sparkles, Send, X, CheckCircle2, Loader2, Trash2, AlertCircle, Moon, Sun, FileJson, FileSpreadsheet, FileCode, File, Presentation } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { documentAPI, queryAPI } from '../services/api';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';

// Main Container
const DashboardContainer = styled.div`
  min-height: 100vh;
  width: 100%;
  background: linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 50%, #f3e8ff 100%);
  padding: 0;
  margin: 0;
  box-sizing: border-box;
`;

// Header
const Header = styled.header`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  padding: 1rem 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
`;

const HeaderContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const LogoSection = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const LogoIcon = styled.div`
  padding: 0.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const LogoText = styled.div`
  h1 {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  p {
    font-size: 0.75rem;
    color: #6b7280;
    margin: 0;
    line-height: 1;
  }
`;

const UserSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: #f3f4f6;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`;

const LogoutButton = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: transparent;
  border: none;
  color: #dc2626;
  font-size: 0.875rem;
  font-weight: 600;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: #fef2f2;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2);
  }
`;

// Main Content
const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 1.5rem;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

// Card Component (reused from LoginPage style)
const Card = styled.div`
  background: white;
  border-radius: 1rem;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
`;

const CardTitle = styled.h2`
  font-size: 1.125rem;
  font-weight: 700;
  color: #111827;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const IconWrapper = styled.div`
  color: #667eea;
  display: flex;
  align-items: center;
`;

// Upload Section
const UploadArea = styled.label`
  display: block;
  width: 100%;
  cursor: pointer;
  
  input {
    display: none;
  }
`;

const UploadBox = styled.div`
  width: 100%;
  border: 2px dashed #d1d5db;
  border-radius: 0.75rem;
  padding: 3rem 2rem;
  text-align: center;
  transition: all 0.2s;
  
  &:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
  }
`;

const UploadIconBox = styled.div`
  width: 4rem;
  height: 4rem;
  margin: 0 auto 1rem;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const UploadText = styled.p`
  font-size: 0.875rem;
  font-weight: 600;
  color: #374151;
  margin: 0 0 0.25rem 0;
`;

const UploadSubtext = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  margin: 0;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 0.5rem;
  background: #e5e7eb;
  border-radius: 0.25rem;
  overflow: hidden;
  margin-top: 1rem;
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 0.25rem;
`;

// Documents List
const DocumentsList = styled.div`
  max-height: 400px;
  overflow-y: auto;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f3f4f6;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 3px;
    
    &:hover {
      background: #9ca3af;
    }
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem 1rem;
  color: #9ca3af;
`;

// Skeleton Loading Components
const SkeletonPulse = styled.div`
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #e0e0e0 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 0.375rem;
  
  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
`;

const SkeletonDocItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: ${props => props.isDark ? '#2a2a3e' : '#f9fafb'};
  border-radius: 0.5rem;
`;

const SkeletonCheckbox = styled(SkeletonPulse)`
  width: 1.25rem;
  height: 1.25rem;
  flex-shrink: 0;
  background: ${props => props.isDark
    ? 'linear-gradient(90deg, #3a3a4e 25%, #4a4a5e 50%, #3a3a4e 75%)'
    : 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)'};
  background-size: 200% 100%;
`;

const SkeletonText = styled(SkeletonPulse)`
  height: ${props => props.height || '1rem'};
  width: ${props => props.width || '100%'};
  background: ${props => props.isDark
    ? 'linear-gradient(90deg, #3a3a4e 25%, #4a4a5e 50%, #3a3a4e 75%)'
    : 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)'};
  background-size: 200% 100%;
`;

const SkeletonMeta = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.25rem;
`;

const DocumentItem = styled(motion.div)`
  padding: 0.75rem;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 0.5rem;
  background: ${props => props.selected ? 'rgba(102, 126, 234, 0.1)' : '#f9fafb'};
  border: 2px solid ${props => props.selected ? '#667eea' : 'transparent'};
  
  &:hover {
    background: ${props => props.selected ? 'rgba(102, 126, 234, 0.15)' : '#f3f4f6'};
  }
`;

const DocumentContent = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const Checkbox = styled.div`
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 0.25rem;
  border: 2px solid ${props => props.checked ? '#667eea' : '#d1d5db'};
  background: ${props => props.checked ? '#667eea' : 'transparent'};
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
`;

const DocumentName = styled.span`
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => props.selected ? '#111827' : '#374151'};
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const DocumentInfo = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
`;

const DocumentMeta = styled.div`
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const DeleteButton = styled.button`
  padding: 0.25rem 0.5rem;
  background: transparent;
  border: none;
  color: #dc2626;
  cursor: pointer;
  border-radius: 0.375rem;
  display: flex;
  align-items: center;
  transition: all 0.2s;
  flex-shrink: 0;
  
  &:hover {
    background: #fef2f2;
    color: #991b1b;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;


const SelectAllButton = styled.button`
  width: 100%;
  padding: 0.625rem;
  margin-top: 0.75rem;
  background: transparent;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  font-weight: 600;
  color: #667eea;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: rgba(102, 126, 234, 0.05);
    border-color: #667eea;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
  }
`;

const DocumentCount = styled.span`
  font-size: 0.875rem;
  font-weight: 600;
  color: #667eea;
  background: rgba(102, 126, 234, 0.1);
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
`;

const TitleRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

// Chat Section
const ChatContainer = styled(Card)`
  display: flex;
  flex-direction: column;
  height: calc(100vh - 10rem);
  padding: 0;
  overflow: hidden;
`;

const MessagesArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f3f4f6;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 3px;
    
    &:hover {
      background: #9ca3af;
    }
  }
`;

const EmptyChatState = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
`;

const EmptyChatIcon = styled.div`
  width: 6rem;
  height: 6rem;
  margin: 0 auto 1rem;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-radius: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const MessageBubble = styled(motion.div)`
  max-width: 75%;
  padding: 1rem;
  border-radius: 1rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  align-self: ${props => props.role === 'user' ? 'flex-end' : 'flex-start'};
  background: ${props => props.role === 'user'
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : '#f3f4f6'};
  color: ${props => props.role === 'user' ? 'white' : '#111827'};
  border: ${props => props.role === 'user' ? 'none' : '1px solid #e5e7eb'};
`;

const MessageText = styled.div`
  font-size: 0.875rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
  
  /* Markdown styling */
  h2 {
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.75rem 0;
    color: ${props => props.role === 'user' ? 'white' : '#111827'};
  }
  
  hr {
    border: none;
    border-top: 1px solid ${props => props.role === 'user' ? 'rgba(255,255,255,0.3)' : '#e5e7eb'};
    margin: 0.75rem 0;
  }
  
  ul, ol {
    margin: 0.5rem 0;
    padding-left: 0;
    list-style: none;
  }
  
  li {
    margin: 0.4rem 0;
    padding-left: 0;
  }
  
  strong {
    font-weight: 600;
  }
  
  em {
    font-style: italic;
    opacity: 0.9;
  }
  
  code {
    background: ${props => props.role === 'user' ? 'rgba(255,255,255,0.2)' : '#f3f4f6'};
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 0.8rem;
  }
  
  p {
    margin: 0.25rem 0;
  }
`;

const MessageMeta = styled.div`
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid ${props => props.role === 'user' ? 'rgba(255, 255, 255, 0.2)' : '#e5e7eb'};
  font-size: 0.75rem;
  color: ${props => props.role === 'user' ? 'rgba(255, 255, 255, 0.9)' : '#6b7280'};
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
`;

const MetricsToggle = styled.button`
  background: rgba(102, 126, 234, 0.1);
  border: 1px solid rgba(102, 126, 234, 0.3);
  color: #667eea;
  padding: 0.25rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.7rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: rgba(102, 126, 234, 0.2);
  }
`;

const MetricsPanel = styled.div`
  margin-top: 0.75rem;
  padding: 0.75rem;
  background: linear-gradient(135deg, #f0f4ff 0%, #f3f0ff 100%);
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
`;

const MetricsTitle = styled.div`
  font-size: 0.7rem;
  font-weight: 700;
  color: #667eea;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.375rem;
`;

const MetricRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.25rem 0;
  font-size: 0.75rem;
  
  &:not(:last-child) {
    border-bottom: 1px dashed #e5e7eb;
  }
`;

const MetricLabel = styled.span`
  color: #4b5563;
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const MetricValue = styled.span`
  font-weight: 600;
  color: ${props => {
    const val = props.value;
    if (val >= 0.8) return '#059669';
    if (val >= 0.6) return '#d97706';
    return '#dc2626';
  }};
`;

const MetricBar = styled.div`
  width: 60px;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  margin-left: 0.5rem;
  overflow: hidden;
`;

const MetricFill = styled.div`
  height: 100%;
  width: ${props => Math.min(props.value * 100, 100)}%;
  background: ${props => {
    const val = props.value;
    if (val >= 0.8) return '#10b981';
    if (val >= 0.6) return '#f59e0b';
    return '#ef4444';
  }};
  border-radius: 3px;
  transition: width 0.3s ease;
`;

const LoadingBubble = styled(motion.div)`
  padding: 1rem;
  border-radius: 1rem;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  align-self: flex-start;
  display: flex;
  gap: 0.5rem;
`;

const LoadingDot = styled.div`
  width: 0.5rem;
  height: 0.5rem;
  background: #9ca3af;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
  animation-delay: ${props => props.$delay || '0s'};
  
  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
`;

const ChatInputForm = styled.form`
  padding: 1.5rem;
  border-top: 1px solid #e5e7eb;
  background: #f9fafb;
  display: flex;
  gap: 0.75rem;
`;

const Input = styled.input`
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.75rem;
  font-size: 0.875rem;
  color: #111827;
  outline: none;
  transition: all 0.2s;
  background: white;
  
  &:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
  
  &::placeholder {
    color: #9ca3af;
  }
  
  &:disabled {
    background: #f3f4f6;
    cursor: not-allowed;
  }
`;

const Button = styled(motion.button)`
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 0.75rem;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  
  &:hover:not(:disabled) {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
  }
`;

const SelectedDocsInfo = styled.div`
  padding: 0 1.5rem 0.75rem;
  font-size: 0.75rem;
  color: #6b7280;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const SelectedBadge = styled.span`
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-weight: 600;
`;

const ErrorMessage = styled.div`
  padding: 0.75rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  margin-top: 1rem;
`;

const SuccessMessage = styled.div`
  padding: 0.75rem;
  background: #f0fdf4;
  border: 1px solid #86efac;
  color: #166534;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  margin-top: 1rem;
`;

// Loading State
const LoadingContainer = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #e0f2fe 0%, #e0e7ff 50%, #f3e8ff 100%);
`;

const LoadingSpinner = styled.div`
  width: 3rem;
  height: 3rem;
  border: 3px solid #e5e7eb;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

// Helper function to get file type icon
const getFileIcon = (filename, isSelected) => {
  const ext = filename?.split('.').pop()?.toLowerCase() || '';
  const selectedColor = '#667eea';
  const defaultColor = '#9ca3af';
  const color = isSelected ? selectedColor : defaultColor;

  const iconMap = {
    pdf: { icon: FileText, color: isSelected ? selectedColor : '#ef4444' },
    docx: { icon: FileText, color: isSelected ? selectedColor : '#2563eb' },
    doc: { icon: FileText, color: isSelected ? selectedColor : '#2563eb' },
    txt: { icon: FileText, color },
    md: { icon: FileCode, color: isSelected ? selectedColor : '#6b7280' },
    json: { icon: FileJson, color: isSelected ? selectedColor : '#f59e0b' },
    csv: { icon: FileSpreadsheet, color: isSelected ? selectedColor : '#10b981' },
    xlsx: { icon: FileSpreadsheet, color: isSelected ? selectedColor : '#10b981' },
    xls: { icon: FileSpreadsheet, color: isSelected ? selectedColor : '#10b981' },
    html: { icon: FileCode, color: isSelected ? selectedColor : '#f97316' },
    pptx: { icon: Presentation, color: isSelected ? selectedColor : '#ea580c' },
    ppt: { icon: Presentation, color: isSelected ? selectedColor : '#ea580c' },
  };

  const { icon: IconComponent, color: iconColor } = iconMap[ext] || { icon: File, color };
  return <IconComponent size={18} color={iconColor} />;
};

const Dashboard = () => {
  const { user, logout } = useAuth();
  const { isDarkMode, toggleTheme, colors } = useTheme();
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false); // Toggle for quality metrics
  const [documentsLoading, setDocumentsLoading] = useState(true); // Documents loading state

  // Load chat history from localStorage on mount
  useEffect(() => {
    if (user && user.user_id) {
      const savedMessages = localStorage.getItem(`chat_history_${user.user_id}`);
      if (savedMessages) {
        try {
          const parsed = JSON.parse(savedMessages);
          setMessages(parsed);
        } catch (e) {
          console.error('Failed to parse chat history:', e);
        }
      }
    }
  }, [user]);

  // Save chat history to localStorage when messages change
  useEffect(() => {
    if (user && user.user_id && messages.length > 0) {
      localStorage.setItem(`chat_history_${user.user_id}`, JSON.stringify(messages));
    }
  }, [messages, user]);

  useEffect(() => {
    if (user && user.user_id) {
      loadDocuments();
    }
  }, [user]);

  const loadDocuments = async () => {
    if (!user || !user.user_id) {
      console.log('Cannot load documents: user not available', { user });
      setDocuments([]); // Ensure empty array
      setDocumentsLoading(false);
      return;
    }
    setDocumentsLoading(true);
    try {
      console.log('Loading documents for user:', user.user_id);
      const docs = await documentAPI.getDocuments(user.user_id);
      console.log('Documents loaded:', docs, 'Type:', typeof docs, 'IsArray:', Array.isArray(docs));

      // Ensure docs is always an array - double check
      if (!Array.isArray(docs)) {
        console.error('Documents is not an array!', docs);
        setDocuments([]);
        return;
      }

      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load documents:', error);
      // For guest users, empty array is fine
      setDocuments([]);
    } finally {
      setDocumentsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) {
      setUploadError('Please select a file');
      return;
    }

    if (!user || !user.user_id) {
      setUploadError('User not authenticated. Please log in again.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setUploadError('');
    setUploadSuccess(false);

    console.log('Starting upload:', { filename: file.name, userId: user.user_id, fileSize: file.size });

    try {
      const result = await documentAPI.upload(file, user.user_id, (progress) => {
        console.log('Upload progress:', progress);
        setUploadProgress(progress);
      });

      console.log('Upload successful:', result);
      setUploadProgress(100);
      setUploadSuccess(true);

      // Reload documents after successful upload
      await loadDocuments();

      // Clear success message after 3 seconds
      setTimeout(() => {
        setUploadSuccess(false);
        setUploadProgress(0);
      }, 3000);
    } catch (error) {
      console.error('Upload failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed. Please try again.';
      setUploadError(errorMessage);
      setUploadProgress(0);

      // Clear error after 5 seconds
      setTimeout(() => {
        setUploadError('');
      }, 5000);
    } finally {
      setUploading(false);
      // Reset file input
      e.target.value = '';
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!question.trim() || loading || !user || !user.user_id) return;

    const userMessage = { role: 'user', content: question, timestamp: new Date() };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      // Use enhanced query API to get evaluation metrics
      const result = await queryAPI.enhancedQuery(
        question,
        user.user_id,
        selectedDocs.length > 0 ? selectedDocs : null,
        { includeEvaluation: true }
      );

      const assistantMessage = {
        role: 'assistant',
        content: result.answer,
        sources: result.sources,
        confidence: result.confidence,
        evaluation: result.evaluation, // Store RAGAS metrics
        enhancedFeatures: result.enhanced_features,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Query failed:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date()
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const toggleDocument = (docId) => {
    setSelectedDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const clearChatHistory = () => {
    if (window.confirm('Are you sure you want to clear chat history?')) {
      setMessages([]);
      if (user && user.user_id) {
        localStorage.removeItem(`chat_history_${user.user_id}`);
      }
    }
  };

  const handleDeleteDocument = async (docId, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingId(docId);
    try {
      await documentAPI.deleteDocument(user.user_id, docId);
      await loadDocuments();
      // Remove from selected docs if it was selected
      setSelectedDocs((prev) => prev.filter((id) => id !== docId));
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Failed to delete document. Please try again.');
    } finally {
      setDeletingId(null);
    }
  };


  // Show loading state if user is not yet available
  if (!user || !user.user_id) {
    console.log('Dashboard: Waiting for user', { user });
    return (
      <LoadingContainer>
        <div style={{ textAlign: 'center' }}>
          <LoadingSpinner />
          <p style={{ marginTop: '1rem', color: '#6b7280', fontSize: '0.875rem' }}>
            Loading dashboard...
          </p>
        </div>
      </LoadingContainer>
    );
  }

  console.log('Dashboard: Rendering with user', { user });

  return (
    <DashboardContainer style={{ background: colors.bg }}>
      <Header style={{ background: isDarkMode ? 'rgba(30, 30, 46, 0.95)' : 'rgba(255, 255, 255, 0.95)', borderBottom: `1px solid ${colors.cardBorder}` }}>
        <HeaderContent>
          <LogoSection>
            <LogoIcon>
              <Sparkles size={24} color="white" />
            </LogoIcon>
            <LogoText>
              <h1>AI Document Q&A</h1>
              <p style={{ color: colors.textMuted }}>Intelligent Document Understanding</p>
            </LogoText>
          </LogoSection>
          <UserSection>
            <motion.button
              onClick={toggleTheme}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              style={{
                padding: '0.5rem',
                background: isDarkMode ? '#374151' : '#f3f4f6',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {isDarkMode ? <Sun size={20} color="#fbbf24" /> : <Moon size={20} color="#6b7280" />}
            </motion.button>
            <UserInfo style={{ background: isDarkMode ? '#374151' : '#f3f4f6', color: colors.text }}>
              <User size={18} />
              <span>{user.email || (user.isGuest ? 'Guest User' : user.user_id)}</span>
            </UserInfo>
            <LogoutButton onClick={logout}>
              <LogOut size={18} />
              <span>Sign Out</span>
            </LogoutButton>
          </UserSection>
        </HeaderContent>
      </Header>

      <MainContent>
        {/* Sidebar */}
        <div>
          {/* Upload Section */}
          <Card style={{ background: colors.cardBg, boxShadow: colors.shadow }}>
            <CardTitle style={{ color: colors.text }}>
              <IconWrapper>
                <Upload size={22} />
              </IconWrapper>
              <span>Upload Document</span>
            </CardTitle>
            <UploadArea>
              <input
                type="file"
                onChange={handleFileUpload}
                accept=".pdf,.txt,.docx,.md,.csv,.json,.html,.xlsx,.pptx"
              />
              <UploadBox>
                {uploading ? (
                  <div>
                    <Loader2 size={40} color="#667eea" style={{ margin: '0 auto 1rem', animation: 'spin 1s linear infinite' }} />
                    <ProgressBar>
                      <ProgressFill
                        initial={{ width: 0 }}
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </ProgressBar>
                    <UploadText>Uploading... {uploadProgress}%</UploadText>
                  </div>
                ) : (
                  <>
                    <UploadIconBox>
                      <Upload size={32} color="#667eea" />
                    </UploadIconBox>
                    <UploadText>Click to upload</UploadText>
                    <UploadSubtext>PDF, TXT, DOCX, MD, CSV, etc.</UploadSubtext>
                  </>
                )}
              </UploadBox>
            </UploadArea>
            {uploadError && (
              <ErrorMessage>{uploadError}</ErrorMessage>
            )}
            {uploadSuccess && (
              <SuccessMessage>✓ Document uploaded successfully!</SuccessMessage>
            )}
          </Card>

          {/* Documents List */}
          <Card style={{ background: colors.cardBg, boxShadow: colors.shadow }}>
            <TitleRow>
              <CardTitle style={{ color: colors.text }}>
                <IconWrapper>
                  <FileText size={22} />
                </IconWrapper>
                <span>Documents</span>
              </CardTitle>
              <DocumentCount>{documents.length}</DocumentCount>
            </TitleRow>
            <DocumentsList>
              {documentsLoading ? (
                // Skeleton loading state
                <>
                  {[1, 2, 3].map((i) => (
                    <SkeletonDocItem key={i} isDark={isDarkMode}>
                      <SkeletonCheckbox isDark={isDarkMode} />
                      <div style={{ flex: 1 }}>
                        <SkeletonText isDark={isDarkMode} width={`${60 + i * 10}%`} height="0.875rem" />
                        <SkeletonMeta>
                          <SkeletonText isDark={isDarkMode} width="4rem" height="0.65rem" style={{ marginTop: '0.25rem' }} />
                          <SkeletonText isDark={isDarkMode} width="3rem" height="0.65rem" style={{ marginTop: '0.25rem' }} />
                        </SkeletonMeta>
                      </div>
                    </SkeletonDocItem>
                  ))}
                </>
              ) : !Array.isArray(documents) || documents.length === 0 ? (
                <EmptyState>
                  <FileText size={48} color="#d1d5db" style={{ marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.875rem', fontWeight: 500, margin: '0 0 0.25rem 0', color: colors.textMuted }}>
                    No documents yet
                  </p>
                  <p style={{ fontSize: '0.75rem', margin: 0, color: colors.textMuted }}>
                    Upload your first document above
                  </p>
                </EmptyState>
              ) : (
                <>
                  {documents.map((doc) => (
                    <DocumentItem
                      key={doc.id}
                      selected={selectedDocs.includes(doc.id)}
                      onClick={() => toggleDocument(doc.id)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <DocumentContent>
                        <Checkbox checked={selectedDocs.includes(doc.id)}>
                          {selectedDocs.includes(doc.id) && (
                            <CheckCircle2 size={14} color="white" />
                          )}
                        </Checkbox>
                        {getFileIcon(doc.filename, selectedDocs.includes(doc.id))}
                        <DocumentInfo>
                          <DocumentName selected={selectedDocs.includes(doc.id)}>
                            {doc.filename}
                          </DocumentName>
                          <DocumentMeta>
                            {doc.days_remaining !== undefined && (
                              <>
                                {doc.will_purge_soon ? (
                                  <span style={{ color: '#dc2626', fontWeight: 600 }}>
                                    <AlertCircle size={12} style={{ display: 'inline', marginRight: '0.25rem' }} />
                                    Auto-deletes in {doc.days_remaining} day{doc.days_remaining !== 1 ? 's' : ''}
                                  </span>
                                ) : (
                                  <span style={{ color: '#6b7280' }}>
                                    Lifetime: {doc.days_remaining} day{doc.days_remaining !== 1 ? 's' : ''} remaining
                                  </span>
                                )}
                              </>
                            )}
                            {doc.file_size && (
                              <span>• {(doc.file_size / 1024).toFixed(1)} KB</span>
                            )}
                          </DocumentMeta>
                        </DocumentInfo>
                        <DeleteButton
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteDocument(doc.id, doc.filename);
                          }}
                          disabled={deletingId === doc.id}
                          title="Delete document"
                        >
                          {deletingId === doc.id ? (
                            <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                          ) : (
                            <Trash2 size={16} />
                          )}
                        </DeleteButton>
                      </DocumentContent>
                    </DocumentItem>
                  ))}
                  {Array.isArray(documents) && documents.length > 0 && (
                    <>
                      <SelectAllButton
                        onClick={() => {
                          if (selectedDocs.length === documents.length) {
                            setSelectedDocs([]);
                          } else {
                            setSelectedDocs(documents.map((d) => d.id));
                          }
                        }}
                      >
                        {selectedDocs.length === documents.length ? 'Deselect All' : 'Select All'}
                      </SelectAllButton>
                    </>
                  )}
                </>
              )}
            </DocumentsList>
          </Card>
        </div>

        {/* Chat Section */}
        <ChatContainer style={{ background: colors.cardBg, boxShadow: colors.shadow }}>
          {/* Chat Header */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '1rem 1.5rem',
            borderBottom: `1px solid ${colors.cardBorder || '#e5e7eb'}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <MessageSquare size={20} color="#667eea" />
              <span style={{ fontWeight: 600, color: colors.text }}>Chat</span>
              {messages.length > 0 && (
                <span style={{
                  fontSize: '0.75rem',
                  background: isDarkMode ? '#374151' : '#f3f4f6',
                  padding: '0.125rem 0.5rem',
                  borderRadius: '0.75rem',
                  color: colors.textMuted
                }}>
                  {messages.length} messages
                </span>
              )}
            </div>
            {messages.length > 0 && (
              <button
                onClick={clearChatHistory}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  padding: '0.375rem 0.75rem',
                  background: 'transparent',
                  border: '1px solid #dc2626',
                  borderRadius: '0.375rem',
                  color: '#dc2626',
                  fontSize: '0.75rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                <Trash2 size={14} />
                Clear
              </button>
            )}
          </div>
          <MessagesArea style={{ background: colors.cardBg }}>
            {messages.length === 0 ? (
              <EmptyChatState>
                <div>
                  <EmptyChatIcon>
                    <MessageSquare size={48} color="#667eea" />
                  </EmptyChatIcon>
                  <p style={{ fontSize: '1.125rem', fontWeight: 600, color: colors.text, marginBottom: '0.5rem' }}>
                    Start asking questions
                  </p>
                  <p style={{ fontSize: '0.875rem', color: colors.textMuted }}>
                    Upload documents and ask questions about them
                  </p>
                </div>
              </EmptyChatState>
            ) : (
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <MessageBubble
                    key={idx}
                    role={msg.role}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <MessageText role={msg.role}>
                      {msg.role === 'assistant' ? (
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </MessageText>
                    {msg.role === 'assistant' && (msg.sources?.length > 0 || msg.evaluation) && (
                      <>
                        <MessageMeta role={msg.role}>
                          {msg.sources?.length > 0 && (
                            <span>📚 {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}</span>
                          )}
                          {msg.confidence !== undefined && msg.confidence !== null && (
                            <span>🎯 {Math.round(msg.confidence * 100)}% confidence</span>
                          )}
                          {msg.evaluation && (
                            <MetricsToggle onClick={() => setShowMetrics(!showMetrics)}>
                              {showMetrics ? '📊 Hide Details' : '📊 Show Details'}
                            </MetricsToggle>
                          )}
                        </MessageMeta>

                        {showMetrics && msg.evaluation && (
                          <MetricsPanel>
                            <MetricsTitle>✨ Answer Quality Metrics</MetricsTitle>
                            <MetricRow>
                              <MetricLabel>🎯 Relevance</MetricLabel>
                              <div style={{ display: 'flex', alignItems: 'center' }}>
                                <MetricValue value={msg.evaluation.answer_relevancy}>
                                  {Math.round(msg.evaluation.answer_relevancy * 100)}%
                                </MetricValue>
                                <MetricBar>
                                  <MetricFill value={msg.evaluation.answer_relevancy} />
                                </MetricBar>
                              </div>
                            </MetricRow>
                            <MetricRow>
                              <MetricLabel>✅ Accuracy</MetricLabel>
                              <div style={{ display: 'flex', alignItems: 'center' }}>
                                <MetricValue value={msg.evaluation.answer_faithfulness}>
                                  {Math.round(msg.evaluation.answer_faithfulness * 100)}%
                                </MetricValue>
                                <MetricBar>
                                  <MetricFill value={msg.evaluation.answer_faithfulness} />
                                </MetricBar>
                              </div>
                            </MetricRow>
                            <MetricRow>
                              <MetricLabel>📄 Grounding</MetricLabel>
                              <div style={{ display: 'flex', alignItems: 'center' }}>
                                <MetricValue value={msg.evaluation.context_relevancy}>
                                  {Math.round(msg.evaluation.context_relevancy * 100)}%
                                </MetricValue>
                                <MetricBar>
                                  <MetricFill value={msg.evaluation.context_relevancy} />
                                </MetricBar>
                              </div>
                            </MetricRow>
                          </MetricsPanel>
                        )}
                      </>
                    )}
                  </MessageBubble>
                ))}
              </AnimatePresence>
            )}
            {loading && (
              <LoadingBubble
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <LoadingDot $delay="0s" />
                <LoadingDot $delay="0.2s" />
                <LoadingDot $delay="0.4s" />
              </LoadingBubble>
            )}
          </MessagesArea>
          <div>
            {selectedDocs.length > 0 && (
              <SelectedDocsInfo>
                <span style={{ fontWeight: 500 }}>Searching in:</span>
                <SelectedBadge>
                  {selectedDocs.length} document{selectedDocs.length > 1 ? 's' : ''}
                </SelectedBadge>
              </SelectedDocsInfo>
            )}
            <ChatInputForm onSubmit={handleQuery}>
              <Input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about your documents..."
                disabled={loading}
              />
              <Button
                type="submit"
                disabled={loading || !question.trim()}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {loading ? (
                  <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <>
                    <Send size={20} />
                    <span>Send</span>
                  </>
                )}
              </Button>
            </ChatInputForm>
          </div>
        </ChatContainer>
      </MainContent>
    </DashboardContainer>
  );
};

export default Dashboard;