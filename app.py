"""
Streamlit frontend for the AI Document Q&A System.
"""
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import time
from typing import Optional, List

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra Modern Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Keep header visible for sidebar toggle button */
    /* header {visibility: hidden;} */
    
    /* Creative Background - Clean Version */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientFlow 15s ease infinite;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Container - Clean */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Remove empty boxes and fix alignment */
    .element-container {
        margin-bottom: 0.25rem !important;
        padding: 0.25rem 0 !important;
    }
    
    /* Clean sidebar - Compact */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.98) !important;
    }
    
    /* Reduce sidebar spacing */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.1rem !important;
        padding: 0.1rem 0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Ensure sidebar toggle button is visible */
    button[kind="header"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Sidebar toggle button styling */
    [data-testid="stSidebar"] [data-testid="collapsedControl"] {
        visibility: visible !important;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: gradientShift 5s ease infinite;
        letter-spacing: -2px;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Feature Badges */
    .feature-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        font-size: 0.9rem;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px 0 rgba(102, 126, 234, 0.39);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Secondary Button */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #475569;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.1);
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        transform: translateY(-2px);
    }
    
    /* Chat Messages - Classy & Elegant */
    .stChatMessage {
        padding: 1.25rem 1.5rem !important;
        border-radius: 16px !important;
        margin: 1rem 0 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        animation: slideIn 0.4s ease-out !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12), 0 2px 5px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    
    /* User messages - Right aligned style */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-left: 3px solid #667eea !important;
    }
    
    /* Assistant messages - Left aligned style */
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%) !important;
        border-left: 3px solid #10b981 !important;
    }
    
    /* Chat message content */
    .stChatMessage .stMarkdown {
        line-height: 1.7 !important;
        color: #1f2937 !important;
        font-size: 1rem !important;
    }
    
    .stChatMessage .stMarkdown p {
        margin: 0.75rem 0 !important;
    }
    
    .stChatMessage .stMarkdown strong {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(15px) scale(0.98);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    /* Chat input - Elegant */
    .stChatInput {
        margin-top: 1.5rem !important;
    }
    
    .stChatInput>div>div>div>textarea {
        border-radius: 16px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput>div>div>div>textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15), 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 1) !important;
    }
    
    /* Source expander - Classy */
    .stExpander {
        margin-top: 0.75rem !important;
    }
    
    .stExpander .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        font-weight: 600 !important;
        color: #667eea !important;
    }
    
    .stExpander .streamlit-expanderContent {
        background: rgba(248, 250, 252, 0.8) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-top: 0.5rem !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Source text styling */
    .stExpander .stText {
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border-left: 3px solid #667eea !important;
        margin: 0.5rem 0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }
    
    /* Sidebar Content Styling */
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        color: white;
        padding: 1rem;
        border-radius: 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar content sections */
    [data-testid="stSidebar"] .element-container {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #64748b;
        font-size: 0.95rem;
    }
    
    /* File Uploader - Clean */
    .uploadedFile {
        border-radius: 8px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9) !important;
        margin: 0.25rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Input Fields - Clean */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 0.75rem;
        background: white !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Chat Input - Clean */
    .stChatInput>div>div>div>textarea {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: white !important;
    }
    
    .stChatInput>div>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File Uploader - Clean */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px;
        border: 2px dashed #cbd5e1 !important;
    }
    
    /* Status Badges */
    .status-online {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .status-offline {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    
    /* Cards with Glass Effect */
    .info-card {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Remove unnecessary boxes from markdown */
    div[data-testid="stMarkdownContainer"] {
        padding: 0.5rem 0;
        margin: 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #667eea;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
        margin: 1.5rem 0;
    }
    
    /* Messages - Clean */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics - Clean */
    [data-testid="stMetricContainer"] {
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Pulse Animation for Status */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    </style>
""", unsafe_allow_html=True)

# API URL - Can be set via environment variable (for deployment)
API_URL = os.getenv("API_URL", os.getenv("HF_API_URL", "http://localhost:8000"))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "user_id" not in st.session_state:
    import uuid
    st.session_state.user_id = str(uuid.uuid4())[:8]  # Generate a short user ID
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_phone" not in st.session_state:
    st.session_state.user_phone = None
if "phone_otp_sent" not in st.session_state:
    st.session_state.phone_otp_sent = False
if "phone_for_verification" not in st.session_state:
    st.session_state.phone_for_verification = None


def check_api_health():
    """Check if the API is running."""
    try:
        # Use a slightly longer timeout and verify the response
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            # Verify we got a valid JSON response
            data = response.json()
            if data.get("status") == "healthy":
                return True
        return False
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
        # Silently fail - don't print errors in production
        return False


def upload_document(file, user_id: str):
    """Upload document to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"user_id": user_id}
        response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=120)  # Increased timeout for model loading
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("detail", "Unknown error") if response.content else "Upload failed"
            st.error(f"Upload failed: {error_msg}")
            return None
    except requests.exceptions.Timeout:
        st.error("Upload timed out. The server may be loading AI models. Please try again in a moment.")
        return None
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None


def query_documents(question: str, user_id: str, document_ids: Optional[List[int]] = None):
    """Query documents via the API."""
    try:
        payload = {"question": question, "user_id": user_id}
        if document_ids:
            payload["document_ids"] = document_ids
        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None


def clear_documents(user_id: str):
    """Clear all documents for a user."""
    try:
        response = requests.delete(f"{API_URL}/clear", params={"user_id": user_id}, timeout=5)
        return response.status_code == 200
    except:
        return False

def purge_old_data(user_id: str, days: int = 7):
    """Purge data older than specified days for a user."""
    try:
        response = requests.post(f"{API_URL}/purge/{user_id}", params={"days": days}, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_user_stats(user_id: str):
    """Get user statistics from database."""
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/stats", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def authenticate_with_google(token: str):
    """Authenticate with Google OAuth token."""
    try:
        response = requests.post(
            f"{API_URL}/auth/google",
            json={"token": token},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def send_phone_otp(phone: str):
    """Send OTP to phone number."""
    try:
        response = requests.post(
            f"{API_URL}/auth/phone/send-otp",
            json={"phone": phone},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def verify_phone_otp(phone: str, code: str):
    """Verify phone OTP."""
    try:
        response = requests.post(
            f"{API_URL}/auth/phone/verify",
            json={"phone": phone, "code": code},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def verify_auth_token(token: str):
    """Verify authentication token."""
    try:
        response = requests.post(
            f"{API_URL}/auth/verify-token",
            data={"token": token},
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_user_documents(user_id: str):
    """Get user documents from database."""
    try:
        response = requests.get(f"{API_URL}/users/{user_id}/documents", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


# Check for token in URL (from OAuth callback)
if "token" in st.query_params:
    token = st.query_params["token"]
    token_info = verify_auth_token(token)
    if token_info and token_info.get("status") == "valid":
        st.session_state.auth_token = token
        st.session_state.authenticated = True
        st.session_state.user_id = token_info.get("user_id")
        st.session_state.user_email = token_info.get("email")
        st.rerun()

# Authentication Check
if not st.session_state.authenticated and st.session_state.auth_token:
    # Verify token on page load
    token_info = verify_auth_token(st.session_state.auth_token)
    if token_info and token_info.get("status") == "valid":
        st.session_state.authenticated = True
        st.session_state.user_id = token_info.get("user_id")
        st.session_state.user_email = token_info.get("email")
    else:
        st.session_state.auth_token = None

# Login Page (if not authenticated) - Compact Single Page Layout
if not st.session_state.authenticated:
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .login-title {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Two-column layout - Left: Content, Right: Sign-in Options
    col_left, col_right = st.columns([1.2, 1], gap="large")
    
    with col_left:
        st.markdown('<h1 class="login-title">✨ AI Document Q&A</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;">Intelligent Document Understanding with Advanced RAG</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='margin-top: 1rem;'>
            <h3 style='color: #1f2937; margin-bottom: 0.75rem; font-size: 1.3rem;'>🚀 Key Features</h3>
            <ul style='color: #64748b; line-height: 2; font-size: 0.95rem; margin: 0; padding-left: 1.5rem;'>
                <li>📄 Upload & process multiple document formats</li>
                <li>💡 Ask questions with AI-powered answers</li>
                <li>🔍 Advanced semantic & hybrid search</li>
                <li>🎯 Confidence scoring for reliability</li>
                <li>💬 Conversation memory & context</li>
                <li>🔒 Secure & private document storage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<h2 style="color: #1f2937; margin-bottom: 1rem; font-size: 1.5rem;">Sign In</h2>', unsafe_allow_html=True)
        
        # Guest Option (Primary)
        if st.button("🚀 Continue as Guest", use_container_width=True, type="primary", key="guest_login_btn"):
            st.session_state.authenticated = True
            st.rerun()
        
        st.markdown("---")
        
        # Get Google OAuth config
        try:
            config_response = requests.get(f"{API_URL}/auth/google/config", timeout=5)
            google_config = config_response.json() if config_response.status_code == 200 else {}
            google_enabled = google_config.get("enabled", False)
            client_id = google_config.get("client_id", "")
        except:
            google_enabled = False
            client_id = ""
        
        # Google Sign-In
        st.markdown("**📧 Sign in with Google**")
        
        if google_enabled and client_id:
            st.markdown(f"""
            <div style="margin: 0.5rem 0; text-align: center;">
                <div id="g_id_onload"
                    data-client_id="{client_id}"
                    data-callback="handleGoogleSignIn"
                    data-auto_prompt="false"
                    data-context="signin">
                </div>
                <div class="g_id_signin" 
                    data-type="standard"
                    data-size="large"
                    data-theme="outline"
                    data-text="sign_in_with"
                    data-shape="rectangular"
                    data-logo_alignment="left"
                    style="margin: 0.5rem auto; display: inline-block;">
                </div>
            </div>
            
            <script src="https://accounts.google.com/gsi/client" async defer></script>
            <script>
            window.handleGoogleSignIn = function(response) {{
                if (response.credential) {{
                    fetch('{API_URL}/auth/google', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{token: response.credential}})
                    }})
                    .then(res => res.json())
                    .then(data => {{
                        if (data.status === 'success') {{
                            window.location.href = '/?token=' + encodeURIComponent(data.token);
                        }} else {{
                            alert('Authentication failed: ' + (data.message || 'Unknown error'));
                        }}
                    }})
                    .catch(err => alert('Error: ' + err.message));
                }}
            }};
            </script>
            """, unsafe_allow_html=True)
            st.caption("🔒 Secure authentication")
        else:
            st.info("⚠️ Google OAuth not configured")
        
        st.markdown("---")
        
        # Phone Sign-In
        st.markdown("**📱 Sign in with Phone**")
        
        phone_input = st.text_input("Phone Number", placeholder="+1234567890", key="phone_login_input", label_visibility="collapsed")
        
        if not st.session_state.phone_otp_sent:
            if st.button("📱 Send Code", use_container_width=True):
                if phone_input:
                    result = send_phone_otp(phone_input)
                    if result and result.get("status") == "success":
                        st.session_state.phone_otp_sent = True
                        st.session_state.phone_for_verification = phone_input
                        st.success(f"✅ Code sent!")
                    else:
                        st.error("❌ Failed to send OTP")
                else:
                    st.warning("⚠️ Enter phone number")
        else:
            st.info(f"📱 Code sent to {st.session_state.phone_for_verification}")
            otp_code = st.text_input("Enter 6-digit code", max_chars=6, key="otp_input")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Verify", use_container_width=True):
                    if otp_code and len(otp_code) == 6:
                        result = verify_phone_otp(st.session_state.phone_for_verification, otp_code)
                        if result and result.get("status") == "success":
                            st.session_state.auth_token = result.get("token")
                            st.session_state.authenticated = True
                            st.session_state.user_id = result.get("user_id")
                            st.session_state.user_phone = result.get("user", {}).get("phone")
                            st.session_state.phone_otp_sent = False
                            st.session_state.phone_for_verification = None
                            st.success("✅ Verified!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('message', 'Failed')}")
                    else:
                        st.warning("⚠️ Enter 6 digits")
            with col_b:
                if st.button("🔄 Resend", use_container_width=True):
                    result = send_phone_otp(st.session_state.phone_for_verification)
                    if result and result.get("status") == "success":
                        st.success("✅ Resent!")
                    else:
                        st.error("❌ Failed")
    
    st.stop()

# Main UI - Ultra Modern Design
st.markdown('<h1 class="main-header">✨ AI Document Q&A</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2.5rem;'>
        <p class="subtitle">Intelligent Document Understanding with Advanced RAG</p>
        <div style='margin-top: 1rem;'>
            <span class="feature-badge">🚀 Smart Chunking</span>
            <span class="feature-badge">🎯 Advanced Embeddings</span>
            <span class="feature-badge">💡 Intelligent Retrieval</span>
            <span class="feature-badge">🔍 Hybrid Search</span>
            <span class="feature-badge">✨ Query Rewriting</span>
            <span class="feature-badge">🎯 Confidence Scoring</span>
            <span class="feature-badge">💬 Conversation Memory</span>
            <span class="feature-badge">🔒 User Privacy</span>
            <span class="feature-badge">💾 Database Tracking</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Ultra Compact Design
with st.sidebar:
    st.markdown("### 📚 Document Q&A")
    
    # User Authentication Info
    if st.session_state.authenticated:
        if st.session_state.user_email:
            st.markdown(f"👤 **{st.session_state.user_email}**")
        elif st.session_state.user_phone:
            st.markdown(f"📱 **{st.session_state.user_phone}**")
        else:
            st.markdown(f"👤 **User:** {st.session_state.user_id}")
        
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_token = None
            st.session_state.user_email = None
            st.session_state.user_phone = None
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
    
    # User Session - Compact
    # Initialize user_id_input in session state if not exists
    if "user_id_input" not in st.session_state:
        st.session_state.user_id_input = st.session_state.user_id
    
    # Check if we need to generate a new ID (from previous button click)
    if "generate_new_id" in st.session_state and st.session_state.generate_new_id:
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.user_id = new_id
        st.session_state.user_id_input = new_id  # Update the widget's session state
        st.session_state.messages = []
        st.session_state.document_uploaded = False
        st.session_state.generate_new_id = False
        st.rerun()
    
    # User ID Input - Full width
    # Only use key parameter - it will sync with session state automatically
    user_id_input = st.text_input(
        "User ID",
        key="user_id_input",
        help="Each user has isolated document storage",
        label_visibility="visible"
    )
    
    # Generate New ID Button - Below input
    if st.button("🔄 Generate New User ID", use_container_width=True, key="btn_new_id"):
        st.session_state.generate_new_id = True
        st.rerun()
    
    # Sync: Update session state if user manually changed the input
    if user_id_input != st.session_state.user_id:
        st.session_state.user_id = user_id_input
        st.session_state.messages = []
        st.session_state.document_uploaded = False
        st.success(f"✅ Switched")
    
    # Status & Stats - Compact
    col1, col2 = st.columns(2)
    with col1:
        api_healthy = check_api_health()
        if api_healthy:
            st.markdown('<p style="color: #10b981; font-weight: 600; margin: 0; font-size: 0.85rem;">🟢 Online</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #ef4444; font-weight: 600; margin: 0; font-size: 0.85rem;">🔴 Offline</p>', unsafe_allow_html=True)
            if st.button("🔄 Retry", key="retry_connection", use_container_width=True):
                st.rerun()
            st.caption("Backend: `python3 -m uvicorn backend:app --reload`")
    with col2:
        if check_api_health():
            try:
                stats = get_user_stats(st.session_state.user_id)
                if stats:
                    doc_count = stats.get('document_count', 0)
                    st.markdown(f'<p style="margin: 0; color: #64748b; font-size: 0.85rem;">📄 {doc_count}</p>', unsafe_allow_html=True)
            except:
                pass
    
    # File Selection - Compact
    if check_api_health():
        try:
            docs_data = get_user_documents(st.session_state.user_id)
        except:
            docs_data = None
        
        if docs_data and docs_data.get("count", 0) > 0:
            documents = docs_data.get("documents", [])
            
            if "selected_document_ids" not in st.session_state:
                st.session_state.selected_document_ids = [doc['id'] for doc in documents]
            
            current_doc_ids = {doc['id'] for doc in documents}
            existing_selected = set(st.session_state.selected_document_ids)
            new_docs = current_doc_ids - existing_selected
            if new_docs:
                st.session_state.selected_document_ids.extend(list(new_docs))
            
            selected_count = len(st.session_state.selected_document_ids)
            total_count = len(documents)
            
            all_selected = selected_count == total_count
            select_all = st.checkbox(
                f"🔍 Files ({selected_count}/{total_count})",
                value=all_selected,
                key="select_all_files"
            )
            
            if select_all:
                st.session_state.selected_document_ids = [doc['id'] for doc in documents]
            elif all_selected and not select_all:
                st.session_state.selected_document_ids = []
            
            with st.expander(f"Select ({selected_count}/{total_count})", expanded=False):
                for doc in documents[:15]:
                    doc_id = doc['id']
                    is_selected = doc_id in st.session_state.selected_document_ids
                    checkbox_key = f"file_checkbox_{doc_id}"
                    filename = doc['filename']
                    if len(filename) > 25:
                        filename = filename[:22] + "..."
                    
                    # Show purge status
                    days_remaining = doc.get('days_remaining', 7)
                    will_purge_soon = doc.get('will_purge_soon', False)
                    
                    if days_remaining > 0:
                        if will_purge_soon:
                            purge_status = f"⚠️ {days_remaining}d left"
                        else:
                            purge_status = f"⏳ {days_remaining}d left"
                    else:
                        purge_status = "🗑️ Purging soon"
                    
                    label = f"{filename} ({purge_status})"
                    
                    checked = st.checkbox(
                        label,
                        value=is_selected,
                        key=checkbox_key
                    )
                    
                    if checked and doc_id not in st.session_state.selected_document_ids:
                        st.session_state.selected_document_ids.append(doc_id)
                    elif not checked and doc_id in st.session_state.selected_document_ids:
                        st.session_state.selected_document_ids.remove(doc_id)
                
                if len(documents) > 15:
                    st.caption(f"+{len(documents) - 15} more")
            
            if selected_count == 0:
                st.warning("⚠️ No files")
        else:
            st.caption("📭 No docs")
            st.session_state.selected_document_ids = []
    else:
        st.session_state.selected_document_ids = []
    
    # Upload - Compact
    uploaded_file = st.file_uploader(
        "📤 Upload",
        type=['pdf', 'txt', 'docx', 'md', 'markdown', 'csv', 'json', 'html', 'htm', 'rtf', 'xlsx', 'xls', 'pptx'],
        help="Upload documents",
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Upload", use_container_width=True):
            with st.spinner("..."):
                result = upload_document(uploaded_file, st.session_state.user_id)
                if result:
                    st.success("✅ Uploaded!")
                    st.info("⏳ Auto-purge in 7 days")
                    st.session_state.document_uploaded = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌")
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
        if clear_documents(st.session_state.user_id):
            st.success("✅")
            st.session_state.document_uploaded = False
            st.session_state.messages = []
            st.rerun()
    
    # Auto-purge old data (1 week)
    with st.expander("🧹 Data Management", expanded=False):
        st.caption("Auto-purge: Data older than 7 days is automatically deleted")
        if st.button("🗑️ Purge Old Data (7 days)", use_container_width=True, key="purge_old"):
            result = purge_old_data(st.session_state.user_id, days=7)
            if result:
                st.success(f"✅ Purged: {result.get('documents_deleted', 0)} docs, {result.get('queries_deleted', 0)} queries")
                st.rerun()
            else:
                st.error("❌ Purge failed")
    
    # Settings - Collapsible
    with st.expander("⚙️ Settings & Info", expanded=False):
        st.markdown(f"**User ID:** `{st.session_state.user_id}`")
        if check_api_health():
            try:
                stats = get_user_stats(st.session_state.user_id)
                if stats:
                    st.markdown("**📊 Statistics:**")
                    st.markdown(f"- Documents: {stats.get('document_count', 0)}")
                    st.markdown(f"- Queries: {stats.get('query_count', 0)}")
                    st.markdown(f"- Characters: {stats.get('total_characters', 0):,}")
            except:
                pass
        if not check_api_health():
            st.info("💡 **Start backend:**\n```bash\npython3 -m uvicorn backend:app --reload\n```")
        st.markdown("""
        **📖 Quick Guide:**
        1. Upload files (PDF, TXT, DOCX, etc.)
        2. Select files to search
        3. Ask questions
        4. Get AI-powered answers!
        """)

# Main Chat Interface - Classy Design
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: rgba(255, 255, 255, 0.7); border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);'>
        <h2 style='margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;'>
            💬 Ask Questions About Your Documents
        </h2>
        <p style='margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.95rem;'>
            Get intelligent answers powered by advanced RAG technology
        </p>
    </div>
""", unsafe_allow_html=True)

# Display chat messages - Enhanced
if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: rgba(255, 255, 255, 0.5); border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.2); margin: 2rem 0;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <h3 style='color: #667eea; margin: 0.5rem 0;'>Ready to Answer Your Questions</h3>
            <p style='color: #64748b; margin: 0.5rem 0;'>Upload documents and start asking questions to get AI-powered answers!</p>
        </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Enhanced message display
        if message["role"] == "user":
            st.markdown(f"""
                <div style='padding: 0.5rem 0;'>
                    <strong style='color: #667eea; font-size: 0.9rem;'>You asked:</strong>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            num_sources = message.get("num_sources", len(message["sources"]))
            with st.expander(f"📚 View Sources ({num_sources})", expanded=False):
                st.markdown("""
                    <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 3px solid #667eea;'>
                        <strong style='color: #667eea;'>📄 Document Sources</strong>
                    </div>
                """, unsafe_allow_html=True)
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                        <div style='background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 8px; margin: 0.75rem 0; border-left: 3px solid #10b981;'>
                            <strong style='color: #10b981;'>Source {i}:</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"```\n{source}\n```")
                    if i < len(message["sources"]):
                        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid rgba(102, 126, 234, 0.1);'>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if not check_api_health():
        st.error("Please start the backend server first!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response - Enhanced Display
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your documents..."):
            # Get selected document_ids from session state
            selected_ids = st.session_state.get("selected_document_ids", [])
            # If no files selected, don't query
            if not selected_ids:
                st.error("⚠️ Please select at least one file to search in the sidebar.")
                st.stop()
            # Pass list of document IDs (or None for all files)
            response = query_documents(prompt, st.session_state.user_id, document_ids=selected_ids)
            
            if response:
                answer = response.get("answer", "No answer available")
                sources = response.get("sources", [])
                # Ensure num_sources is always an integer (default to len(sources) if not provided)
                num_sources = response.get("num_sources")
                if num_sources is None:
                    num_sources = len(sources) if sources else 0
                else:
                    num_sources = int(num_sources)
                confidence = response.get("confidence", None)
                enhanced = response.get("enhanced", False)
                
                # Classy answer display
                st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); 
                                padding: 1rem; border-radius: 12px; margin-bottom: 1rem; 
                                border-left: 4px solid #10b981;'>
                        <strong style='color: #10b981; font-size: 1.1rem;'>💡 Answer</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display confidence and enhanced status
                if confidence is not None or enhanced:
                    col1, col2 = st.columns(2)
                    with col1:
                        if confidence is not None:
                            confidence_color = "#10b981" if confidence >= 0.7 else "#f59e0b" if confidence >= 0.4 else "#ef4444"
                            st.markdown(f"""
                                <div style='display: inline-block; background: {confidence_color}20; 
                                            color: {confidence_color}; padding: 0.5rem 1rem; border-radius: 20px; 
                                            font-weight: 600; font-size: 0.9rem; border: 1px solid {confidence_color}40;'>
                                    🎯 Confidence: {confidence:.0%}
                                </div>
                            """, unsafe_allow_html=True)
                    with col2:
                        if enhanced:
                            st.markdown("""
                                <div style='display: inline-block; background: rgba(102, 126, 234, 0.1); 
                                            color: #667eea; padding: 0.5rem 1rem; border-radius: 20px; 
                                            font-weight: 600; font-size: 0.9rem; border: 1px solid rgba(102, 126, 234, 0.3);'>
                                    ✨ Enhanced Search
                                </div>
                            """, unsafe_allow_html=True)
                
                # Enhanced answer text
                st.markdown(f"""
                    <div style='line-height: 1.8; color: #1f2937; font-size: 1.05rem; padding: 0.5rem 0;'>
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                
                # Source count badge
                if num_sources and num_sources > 0:
                    st.markdown(f"""
                        <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                                    margin: 1rem 0; font-weight: 600; font-size: 0.9rem;'>
                            📊 Found {num_sources} relevant source{'' if num_sources == 1 else 's'}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced source display
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)})", expanded=False):
                        st.markdown("""
                            <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 3px solid #667eea;'>
                                <strong style='color: #667eea;'>📄 Document Sources</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 8px; margin: 0.75rem 0; border-left: 3px solid #10b981;'>
                                    <strong style='color: #10b981;'>Source {i}:</strong>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"```\n{source}\n```")
                            if i < len(sources):
                                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid rgba(102, 126, 234, 0.1);'>", unsafe_allow_html=True)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "num_sources": num_sources
                })
            else:
                error_msg = "Sorry, I couldn't process your question. Please try again."
                st.markdown(f"""
                    <div style='background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 12px; 
                                border-left: 4px solid #ef4444; color: #dc2626;'>
                        <strong>❌ Error:</strong> {error_msg}
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer - Clean
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p style='font-size: 0.9rem; margin: 0.5rem 0;'>
            Built with ❤️ using Streamlit, FastAPI, LangChain, ChromaDB
        </p>
        <p style='font-size: 0.85rem; color: #94a3b8; margin: 0;'>
            🎯 Semantic Chunking • 🔍 BGE Embeddings • 💾 SQLite Database
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

