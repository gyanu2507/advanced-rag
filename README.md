# Advanced RAG - AI Document Q&A System 🚀

A powerful, production-ready RAG (Retrieval Augmented Generation) powered document question-answering system with multi-format support, user isolation, automatic data purging, and a beautiful modern UI.

**✨ 100% FREE - No paid API keys required!**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### Core Functionality
- 📄 **Multi-Format Support**: PDF, TXT, DOCX, Markdown, CSV, JSON, HTML, RTF, Excel, PowerPoint
- 🔍 **Advanced RAG**: Semantic search with BGE embeddings and intelligent chunking
- 💬 **AI-Powered Q&A**: Get intelligent answers from your documents
- 🔒 **User Isolation**: Each user has isolated document storage and queries
- 💾 **Database Tracking**: SQLite database for users, documents, and query history
- 🧹 **Auto-Purge**: Automatic data purging after 7 days (configurable)

### Advanced Features
- 🎯 **Smart Chunking**: Multiple chunking strategies (smart, semantic, sentence-aware, paragraph)
- 📊 **File Selection**: Select multiple files to search simultaneously
- 📈 **Statistics Dashboard**: Track documents, queries, and characters per user
- 🎨 **Modern UI**: Beautiful, responsive interface with glass morphism effects
- ⚡ **Lazy Loading**: AI models load only when needed
- 🔄 **Fallback System**: Graceful degradation if LLM fails

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gyanu2507/advanced-rag.git
   cd advanced-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend:**
   ```bash
   python3 -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Start the frontend (in a new terminal):**
   ```bash
   streamlit run app.py --server.port 8501
   ```

5. **Open your browser:**
   ```
   http://localhost:8501
   ```

### Using the Startup Script

```bash
chmod +x run.sh
./run.sh
```

## 📋 Supported File Types

| Format | Extension | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Supported |
| Text | `.txt` | ✅ Supported |
| Word | `.docx` | ✅ Supported |
| Markdown | `.md`, `.markdown` | ✅ Supported |
| CSV | `.csv` | ✅ Supported |
| JSON | `.json` | ✅ Supported |
| HTML | `.html`, `.htm` | ✅ Supported |
| RTF | `.rtf` | ✅ Supported |
| Excel | `.xlsx`, `.xls` | ✅ Supported |
| PowerPoint | `.pptx` | ✅ Supported |

## 🏗️ Architecture

### Tech Stack

**Backend:**
- **FastAPI** - Modern, fast web framework
- **LangChain** - LLM application framework
- **ChromaDB** - Vector database for embeddings
- **SQLAlchemy** - Database ORM
- **Hugging Face** - Free embeddings and LLMs

**Frontend:**
- **Streamlit** - Interactive web app framework
- **Custom CSS** - Modern, responsive design

**AI Models:**
- **Embeddings**: `BAAI/bge-small-en-v1.5` (free, open-source)
- **LLM**: Hugging Face models (free tier available)

### System Architecture

```
┌─────────────┐
│   Streamlit │  Frontend (Port 8501)
│     App     │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────┐
│   FastAPI   │  Backend (Port 8000)
│   Backend   │
└──────┬──────┘
       │
       ├──► ChromaDB (Vector Store)
       ├──► SQLite (Metadata & History)
       └──► Hugging Face (AI Models)
```

## 📁 Project Structure

```
advanced-rag/
├── app.py                 # Streamlit frontend
├── backend.py             # FastAPI backend
├── rag_system.py          # RAG implementation
├── document_processor.py  # File processing
├── database.py           # Database models & operations
├── advanced_chunking.py   # Chunking strategies
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── render.yaml           # Render.com deployment
├── Procfile              # Heroku deployment
├── railway.json          # Railway deployment
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
# API Configuration
API_URL=http://localhost:8000

# Hugging Face (Optional - for faster API access)
HUGGINGFACE_API_TOKEN=your_token_here

# CORS (for deployment)
ALLOWED_ORIGINS=http://localhost:8501,https://your-app.onrender.com
```

### Database

The application uses SQLite for data persistence:
- **Database file**: `documents.db`
- **Vector store**: `chroma_db/` directory
- **Auto-purge**: Data older than 7 days is automatically deleted

## 🎯 Usage

### 1. Upload Documents

1. Click "📤 Upload" in the sidebar
2. Select a file (PDF, TXT, DOCX, etc.)
3. Click "🚀 Upload"
4. Wait for processing (first upload may take longer as models load)

### 2. Select Files to Search

1. Go to "🔍 Search Files" section
2. Use "Select All" or choose specific files
3. Files show days remaining before auto-purge

### 3. Ask Questions

1. Type your question in the chat input
2. Get AI-powered answers based on your documents
3. View sources used for the answer

### 4. Manage Data

- **User Session**: Switch between users for isolated data
- **Statistics**: View document and query counts
- **Purge**: Manually purge old data (7+ days)

## 🔐 User Management

- **User Isolation**: Each user ID has separate document storage
- **Session Management**: Switch users to access different document sets
- **Auto-Generate ID**: Click "🔄 Generate New User ID" for a new session

## 🧹 Data Purging

- **Automatic**: Data older than 7 days is purged on backend startup
- **Manual**: Use "🗑️ Purge Old Data" button in sidebar
- **Status**: Each file shows days remaining before purge
- **Warning**: Files expiring in ≤2 days show warning

## 🚀 Deployment

### 🆓 Free Deployment Options

#### Option 1: Railway (Recommended - 100% Free Tier)

**Quick Deploy:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Or use the deployment script:**
```bash
chmod +x deploy_railway.sh
./deploy_railway.sh
```

**📖 See `RAILWAY_DEPLOY.md` for complete guide!**

**Alternative: Streamlit Cloud (Frontend) + Railway (Backend)**
- Frontend: [share.streamlit.io](https://share.streamlit.io) - Connect GitHub repo
- Backend: Railway (as above)
- Set env: `API_URL=https://your-backend.railway.app`

#### Option 2: Fly.io (Free Tier)
```bash
fly auth login
fly launch
fly deploy
```

#### Option 3: Hugging Face Spaces (Free)
- Deploy Streamlit app on [Hugging Face Spaces](https://huggingface.co/spaces)
- Connect GitHub repo
- Auto-deploys!

#### Option 4: Docker (Any Platform)
```bash
docker-compose up -d
```

**📖 See `FREE_DEPLOYMENT.md` for all free deployment options!**

## 📊 API Endpoints

### Backend API (FastAPI)

- `GET /health` - Health check
- `POST /upload` - Upload document
- `POST /query` - Query documents
- `DELETE /clear` - Clear user documents
- `GET /users/{user_id}/documents` - Get user documents
- `GET /users/{user_id}/queries` - Get query history
- `GET /users/{user_id}/stats` - Get user statistics
- `POST /purge/{user_id}` - Purge old data for user
- `POST /purge/all` - Purge old data for all users

## 🛠️ Development

### Running Locally

```bash
# Backend
python3 -m uvicorn backend:app --reload

# Frontend
streamlit run app.py
```

### Database Management

View database contents:
```bash
python3 view_database.py
```

## 📦 Dependencies

Key dependencies:
- `fastapi` - Web framework
- `streamlit` - Frontend framework
- `langchain` - LLM framework
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `transformers` - AI models
- `sqlalchemy` - Database ORM
- `pypdf2` - PDF processing
- `python-docx` - Word document processing
- `beautifulsoup4` - HTML parsing
- `openpyxl` - Excel processing
- `python-pptx` - PowerPoint processing

See `requirements.txt` for complete list.

## 🎨 Features in Detail

### Advanced Chunking
- **Smart Chunking**: Intelligent text splitting with context preservation
- **Semantic Chunking**: Chunks based on semantic similarity
- **Sentence-Aware**: Respects sentence boundaries
- **Paragraph Chunking**: Splits by paragraphs

### User Isolation
- Separate ChromaDB vector stores per user
- Isolated document storage
- Independent query history
- Per-user statistics

### Modern UI
- Glass morphism effects
- Animated gradients
- Responsive design
- Clean, minimal interface
- Real-time status indicators

## 🐛 Troubleshooting

### Backend Not Connecting
- Check if backend is running: `curl http://localhost:8000/health`
- Verify `API_URL` in frontend
- Check firewall/port settings

### Models Not Loading
- First upload may take 2-5 minutes (model download)
- Ensure sufficient memory (2GB+ recommended)
- Check internet connection for model downloads

### File Upload Fails
- Check file size limits
- Verify file format is supported
- Check backend logs for errors

### Database Errors
- Ensure write permissions for `documents.db`
- Check disk space
- Verify SQLite is installed

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- **LangChain** - LLM application framework
- **Hugging Face** - Free AI models and embeddings
- **ChromaDB** - Vector database
- **FastAPI** - Modern web framework
- **Streamlit** - Interactive app framework

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using open-source technologies**
