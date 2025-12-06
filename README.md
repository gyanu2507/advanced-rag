# Advanced RAG - AI Document Q&A System 🚀

A powerful RAG (Retrieval Augmented Generation) system with state-of-the-art features for document question-answering.

**✨ 100% FREE - No paid API keys required!**

## 🌟 Features

### Core
- 📄 **Multi-Format Support**: PDF, TXT, DOCX, Markdown, CSV, JSON, HTML, RTF, Excel, PowerPoint
- 🔍 **Advanced RAG**: Semantic search with BGE embeddings
- 🔒 **User Isolation**: Each user has isolated document storage
- 🧹 **Auto-Purge**: Automatic data cleanup after 7 days

### 🚀 v2.0 Enhancements
- **Cross-Encoder Reranking**: 10x better relevance than keyword matching
- **RRF (Reciprocal Rank Fusion)**: Optimal multi-retrieval combination  
- **HyDE**: Hypothetical Document Embeddings for better search
- **RAGAS Evaluation**: Real-time quality metrics
- **Hallucination Detection**: Identifies ungrounded claims

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Streamlit + React/Vite |
| Vector DB | ChromaDB |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (new terminal)
streamlit run app.py --server.port 8501
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload document |
| `/query` | POST | Query documents |
| `/query/enhanced` | POST | **v2.0** Query with RRF, HyDE, evaluation |
| `/users/{id}/documents` | GET | List user documents |
| `/users/{id}/stats` | GET | User statistics |

### Enhanced Query Example

```bash
curl -X POST http://localhost:8000/query/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What causes pollution?",
    "use_hyde": true,
    "use_rrf": true,
    "include_evaluation": true
  }'
```

**Response includes:**
- `answer`: Generated response
- `confidence`: 0.0-1.0 score
- `evaluation`: RAGAS metrics (context_relevancy, answer_faithfulness, answer_relevancy)
- `enhanced_features`: Features used (RRF, CrossEncoder, HyDE, etc.)

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend
├── backend.py              # FastAPI backend
├── rag_system.py           # RAG implementation (v2.0)
├── retrieval_strategies.py # RRF, HyDE, compression
├── advanced_chunking.py    # Chunking strategies
├── document_processor.py   # File processing
├── database.py             # SQLAlchemy models
├── auth.py                 # Authentication
├── frontend/               # React frontend
└── requirements.txt        # Dependencies
```

## 📝 License

MIT License

---
**Built with ❤️ using open-source technologies**
