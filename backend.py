"""
FastAPI backend for the document Q&A system.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import time
from sqlalchemy.orm import Session
from rag_system import RAGSystem
from document_processor import DocumentProcessor
from database import (
    init_db, get_db, get_or_create_user, create_document_record,
    create_query_record, get_user_documents, get_user_query_history,
    delete_user_documents, delete_user_queries, get_user_stats,
    purge_old_data, purge_all_old_data,
    Document, QueryHistory, User
)
from auth import (
    authenticate_with_google, verify_jwt_token, initiate_phone_verification,
    verify_phone_otp, generate_jwt_token
)

app = FastAPI(title="AI Document Q&A API")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("✓ Database ready")
    # Auto-purge old data on startup
    try:
        db = next(get_db())
        result = purge_all_old_data(db, days=7)
        print(f"✓ Auto-purged: {result['documents_deleted']} docs, {result['queries_deleted']} queries")
        db.close()
    except Exception as e:
        print(f"⚠️ Auto-purge error: {e}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store RAG systems per user
user_rag_systems: dict[str, RAGSystem] = {}
processor = DocumentProcessor()

def get_rag_system(user_id: str = "default") -> RAGSystem:
    """Get or initialize the RAG system for a specific user."""
    if user_id not in user_rag_systems:
        try:
            # Create user-specific directory for vectorstore
            persist_dir = f"./chroma_db/user_{user_id}"
            user_rag_systems[user_id] = RAGSystem(persist_directory=persist_dir)
            print(f"Initialized RAG system for user: {user_id}")
        except Exception as e:
            print(f"Error initializing RAG system for user {user_id}: {e}")
            raise
    return user_rag_systems[user_id]


class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = "default"
    document_id: Optional[int] = None  # Legacy: single document filter
    document_ids: Optional[List[int]] = None  # Filter by multiple documents


class QueryResponse(BaseModel):
    answer: str
    sources: list
    num_sources: Optional[int] = None
    confidence: Optional[float] = None
    enhanced: Optional[bool] = False


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
    db: Session = Depends(get_db)
):
    """Upload and process a document for a specific user."""
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx', '.md', '.markdown', '.csv', '.json', '.html', '.htm', '.rtf', '.xlsx', '.xls', '.pptx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
        file_size = len(content)
    
    try:
        # Process document
        text = processor.process_file(tmp_file_path)
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Document appears to be empty or could not be processed"
            )
        
        # Get or create user in database
        user = get_or_create_user(db, user_id)
        
        # Save document metadata to database first to get document_id
        doc_record = create_document_record(
            db=db,
            user_id=user_id,
            filename=file.filename,
            file_type=file_ext[1:],  # Remove the dot
            file_size=file_size,
            character_count=len(text)
        )
        
        # Add to user-specific RAG system with document_id in metadata
        rag_system = get_rag_system(user_id)
        rag_system.add_documents(
            texts=[text],
            metadatas=[{"filename": file.filename, "user_id": user_id, "document_id": str(doc_record.id)}]
        )
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully for user {user_id}",
            "characters": len(text),
            "user_id": user_id,
            "document_id": doc_record.id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query documents with a question for a specific user."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    try:
        user_id = request.user_id or "default"
        
        # Get or create user
        get_or_create_user(db, user_id)
        
        # Query RAG system with optional document filter and enhanced features
        rag_system = get_rag_system(user_id)
        # Support both single document_id (legacy) and multiple document_ids
        if request.document_ids:
            document_ids = [str(doc_id) for doc_id in request.document_ids]
        elif request.document_id:
            document_ids = [str(request.document_id)]
        else:
            document_ids = None
        result = rag_system.query(
            request.question, 
            document_ids=document_ids,
            user_id=user_id,
            use_enhancements=True  # Enable all enhancements
        )
        
        response_time = time.time() - start_time
        
        # Save query to database
        create_query_record(
            db=db,
            user_id=user_id,
            question=request.question,
            answer=result["answer"],
            response_time=response_time
        )
        
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_documents(
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """Clear all documents for a specific user."""
    try:
        # Clear from RAG system
        if user_id in user_rag_systems:
            user_rag_systems[user_id].clear_documents(user_id=user_id)
            del user_rag_systems[user_id]
        else:
            # Clear the directory even if not in memory
            import shutil
            persist_dir = f"./chroma_db/user_{user_id}"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
        
        # Clear from database
        doc_count = delete_user_documents(db, user_id)
        query_count = delete_user_queries(db, user_id)
        
        return {
            "status": "success",
            "message": f"All documents cleared for user {user_id}",
            "documents_deleted": doc_count,
            "queries_deleted": query_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New database endpoints
@app.get("/users/{user_id}/documents")
async def get_documents(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get all documents for a user with purge status."""
    from datetime import datetime, timedelta
    documents = get_user_documents(db, user_id)
    
    # Calculate days remaining for purge (7 days from upload)
    purge_days = 7
    now = datetime.utcnow()
    
    result_docs = []
    for doc in documents:
        upload_date = doc.upload_date
        if isinstance(upload_date, str):
            try:
                upload_date = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
            except:
                upload_date = datetime.utcnow()
        
        # Handle timezone-aware datetime
        if upload_date.tzinfo is not None:
            upload_date = upload_date.replace(tzinfo=None)
        
        days_since_upload = (now - upload_date).days
        days_remaining = max(0, purge_days - days_since_upload)
        
        result_docs.append({
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "character_count": doc.character_count,
            "upload_date": doc.upload_date.isoformat() if hasattr(doc.upload_date, 'isoformat') else str(doc.upload_date),
            "status": doc.status,
            "days_remaining": days_remaining,
            "days_since_upload": days_since_upload,
            "will_purge_soon": days_remaining <= 2
        })
    
    return {
        "user_id": user_id,
        "count": len(documents),
        "documents": result_docs
    }


@app.get("/users/{user_id}/queries")
async def get_queries(
    user_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get query history for a user."""
    queries = get_user_query_history(db, user_id, limit)
    return {
        "user_id": user_id,
        "count": len(queries),
        "queries": [
            {
                "id": q.id,
                "question": q.question,
                "answer": q.answer[:200] + "..." if len(q.answer) > 200 else q.answer,
                "query_date": q.query_date.isoformat(),
                "response_time": q.response_time
            }
            for q in queries
        ]
    }


@app.get("/users/{user_id}/stats")
async def get_stats(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get statistics for a user."""
    return get_user_stats(db, user_id)


@app.get("/db/stats")
async def get_db_stats(db: Session = Depends(get_db)):
    """Get overall database statistics."""
    from sqlalchemy import func
    from database import Document, QueryHistory, User
    
    user_count = db.query(User).count()
    doc_count = db.query(Document).count()
    query_count = db.query(QueryHistory).count()
    total_chars = db.query(func.sum(Document.character_count)).scalar() or 0
    avg_response = db.query(func.avg(QueryHistory.response_time)).filter(
        QueryHistory.response_time.isnot(None)
    ).scalar() or 0
    
    return {
        "total_users": user_count,
        "total_documents": doc_count,
        "total_queries": query_count,
        "total_characters": total_chars,
        "average_response_time": round(avg_response, 2) if avg_response else 0
    }

@app.post("/purge/{user_id}")
async def purge_user_data(
    user_id: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Purge data older than specified days for a user."""
    try:
        result = purge_old_data(db, user_id, days)
        return {
            "status": "success",
            "message": f"Purged data older than {days} days for user {user_id}",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/purge/all")
async def purge_all_data(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Purge old data for all users."""
    try:
        result = purge_all_old_data(db, days)
        return {
            "status": "success",
            "message": f"Purged data older than {days} days for all users",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Authentication Endpoints

class GoogleAuthRequest(BaseModel):
    token: str


class PhoneVerificationRequest(BaseModel):
    phone: str


class PhoneOTPVerifyRequest(BaseModel):
    phone: str
    code: str


@app.post("/auth/google")
async def google_auth(
    request: GoogleAuthRequest,
    db: Session = Depends(get_db)
):
    """Authenticate with Google OAuth token."""
    try:
        result = authenticate_with_google(request.token, db)
        if result:
            return result
        else:
            raise HTTPException(status_code=401, detail="Invalid Google token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/phone/send-otp")
async def send_phone_otp(
    request: PhoneVerificationRequest,
    db: Session = Depends(get_db)
):
    """Send OTP to phone number."""
    try:
        result = initiate_phone_verification(request.phone)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/phone/verify")
async def verify_phone(
    request: PhoneOTPVerifyRequest,
    db: Session = Depends(get_db)
):
    """Verify phone OTP and authenticate."""
    try:
        result = verify_phone_otp(request.phone, request.code, db)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/verify-token")
async def verify_token(token: str = Form(...)):
    """Verify JWT token and return user info."""
    payload = verify_jwt_token(token)
    if payload:
        return {
            "status": "valid",
            "user_id": payload.get("user_id"),
            "email": payload.get("email")
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@app.get("/auth/user/{user_id}")
async def get_auth_user(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get authenticated user information."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user.user_id,
        "email": user.email,
        "phone": user.phone,
        "auth_type": user.auth_type,
        "is_verified": user.is_verified,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_active": user.last_active.isoformat() if user.last_active else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

