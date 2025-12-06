"""
FastAPI backend for the document Q&A system.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import time
import asyncio
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
    verify_phone_otp, generate_jwt_token, create_user_with_email_password,
    authenticate_with_email_password
)

app = FastAPI(title="AI Document Q&A API")

# Periodic auto-purge (runs every hour)
import asyncio

async def periodic_purge():
    """Run auto-purge every hour."""
    while True:
        await asyncio.sleep(3600)  # Wait 1 hour
        try:
            db = next(get_db())
            result = purge_all_old_data(db, days=7)
            if result['documents_deleted'] > 0 or result['queries_deleted'] > 0:
                print(f"🔄 Periodic purge: {result['documents_deleted']} docs, {result['queries_deleted']} queries deleted")
            db.close()
        except Exception as e:
            print(f"⚠️ Periodic purge error: {e}")

async def periodic_purge():
    """Run auto-purge every hour in the background."""
    while True:
        await asyncio.sleep(3600)  # Wait 1 hour
        try:
            db = next(get_db())
            result = purge_all_old_data(db, days=7)
            if result['documents_deleted'] > 0 or result['queries_deleted'] > 0:
                print(f"🔄 Periodic purge: {result['documents_deleted']} docs, {result['queries_deleted']} queries deleted")
            db.close()
        except Exception as e:
            print(f"⚠️ Periodic purge error: {e}")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("✓ Database ready")
    # Auto-purge old data on startup (runs automatically)
    try:
        db = next(get_db())
        result = purge_all_old_data(db, days=7)
        if result['documents_deleted'] > 0 or result['queries_deleted'] > 0:
            print(f"✓ Auto-purged: {result['documents_deleted']} docs, {result['queries_deleted']} queries (older than 7 days)")
        else:
            print("✓ Auto-purge: No old data to clean")
        db.close()
    except Exception as e:
        print(f"⚠️ Auto-purge error: {e}")
    
    # Start periodic purge task (runs every hour automatically)
    asyncio.create_task(periodic_purge())
    print("✓ Periodic auto-purge started (runs every hour automatically)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
    ],
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


# 🚀 GREATEST RAG EVER - v2.0 Enhanced Models
class EnhancedQueryRequest(BaseModel):
    """Request model for enhanced RAG query with all v2.0 features."""
    question: str
    user_id: Optional[str] = "default"
    document_ids: Optional[List[int]] = None
    use_hyde: bool = True       # Hypothetical Document Embeddings
    use_rrf: bool = True        # Reciprocal Rank Fusion
    use_compression: bool = False  # Contextual compression
    include_evaluation: bool = True  # RAGAS-style metrics


class EvaluationMetrics(BaseModel):
    """RAGAS-inspired evaluation metrics."""
    context_relevancy: float
    answer_faithfulness: float
    answer_relevancy: float
    overall_score: float


class EnhancedQueryResponse(BaseModel):
    """Enhanced response with evaluation metrics and feature tracking."""
    answer: str
    sources: list
    num_sources: Optional[int] = None
    confidence: float = 0.0
    evaluation: Optional[EvaluationMetrics] = None
    enhanced_features: List[str] = []


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
    print(f"📤 Upload request: filename={file.filename}, user_id={user_id}")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx', '.md', '.markdown', '.csv', '.json', '.html', '.htm', '.rtf', '.xlsx', '.xls', '.pptx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    print(f"   File extension: {file_ext}")
    
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
        print(f"   Processing file: {tmp_file_path}")
        # Process document
        text = processor.process_file(tmp_file_path)
        print(f"   Extracted {len(text)} characters")
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Document appears to be empty or could not be processed"
            )
        
        # Get or create user in database
        print(f"   Getting/creating user: {user_id}")
        user = get_or_create_user(db, user_id)
        
        # Save document metadata to database first to get document_id
        print(f"   Creating document record...")
        doc_record = create_document_record(
            db=db,
            user_id=user_id,
            filename=file.filename,
            file_type=file_ext[1:],  # Remove the dot
            file_size=file_size,
            character_count=len(text)
        )
        
        # Add to user-specific RAG system with document_id in metadata
        print(f"   Getting RAG system for user: {user_id}")
        rag_system = get_rag_system(user_id)
        print(f"   Adding documents to RAG system...")
        rag_system.add_documents(
            texts=[text],
            metadatas=[{"filename": file.filename, "user_id": user_id, "document_id": str(doc_record.id)}]
        )
        
        print(f"✅ Upload successful: document_id={doc_record.id}")
        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully for user {user_id}",
            "characters": len(text),
            "user_id": user_id,
            "document_id": doc_record.id
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Upload error: {str(e)}")
        print(f"📋 Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temp file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not delete temp file {tmp_file_path}: {cleanup_error}")


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


# 🚀 GREATEST RAG EVER - v2.0 Enhanced Query Endpoint
@app.post("/query/enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query_documents(
    request: EnhancedQueryRequest,
    db: Session = Depends(get_db)
):
    """
    Enhanced query with all v2.0 features:
    - HyDE (Hypothetical Document Embeddings)
    - RRF (Reciprocal Rank Fusion)
    - Cross-encoder reranking
    - Contextual compression (optional)
    - RAGAS-style evaluation metrics
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    try:
        user_id = request.user_id or "default"
        get_or_create_user(db, user_id)
        
        rag_system = get_rag_system(user_id)
        
        # Convert document IDs
        document_ids = [str(doc_id) for doc_id in request.document_ids] if request.document_ids else None
        
        # Use enhanced query method
        result = rag_system.enhanced_query(
            question=request.question,
            document_ids=document_ids,
            user_id=user_id,
            use_hyde=request.use_hyde,
            use_rrf=request.use_rrf,
            use_compression=request.use_compression,
            include_evaluation=request.include_evaluation
        )
        
        response_time = time.time() - start_time
        
        # Save to database
        create_query_record(
            db=db,
            user_id=user_id,
            question=request.question,
            answer=result["answer"],
            response_time=response_time
        )
        
        # Build response
        evaluation = None
        if result.get("evaluation"):
            evaluation = EvaluationMetrics(**result["evaluation"])
        
        return EnhancedQueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            num_sources=result.get("num_sources", len(result.get("sources", []))),
            confidence=result.get("confidence", 0.0),
            evaluation=evaluation,
            enhanced_features=result.get("enhanced_features", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}/documents/{document_id}")
async def delete_document(
    user_id: str,
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a specific document for a user."""
    try:
        print(f"🗑️ Delete request: user_id={user_id}, document_id={document_id}")
        
        # Get the document
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from RAG system (filter by document_id in metadata)
        rag_system = get_rag_system(user_id)
        try:
            # Delete document chunks from vectorstore by document_id metadata
            rag_system.delete_documents_by_metadata({"document_id": str(document_id)})
            print(f"✓ Removed document chunks from vectorstore")
        except Exception as e:
            print(f"⚠️ Warning: Could not remove from vectorstore: {e}")
            # Continue with database deletion even if vectorstore deletion fails
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        print(f"✅ Document deleted: {document.filename}")
        return {
            "status": "success",
            "message": f"Document '{document.filename}' deleted successfully",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
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


@app.post("/users/{user_id}/purge")
async def purge_user_data(
    user_id: str,
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Manually purge old data for a user."""
    try:
        print(f"🧹 Purging old data for user {user_id}, older than {days} days")
        result = purge_old_data(db, user_id, days)
        print(f"✅ Purged: {result.get('documents_deleted', 0)} docs, {result.get('queries_deleted', 0)} queries")
        return {
            "status": "success",
            "message": f"Purged data older than {days} days",
            **result
        }
    except Exception as e:
        print(f"❌ Purge error: {e}")
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


class EmailPasswordRequest(BaseModel):
    email: str
    password: str


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


@app.get("/auth/google/callback")
async def google_oauth_callback(
    code: Optional[str] = None,
    error: Optional[str] = None
):
    """Handle Google OAuth callback."""
    if error:
        return {"status": "error", "message": f"OAuth error: {error}"}
    
    if not code:
        return {"status": "error", "message": "No authorization code received"}
    
    # Exchange code for token
    try:
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/auth/callback"),
            "grant_type": "authorization_code"
        }
        
        response = requests.post(token_url, data=token_data, timeout=10)
        if response.status_code != 200:
            return {"status": "error", "message": "Failed to exchange code for token"}
        
        token_info = response.json()
        access_token = token_info.get("access_token")
        
        if not access_token:
            return {"status": "error", "message": "No access token received"}
        
        # Get user info
        user_info_response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )
        
        if user_info_response.status_code != 200:
            return {"status": "error", "message": "Failed to get user info"}
        
        user_info = user_info_response.json()
        
        return {
            "status": "success",
            "access_token": access_token,
            "user_info": user_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/auth/google/config")
async def get_google_config():
    """Get Google OAuth configuration for frontend."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/auth/callback")
    
    return {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "enabled": bool(client_id and client_secret),
        "configured": bool(client_id)
    }


@app.post("/auth/email/signup")
async def signup_with_email(
    request: EmailPasswordRequest,
    db: Session = Depends(get_db)
):
    """Sign up with email and password."""
    try:
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, request.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Validate password strength
        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
        user = create_user_with_email_password(db, request.email, request.password)
        if not user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate JWT token
        jwt_token = generate_jwt_token(user.user_id, user.email)
        
        return {
            "status": "success",
            "message": "Account created successfully",
            "token": jwt_token,
            "user_id": user.user_id,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "is_verified": user.is_verified
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/email/login")
async def login_with_email(
    request: EmailPasswordRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password."""
    try:
        result = authenticate_with_email_password(db, request.email, request.password)
        if result:
            return result
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

