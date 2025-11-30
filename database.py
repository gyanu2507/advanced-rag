"""
Database models and session management for the document Q&A system.
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta
from typing import Optional, List
import os

# Database file path
DATABASE_URL = "sqlite:///./documents.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    phone = Column(String, unique=True, index=True, nullable=True)
    auth_type = Column(String, nullable=True)  # 'email', 'phone', 'google', 'anonymous'
    google_id = Column(String, unique=True, index=True, nullable=True)
    is_verified = Column(String, default="false")  # 'true', 'false'
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}', phone='{self.phone}')>"


class Document(Base):
    """Document model for storing document metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, txt, docx
    file_size = Column(Integer, nullable=False)  # in bytes
    character_count = Column(Integer, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="processed")  # processed, error, pending
    
    def __repr__(self):
        return f"<Document(filename='{self.filename}', user_id='{self.user_id}')>"


class QueryHistory(Base):
    """Query history model for storing user queries and responses."""
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    query_date = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Float, nullable=True)  # in seconds
    
    def __repr__(self):
        return f"<QueryHistory(user_id='{self.user_id}', question='{self.question[:50]}...')>"


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized")


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let the caller manage it


def get_or_create_user(db: Session, user_id: str) -> User:
    """Get or create a user."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update last active
        user.last_active = datetime.utcnow()
        db.commit()
    return user


def create_document_record(
    db: Session,
    user_id: str,
    filename: str,
    file_type: str,
    file_size: int,
    character_count: int
) -> Document:
    """Create a document record in the database."""
    doc = Document(
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        character_count=character_count
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def create_query_record(
    db: Session,
    user_id: str,
    question: str,
    answer: str,
    response_time: Optional[float] = None
) -> QueryHistory:
    """Create a query history record."""
    query = QueryHistory(
        user_id=user_id,
        question=question,
        answer=answer,
        response_time=response_time
    )
    db.add(query)
    db.commit()
    db.refresh(query)
    return query


def get_user_documents(db: Session, user_id: str) -> List[Document]:
    """Get all documents for a user."""
    return db.query(Document).filter(Document.user_id == user_id).order_by(Document.upload_date.desc()).all()


def get_user_query_history(db: Session, user_id: str, limit: int = 50) -> List[QueryHistory]:
    """Get query history for a user."""
    return db.query(QueryHistory).filter(QueryHistory.user_id == user_id).order_by(QueryHistory.query_date.desc()).limit(limit).all()


def delete_user_documents(db: Session, user_id: str) -> int:
    """Delete all documents for a user."""
    count = db.query(Document).filter(Document.user_id == user_id).delete()
    db.commit()
    return count


def delete_user_queries(db: Session, user_id: str) -> int:
    """Delete all query history for a user."""
    count = db.query(QueryHistory).filter(QueryHistory.user_id == user_id).delete()
    db.commit()
    return count


def get_user_stats(db: Session, user_id: str) -> dict:
    """Get statistics for a user."""
    from sqlalchemy import func
    doc_count = db.query(Document).filter(Document.user_id == user_id).count()
    query_count = db.query(QueryHistory).filter(QueryHistory.user_id == user_id).count()
    total_chars = db.query(func.sum(Document.character_count)).filter(Document.user_id == user_id).scalar() or 0
    
    return {
        "user_id": user_id,
        "document_count": doc_count,
        "query_count": query_count,
        "total_characters": total_chars
    }

def purge_old_data(db: Session, user_id: str, days: int = 7) -> dict:
    """Purge data older than specified days for a user.
    
    Args:
        db: Database session
        user_id: User ID to purge data for
        days: Number of days (default 7 for 1 week)
    
    Returns:
        Dictionary with counts of deleted items
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Count old documents before deletion
    old_docs = db.query(Document).filter(
        Document.user_id == user_id,
        Document.upload_date < cutoff_date
    ).all()
    doc_count = len(old_docs)
    
    # Count old queries before deletion
    old_queries = db.query(QueryHistory).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.query_date < cutoff_date
    ).all()
    query_count = len(old_queries)
    
    # Delete from database
    db.query(Document).filter(
        Document.user_id == user_id,
        Document.upload_date < cutoff_date
    ).delete()
    
    db.query(QueryHistory).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.query_date < cutoff_date
    ).delete()
    
    db.commit()
    
    return {
        "documents_deleted": doc_count,
        "queries_deleted": query_count,
        "cutoff_date": cutoff_date.isoformat()
    }

def purge_all_old_data(db: Session, days: int = 7) -> dict:
    """Purge old data for all users.
    
    Args:
        db: Database session
        days: Number of days (default 7 for 1 week)
    
    Returns:
        Dictionary with counts of deleted items
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Delete old documents for all users
    doc_count = db.query(Document).filter(
        Document.upload_date < cutoff_date
    ).delete()
    
    # Delete old queries for all users
    query_count = db.query(QueryHistory).filter(
        QueryHistory.query_date < cutoff_date
    ).delete()
    
    db.commit()
    
    return {
        "documents_deleted": doc_count,
        "queries_deleted": query_count,
        "cutoff_date": cutoff_date.isoformat()
    }

