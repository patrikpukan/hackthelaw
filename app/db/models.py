from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, 
    ForeignKey, Boolean, BigInteger, UUID
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
import uuid
from datetime import datetime

Base = declarative_base()


class Document(Base):
    """Document table for storing uploaded legal documents."""

    __tablename__ = "documents"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    upload_date = Column(DateTime, default=func.now())
    file_size = Column(BigInteger, nullable=False)
    extracted_text = Column(Text)
    document_hash = Column(String(64), unique=True)
    user_id = Column(PostgresUUID(as_uuid=True), nullable=True)
    processing_status = Column(String(20), default="pending")

    # Versioning fields
    document_family_id = Column(PostgresUUID(as_uuid=True), nullable=True)  # Groups related document versions
    version_number = Column(Integer, default=1)
    is_latest_version = Column(Boolean, default=True)
    parent_document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    version_description = Column(Text)  # User description of changes
    similarity_to_parent = Column(Float)  # Similarity score to parent version

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    embeddings = relationship("Embedding", back_populates="document", cascade="all, delete-orphan")
    clause_versions = relationship("ClauseVersion", back_populates="document", cascade="all, delete-orphan")
    parent_document = relationship("Document", remote_side=[id], backref="child_versions")
    processing_tasks = relationship("ProcessingTask", back_populates="document", cascade="all, delete-orphan")
    document_versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    changes_from = relationship("DocumentChange", foreign_keys="DocumentChange.from_document_id", cascade="all, delete-orphan")
    changes_to = relationship("DocumentChange", foreign_keys="DocumentChange.to_document_id", cascade="all, delete-orphan")


class Embedding(Base):
    """Embeddings table for storing vector representations of document chunks."""
    
    __tablename__ = "embeddings"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_id = Column(String(50), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_type = Column(String(20), default="paragraph")  # 'clause', 'paragraph', 'section'
    start_char = Column(Integer)
    end_char = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    
    # Note: Actual embedding vector will be stored in vector database (FAISS/Weaviate)
    # This table maintains metadata and references
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")


class ClauseVersion(Base):
    """Clause versions table for tracking changes in legal clauses over time."""
    
    __tablename__ = "clause_versions"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_id = Column(PostgresUUID(as_uuid=True), nullable=False)  # Logical clause identifier
    document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    clause_text = Column(Text, nullable=False)
    clause_type = Column(String(50))  # 'liability', 'termination', 'payment', etc.
    version_number = Column(Integer, default=1)
    change_type = Column(String(20))  # 'added', 'modified', 'deleted'
    change_description = Column(Text)
    confidence_score = Column(Float)
    previous_version_id = Column(PostgresUUID(as_uuid=True), ForeignKey("clause_versions.id"))
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="clause_versions")
    previous_version = relationship("ClauseVersion", remote_side=[id], backref="next_versions")
    changes_from = relationship("ClauseChange", foreign_keys="ClauseChange.from_version_id", back_populates="from_version")
    changes_to = relationship("ClauseChange", foreign_keys="ClauseChange.to_version_id", back_populates="to_version")


class ClauseChange(Base):
    """Clause changes table for tracking specific changes between clause versions."""
    
    __tablename__ = "clause_changes"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_version_id = Column(PostgresUUID(as_uuid=True), ForeignKey("clause_versions.id"))
    to_version_id = Column(PostgresUUID(as_uuid=True), ForeignKey("clause_versions.id"))
    change_summary = Column(Text)
    semantic_similarity = Column(Float)
    detected_at = Column(DateTime, default=func.now())
    
    # Relationships
    from_version = relationship("ClauseVersion", foreign_keys=[from_version_id], back_populates="changes_from")
    to_version = relationship("ClauseVersion", foreign_keys=[to_version_id], back_populates="changes_to")


class DetectedConflict(Base):
    """Detected conflicts table for storing identified conflicts between clauses."""
    
    __tablename__ = "detected_conflicts"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clause_1_id = Column(PostgresUUID(as_uuid=True), ForeignKey("clause_versions.id"))
    clause_2_id = Column(PostgresUUID(as_uuid=True), ForeignKey("clause_versions.id"))
    conflict_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")  # 'low', 'medium', 'high', 'critical'
    description = Column(Text)
    suggested_resolution = Column(Text)
    status = Column(String(20), default="open")  # 'open', 'resolved', 'dismissed'
    detected_at = Column(DateTime, default=func.now())
    resolved_at = Column(DateTime)
    
    # Relationships
    clause_1 = relationship("ClauseVersion", foreign_keys=[clause_1_id])
    clause_2 = relationship("ClauseVersion", foreign_keys=[clause_2_id])


class ChatSession(Base):
    """Chat sessions table for storing user chat interactions."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), nullable=True)
    session_name = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat messages table for storing individual messages in chat sessions."""
    
    __tablename__ = "chat_messages"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PostgresUUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    message_type = Column(String(20), nullable=False)  # 'user', 'assistant'
    content = Column(Text, nullable=False)
    context_documents = Column(Text)  # JSON array of document IDs used for context
    message_metadata = Column(Text)  # JSON metadata about the message
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class DocumentChange(Base):
    """Document changes table for tracking specific changes between document versions."""

    __tablename__ = "document_changes"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    to_document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    change_type = Column(String(20), nullable=False)  # 'added', 'modified', 'deleted', 'moved'
    section_title = Column(String(255))  # Title of the section that changed
    old_text = Column(Text)  # Original text (null for additions)
    new_text = Column(Text)  # New text (null for deletions)
    start_position = Column(Integer)  # Character position where change starts
    end_position = Column(Integer)  # Character position where change ends
    similarity_score = Column(Float)  # Similarity score between old and new text
    change_summary = Column(Text)  # Human-readable summary of the change
    confidence_score = Column(Float, default=1.0)  # Confidence in change detection
    detected_at = Column(DateTime, default=func.now())

    # Relationships
    from_document = relationship("Document", foreign_keys=[from_document_id])
    to_document = relationship("Document", foreign_keys=[to_document_id])


class DocumentVersion(Base):
    """Document versions table for enhanced version tracking with metadata."""

    __tablename__ = "document_versions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    version_tag = Column(String(50))  # User-defined version tag (e.g., "v1.0", "draft", "final")
    version_notes = Column(Text)  # Detailed notes about this version
    author = Column(String(255))  # Who created this version
    approval_status = Column(String(20), default="draft")  # 'draft', 'review', 'approved', 'rejected'
    approved_by = Column(String(255))  # Who approved this version
    approved_at = Column(DateTime)
    is_published = Column(Boolean, default=False)  # Whether this version is published/active
    change_summary = Column(Text)  # Summary of changes from previous version
    created_at = Column(DateTime, default=func.now())

    # Relationships
    document = relationship("Document")


class SearchHistory(Base):
    """Search history table for tracking version-aware searches."""

    __tablename__ = "search_history"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), nullable=True)
    query = Column(Text, nullable=False)
    search_type = Column(String(20), default="all")  # 'latest', 'all', 'specific_version', 'version_range'
    target_versions = Column(Text)  # JSON array of version IDs or criteria
    results_count = Column(Integer, default=0)
    execution_time = Column(Float)  # Search execution time in seconds
    created_at = Column(DateTime, default=func.now())


class ProcessingTask(Base):
    """Processing tasks table for tracking background document processing tasks."""

    __tablename__ = "processing_tasks"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(50), nullable=False)  # 'document_processing', 'change_analysis', etc.
    document_id = Column(PostgresUUID(as_uuid=True), ForeignKey("documents.id"))
    status = Column(String(20), default="pending")  # 'pending', 'running', 'completed', 'failed'
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    result_data = Column(Text)  # JSON result data
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    document = relationship("Document")