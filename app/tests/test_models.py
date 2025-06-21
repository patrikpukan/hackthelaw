import pytest
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Document, Embedding, ClauseVersion, ClauseChange,
    DetectedConflict, ChatSession, ChatMessage, ProcessingTask
)


@pytest.mark.unit
class TestDocumentModel:
    """Test Document model."""
    
    async def test_create_document(self, db_session: AsyncSession):
        """Test creating a document."""
        document = Document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            extracted_text="Test content",
            document_hash="abc123",
            processing_status="completed"
        )
        
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        
        assert document.id is not None
        assert document.filename == "test.pdf"
        assert document.file_type == "pdf"
        assert document.processing_status == "completed"
        assert isinstance(document.upload_date, datetime)


@pytest.mark.unit
class TestEmbeddingModel:
    """Test Embedding model."""
    
    async def test_create_embedding(self, db_session: AsyncSession):
        """Test creating an embedding."""
        # First create a document
        document = Document(
            filename="test.pdf",
            file_type="pdf",
            file_size=1024,
            extracted_text="Test content",
            document_hash="abc123"
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        
        # Create embedding
        embedding = Embedding(
            document_id=document.id,
            chunk_id="chunk_1",
            chunk_text="This is a test chunk",
            chunk_type="paragraph",
            start_char=0,
            end_char=20
        )
        
        db_session.add(embedding)
        await db_session.commit()
        await db_session.refresh(embedding)
        
        assert embedding.id is not None
        assert embedding.document_id == document.id
        assert embedding.chunk_text == "This is a test chunk"
        assert embedding.chunk_type == "paragraph"


@pytest.mark.unit
class TestClauseVersionModel:
    """Test ClauseVersion model."""
    
    async def test_create_clause_version(self, db_session: AsyncSession):
        """Test creating a clause version."""
        # Create document
        document = Document(
            filename="contract.pdf",
            file_type="pdf",
            file_size=2048,
            document_hash="def456"
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        
        # Create clause version
        clause_version = ClauseVersion(
            clause_id=uuid.uuid4(),
            document_id=document.id,
            clause_text="The term of employment shall be for a period of two (2) years.",
            clause_type="termination",
            version_number=1,
            change_type="added",
            confidence_score=0.95
        )
        
        db_session.add(clause_version)
        await db_session.commit()
        await db_session.refresh(clause_version)
        
        assert clause_version.id is not None
        assert clause_version.clause_type == "termination"
        assert clause_version.version_number == 1
        assert clause_version.confidence_score == 0.95


@pytest.mark.unit
class TestClauseChangeModel:
    """Test ClauseChange model."""
    
    async def test_create_clause_change(self, db_session: AsyncSession):
        """Test creating a clause change record."""
        # Create document
        document = Document(
            filename="contract_v2.pdf",
            file_type="pdf",
            file_size=2048,
            document_hash="ghi789"
        )
        db_session.add(document)
        await db_session.commit()
        
        # Create clause versions
        clause_id = uuid.uuid4()
        
        version1 = ClauseVersion(
            clause_id=clause_id,
            document_id=document.id,
            clause_text="Term: one (1) year",
            clause_type="term",
            version_number=1
        )
        
        version2 = ClauseVersion(
            clause_id=clause_id,
            document_id=document.id,
            clause_text="Term: two (2) years",
            clause_type="term",
            version_number=2,
            previous_version_id=version1.id
        )
        
        db_session.add_all([version1, version2])
        await db_session.commit()
        await db_session.refresh(version1)
        await db_session.refresh(version2)
        
        # Create change record
        change = ClauseChange(
            from_version_id=version1.id,
            to_version_id=version2.id,
            change_summary="Term extended from 1 year to 2 years",
            semantic_similarity=0.85
        )
        
        db_session.add(change)
        await db_session.commit()
        await db_session.refresh(change)
        
        assert change.id is not None
        assert change.semantic_similarity == 0.85
        assert "1 year to 2 years" in change.change_summary


@pytest.mark.unit
class TestDetectedConflictModel:
    """Test DetectedConflict model."""
    
    async def test_create_detected_conflict(self, db_session: AsyncSession):
        """Test creating a detected conflict."""
        # Create document and clauses
        document = Document(
            filename="contract_conflicts.pdf",
            file_type="pdf",
            file_size=3072,
            document_hash="jkl012"
        )
        db_session.add(document)
        await db_session.commit()
        
        clause1 = ClauseVersion(
            clause_id=uuid.uuid4(),
            document_id=document.id,
            clause_text="Payment due in 30 days",
            clause_type="payment"
        )
        
        clause2 = ClauseVersion(
            clause_id=uuid.uuid4(),
            document_id=document.id,
            clause_text="Payment due in 45 days",
            clause_type="payment"
        )
        
        db_session.add_all([clause1, clause2])
        await db_session.commit()
        await db_session.refresh(clause1)
        await db_session.refresh(clause2)
        
        # Create conflict
        conflict = DetectedConflict(
            clause_1_id=clause1.id,
            clause_2_id=clause2.id,
            conflict_type="payment_terms",
            severity="high",
            description="Conflicting payment terms: 30 days vs 45 days",
            suggested_resolution="Clarify which payment term applies"
        )
        
        db_session.add(conflict)
        await db_session.commit()
        await db_session.refresh(conflict)
        
        assert conflict.id is not None
        assert conflict.conflict_type == "payment_terms"
        assert conflict.severity == "high"
        assert conflict.status == "open"  # default


@pytest.mark.unit
class TestChatSessionModel:
    """Test ChatSession model."""
    
    async def test_create_chat_session(self, db_session: AsyncSession):
        """Test creating a chat session."""
        session = ChatSession(
            user_id=uuid.uuid4(),
            session_name="Legal Document Review"
        )
        
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        assert session.id is not None
        assert session.session_name == "Legal Document Review"
        assert isinstance(session.created_at, datetime)


@pytest.mark.unit
class TestChatMessageModel:
    """Test ChatMessage model."""
    
    async def test_create_chat_message(self, db_session: AsyncSession):
        """Test creating a chat message."""
        # Create session first
        session = ChatSession(session_name="Test Chat")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Create message
        message = ChatMessage(
            session_id=session.id,
            message_type="user",
            content="What are the payment terms?",
            context_documents='["doc1", "doc2"]',
            message_metadata='{"tokens": 100}'
        )
        
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        
        assert message.id is not None
        assert message.message_type == "user"
        assert message.content == "What are the payment terms?"
        assert '"doc1"' in message.context_documents


@pytest.mark.unit
class TestProcessingTaskModel:
    """Test ProcessingTask model."""
    
    async def test_create_processing_task(self, db_session: AsyncSession):
        """Test creating a processing task."""
        # Create document
        document = Document(
            filename="processing_test.pdf",
            file_type="pdf",
            file_size=1536,
            document_hash="mno345"
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        
        # Create task
        task = ProcessingTask(
            task_type="document_processing",
            document_id=document.id,
            status="pending",
            progress=0.0
        )
        
        db_session.add(task)
        await db_session.commit()
        await db_session.refresh(task)
        
        assert task.id is not None
        assert task.task_type == "document_processing"
        assert task.status == "pending"
        assert task.progress == 0.0
        assert isinstance(task.created_at, datetime)


@pytest.mark.unit
class TestModelRelationships:
    """Test model relationships."""
    
    async def test_document_embeddings_relationship(self, db_session: AsyncSession):
        """Test document to embeddings relationship."""
        document = Document(
            filename="relationship_test.pdf",
            file_type="pdf",
            file_size=1024,
            document_hash="pqr678"
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        
        # Create embeddings
        embedding1 = Embedding(
            document_id=document.id,
            chunk_id="chunk_1",
            chunk_text="First chunk"
        )
        embedding2 = Embedding(
            document_id=document.id,
            chunk_id="chunk_2",
            chunk_text="Second chunk"
        )
        
        db_session.add_all([embedding1, embedding2])
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(document)
        assert len(document.embeddings) == 2
        assert document.embeddings[0].chunk_text in ["First chunk", "Second chunk"]
    
    async def test_chat_session_messages_relationship(self, db_session: AsyncSession):
        """Test chat session to messages relationship."""
        session = ChatSession(session_name="Test Relationship")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        
        # Create messages
        message1 = ChatMessage(
            session_id=session.id,
            message_type="user",
            content="First message"
        )
        message2 = ChatMessage(
            session_id=session.id,
            message_type="assistant",
            content="First response"
        )
        
        db_session.add_all([message1, message2])
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(session)
        assert len(session.messages) == 2 