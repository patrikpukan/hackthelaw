import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock
import io
from pathlib import Path


@pytest.mark.api
class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Legal RAG Agent API" in data["message"]
        assert "status" in data
    
    async def test_health_check(self, client: AsyncClient):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.api
class TestDocumentUploadAPI:
    """Test document upload API endpoints."""
    
    async def test_upload_text_file(self, client: AsyncClient, sample_text_content: str):
        """Test uploading a text file."""
        files = {
            "file": ("test_contract.txt", sample_text_content, "text/plain")
        }
        
        with patch("app.workers.worker.process_document.delay") as mock_task:
            mock_task.return_value = MagicMock(id="task-123")
            
            response = await client.post("/api/v1/documents/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "document_id" in data
            assert "task_id" in data
            assert data["filename"] == "test_contract.txt"
            assert data["status"] == "processing"
            
            # Verify task was called
            mock_task.assert_called_once()
    
    async def test_upload_invalid_file_type(self, client: AsyncClient):
        """Test uploading invalid file type."""
        files = {
            "file": ("test.exe", b"invalid content", "application/exe")
        }
        
        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 422
        data = response.json()
        assert "not supported" in data["detail"]
    
    async def test_upload_no_file(self, client: AsyncClient):
        """Test upload without file."""
        response = await client.post("/api/v1/documents/upload")
        assert response.status_code == 422
    
    async def test_upload_large_file(self, client: AsyncClient):
        """Test uploading file that's too large."""
        large_content = "x" * (60 * 1024 * 1024)  # 60MB
        files = {
            "file": ("large_file.txt", large_content, "text/plain")
        }
        
        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 413
        data = response.json()
        assert "too large" in data["detail"]


@pytest.mark.api
class TestDocumentListAPI:
    """Test document listing API."""
    
    async def test_list_documents_empty(self, client: AsyncClient):
        """Test listing documents when none exist."""
        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0
    
    @patch("app.api.routers.documents.get_documents")
    async def test_list_documents_with_data(self, mock_get_docs, client: AsyncClient):
        """Test listing documents with existing data."""
        # Mock database response
        mock_get_docs.return_value = {
            "documents": [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "filename": "contract1.pdf",
                    "file_type": "pdf",
                    "upload_date": "2024-01-01T10:00:00",
                    "processing_status": "completed"
                }
            ],
            "total": 1
        }
        
        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "contract1.pdf"
        assert data["total"] == 1
    
    async def test_list_documents_with_pagination(self, client: AsyncClient):
        """Test document listing with pagination."""
        response = await client.get("/api/v1/documents/?limit=10&offset=0")
        assert response.status_code == 200
        # Should not error even with no documents


@pytest.mark.api
class TestDocumentDetailAPI:
    """Test document detail API."""
    
    async def test_get_nonexistent_document(self, client: AsyncClient):
        """Test getting a document that doesn't exist."""
        fake_id = "123e4567-e89b-12d3-a456-426614174999"
        response = await client.get(f"/api/v1/documents/{fake_id}")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch("app.api.routers.documents.get_document_by_id")
    async def test_get_document_success(self, mock_get_doc, client: AsyncClient):
        """Test successfully getting a document."""
        doc_id = "123e4567-e89b-12d3-a456-426614174000"
        mock_get_doc.return_value = {
            "id": doc_id,
            "filename": "test.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "processing_status": "completed",
            "extracted_text": "Sample text content"
        }
        
        response = await client.get(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["processing_status"] == "completed"


@pytest.mark.api
class TestChatAPI:
    """Test chat API endpoints."""
    
    async def test_chat_query_success(self, client: AsyncClient):
        """Test successful chat query."""
        query_data = {
            "query": "What are the payment terms in the contract?",
            "session_id": None,
            "document_filters": []
        }
        
        with patch("app.core.chat.generator.RAGQueryProcessor.process_query") as mock_process:
            mock_process.return_value = {
                "response": "The payment terms are 30 days net.",
                "sources": ["contract1.pdf"],
                "session_id": "new-session-123"
            }
            
            response = await client.post("/api/v1/chat/query", json=query_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "sources" in data
            assert "session_id" in data
    
    async def test_chat_query_empty_query(self, client: AsyncClient):
        """Test chat query with empty query."""
        query_data = {
            "query": "",
            "session_id": None
        }
        
        response = await client.post("/api/v1/chat/query", json=query_data)
        assert response.status_code == 422
    
    async def test_chat_query_invalid_session(self, client: AsyncClient):
        """Test chat query with invalid session ID."""
        query_data = {
            "query": "Test question",
            "session_id": "invalid-session-id"
        }
        
        response = await client.post("/api/v1/chat/query", json=query_data)
        # Should handle gracefully or return appropriate error
        assert response.status_code in [200, 404, 422]


@pytest.mark.api
class TestHistoryAPI:
    """Test history API endpoints."""
    
    async def test_get_document_history_empty(self, client: AsyncClient):
        """Test getting history for document with no changes."""
        doc_id = "123e4567-e89b-12d3-a456-426614174000"
        
        with patch("app.api.routers.history.get_document_changes") as mock_get_changes:
            mock_get_changes.return_value = []
            
            response = await client.get(f"/api/v1/history/document/{doc_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["changes"] == []
    
    async def test_get_clause_versions(self, client: AsyncClient):
        """Test getting clause version history."""
        clause_id = "clause-123"
        
        with patch("app.api.routers.history.get_clause_versions") as mock_get_versions:
            mock_get_versions.return_value = [
                {
                    "id": "version-1",
                    "clause_text": "Original clause text",
                    "version_number": 1,
                    "change_type": "added"
                },
                {
                    "id": "version-2", 
                    "clause_text": "Modified clause text",
                    "version_number": 2,
                    "change_type": "modified"
                }
            ]
            
            response = await client.get(f"/api/v1/history/clause/{clause_id}/versions")
            assert response.status_code == 200
            data = response.json()
            assert len(data["versions"]) == 2
            assert data["versions"][0]["version_number"] == 1


@pytest.mark.api
class TestConflictsAPI:
    """Test conflicts API endpoints."""
    
    async def test_get_conflicts_empty(self, client: AsyncClient):
        """Test getting conflicts when none exist."""
        response = await client.get("/api/v1/conflicts/")
        assert response.status_code == 200
        data = response.json()
        assert data["conflicts"] == []
        assert data["total"] == 0
    
    @patch("app.api.routers.conflicts.get_detected_conflicts")
    async def test_get_conflicts_with_data(self, mock_get_conflicts, client: AsyncClient):
        """Test getting conflicts with existing data."""
        mock_get_conflicts.return_value = {
            "conflicts": [
                {
                    "id": "conflict-1",
                    "conflict_type": "payment_terms",
                    "severity": "high",
                    "description": "Conflicting payment terms",
                    "status": "open"
                }
            ],
            "total": 1
        }
        
        response = await client.get("/api/v1/conflicts/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["conflicts"]) == 1
        assert data["conflicts"][0]["severity"] == "high"
    
    async def test_resolve_conflict(self, client: AsyncClient):
        """Test resolving a conflict."""
        conflict_id = "conflict-123"
        resolution_data = {
            "resolution": "Use the 30-day payment term as specified in Section 5",
            "resolved_by": "Legal Team"
        }
        
        with patch("app.api.routers.conflicts.resolve_conflict") as mock_resolve:
            mock_resolve.return_value = {"status": "resolved"}
            
            response = await client.put(
                f"/api/v1/conflicts/{conflict_id}/resolve",
                json=resolution_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resolved"


@pytest.mark.integration
class TestFullAPIWorkflow:
    """Integration tests for complete API workflows."""
    
    async def test_document_upload_and_query_workflow(self, client: AsyncClient, sample_text_content: str):
        """Test complete workflow: upload document -> process -> query."""
        # Step 1: Upload document
        files = {
            "file": ("contract.txt", sample_text_content, "text/plain")
        }
        
        with patch("app.workers.worker.process_document.delay") as mock_task:
            mock_task.return_value = MagicMock(id="task-123")
            
            upload_response = await client.post("/api/v1/documents/upload", files=files)
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            doc_id = upload_data["document_id"]
        
        # Step 2: Simulate document processing completion
        with patch("app.api.routers.documents.get_document_by_id") as mock_get_doc:
            mock_get_doc.return_value = {
                "id": doc_id,
                "filename": "contract.txt",
                "processing_status": "completed",
                "extracted_text": sample_text_content
            }
            
            # Check document status
            doc_response = await client.get(f"/api/v1/documents/{doc_id}")
            assert doc_response.status_code == 200
            assert doc_response.json()["processing_status"] == "completed"
        
        # Step 3: Query the document
        query_data = {
            "query": "What is the employment term?",
            "document_filters": [doc_id]
        }
        
        with patch("app.core.chat.generator.RAGQueryProcessor.process_query") as mock_process:
            mock_process.return_value = {
                "response": "The employment term is two (2) years.",
                "sources": ["contract.txt"],
                "session_id": "session-123"
            }
            
            chat_response = await client.post("/api/v1/chat/query", json=query_data)
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            assert "two (2) years" in chat_data["response"] 