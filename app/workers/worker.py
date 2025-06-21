from celery import Celery
import asyncio
import logging
from typing import Dict, Any

from app.utils.config import settings
from app.core.ingest.processors import DocumentProcessor
from app.db.connection import async_session
from app.db.models import Document, ProcessingTask

# Initialize Celery
celery_app = Celery(
    "legalrag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.worker"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_max_tasks_per_child=1000,
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def process_document_task(self, document_id: str, file_path: str):
    """Background task to process a document."""
    
    try:
        # Update task status
        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "Starting document processing"})
        
        # Initialize processor
        processor = DocumentProcessor()
        
        # Update progress
        self.update_state(state="PROGRESS", meta={"progress": 25, "status": "Extracting text from document"})
        
        # Process document
        result = processor.process_document(file_path)
        
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error during processing'))
        
        # Update progress
        self.update_state(state="PROGRESS", meta={"progress": 75, "status": "Saving processed data"})
        
        # Save results to database (run async function in sync context)
        try:
            asyncio.run(_save_processing_results(document_id, result))
        except Exception as e:
            logger.error(f"Error saving processing results for document {document_id}: {e}")
            raise
        
        # Final update
        self.update_state(state="SUCCESS", meta={"progress": 100, "status": "Document processing completed"})
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_created": result['processing_stats']['total_chunks'],
            "message": "Document processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        
        # Update task status in database
        try:
            asyncio.run(_update_task_status(document_id, "failed", str(e)))
        except Exception as db_error:
            logger.error(f"Error updating task status for document {document_id}: {db_error}")
        
        self.update_state(
            state="FAILURE",
            meta={"progress": 0, "status": f"Error: {str(e)}"}
        )
        
        return {
            "success": False,
            "document_id": document_id,
            "error": str(e)
        }


async def _save_processing_results(document_id: str, processing_result: Dict[str, Any]):
    """Save document processing results to database."""
    
    async with async_session() as session:
        try:
            # Update document with extracted text
            document = await session.get(Document, document_id)
            if document:
                document.extracted_text = processing_result['cleaned_text']
                document.processing_status = "completed"
            
            # Create embedding records for chunks
            from app.db.models import Embedding
            
            for chunk in processing_result['chunks']:
                embedding = Embedding(
                    document_id=document_id,
                    chunk_id=chunk['chunk_id'],
                    chunk_text=chunk['text'],
                    chunk_type=chunk['chunk_type'],
                    start_char=chunk['start_char'],
                    end_char=chunk['end_char']
                )
                session.add(embedding)
            
            # Update processing task status
            await _update_task_status_in_session(session, document_id, "completed")
            
            await session.commit()
            logger.info(f"Successfully saved processing results for document {document_id}")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error saving processing results for document {document_id}: {e}")
            raise


async def _update_task_status(document_id: str, status: str, error_message: str = None):
    """Update processing task status."""
    
    async with async_session() as session:
        await _update_task_status_in_session(session, document_id, status, error_message)
        await session.commit()


async def _update_task_status_in_session(session, document_id: str, status: str, error_message: str = None):
    """Update processing task status within existing session."""
    
    from sqlalchemy import text
    
    query = """
        UPDATE processing_tasks 
        SET status = :status, error_message = :error_message, completed_at = NOW()
        WHERE document_id = :document_id AND task_type = 'document_processing'
    """
    
    await session.execute(
        text(query),
        {
            "status": status,
            "error_message": error_message,
            "document_id": document_id
        }
    )


# Additional tasks can be added here

@celery_app.task
def analyze_clause_changes_task(document_id: str):
    """Background task to analyze clause changes."""
    # TODO: Implement clause change analysis
    return {"message": "Clause change analysis not yet implemented"}


@celery_app.task
def detect_conflicts_task(document_id: str):
    """Background task to detect conflicts."""
    # TODO: Implement conflict detection
    return {"message": "Conflict detection not yet implemented"}


if __name__ == "__main__":
    celery_app.start() 