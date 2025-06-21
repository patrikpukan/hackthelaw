from celery import Celery
import logging
from typing import Dict, Any
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor

from app.utils.config import settings
from app.core.ingest.processors import DocumentProcessor

# Initialize Celery
celery_app = Celery(
    "legalrag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.simple_worker"]
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


def get_sync_db_connection():
    """Get synchronous database connection."""
    # Convert async URL to sync URL
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    return psycopg2.connect(db_url)


@celery_app.task(bind=True)
def process_document_task(self, document_id: str, file_path: str):
    """Background task to process a document using synchronous approach."""
    
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
        
        # Save results to database using sync connection
        _save_processing_results_sync(document_id, result)
        
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
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update task status in database
        try:
            _update_task_status_sync(document_id, "failed", str(e))
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


def _save_processing_results_sync(document_id: str, processing_result: Dict[str, Any]):
    """Save document processing results to database using sync approach."""
    
    conn = None
    try:
        conn = get_sync_db_connection()
        cursor = conn.cursor()
        
        # Update document with extracted text
        cursor.execute("""
            UPDATE documents 
            SET extracted_text = %s, processing_status = 'completed'
            WHERE id = %s
        """, (processing_result['cleaned_text'], document_id))
        
        # Create embedding records for chunks
        for chunk in processing_result['chunks']:
            cursor.execute("""
                INSERT INTO embeddings (id, document_id, chunk_id, chunk_text, chunk_type, start_char, end_char)
                VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s)
            """, (
                document_id,
                chunk['chunk_id'],
                chunk['text'],
                chunk['chunk_type'],
                chunk['start_char'],
                chunk['end_char']
            ))
        
        # Update processing task status
        cursor.execute("""
            UPDATE processing_tasks 
            SET status = 'completed', completed_at = NOW()
            WHERE document_id = %s AND task_type = 'document_processing'
        """, (document_id,))
        
        conn.commit()
        logger.info(f"Successfully saved processing results for document {document_id}")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error saving processing results for document {document_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()


def _update_task_status_sync(document_id: str, status: str, error_message: str = None):
    """Update processing task status using sync approach."""
    
    conn = None
    try:
        conn = get_sync_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE processing_tasks 
            SET status = %s, error_message = %s, completed_at = NOW()
            WHERE document_id = %s AND task_type = 'document_processing'
        """, (status, error_message, document_id))
        
        conn.commit()
        logger.info(f"Updated task status for document {document_id} to {status}")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating task status for document {document_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()


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