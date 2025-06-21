from celery import Celery
from sqlalchemy import select, update
import asyncio
from datetime import datetime
import logging
import os

from app.db.connection import async_session
from app.db.models import DocumentVersion, DocumentChange, Document
from app.core.ingest.extractors import DocumentExtractorFactory
from app.core.history.diff_generator import extract_document_sections, compare_sections
from app.core.embedding.processor import process_document_chunks

celery_app = Celery('document_processor')
celery_app.conf.broker_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

logger = logging.getLogger(__name__)

@celery_app.task
def process_document_version(version_id: str):
    """Process a document version and detect changes."""
    asyncio.run(_process_version(version_id))

async def _process_version(version_id: str):
    """Process document version asynchronously."""
    async with async_session() as db:
        try:
            # Get version info
            version = await db.execute(
                select(DocumentVersion).where(DocumentVersion.id == version_id)
            )
            version = version.scalar_one_or_none()
            
            if not version:
                logger.error(f"Version {version_id} not found")
                return
            
            # Update status
            version.processing_status = "processing"
            await db.commit()
            
            # Extract text from file
            extractor = DocumentExtractorFactory.get_extractor(version.file_path)
            extracted_text = extractor.extract(version.file_path)
            
            # Update version with extracted text
            version.extracted_text = extracted_text
            await db.commit()
            
            # Process document for embeddings
            await process_document_chunks(version.id, extracted_text, db)
            
            # If this is not the first version, detect changes
            if version.version_number > 1:
                # Get previous version
                prev_version = await db.execute(
                    select(DocumentVersion)
                    .where(DocumentVersion.document_id == version.document_id)
                    .where(DocumentVersion.version_number == version.version_number - 1)
                )
                prev_version = prev_version.scalar_one_or_none()
                
                if prev_version and prev_version.extracted_text:
                    # Extract sections from both versions
                    old_sections = extract_document_sections(prev_version.extracted_text)
                    new_sections = extract_document_sections(extracted_text)
                    
                    # Compare sections and detect changes
                    changes = compare_sections(old_sections, new_sections)
                    
                    # Save changes to database
                    for change_data in changes:
                        change = DocumentChange(
                            version_id=version.id,
                            change_type=change_data["change_type"],
                            section_title=change_data["section_title"],
                            old_text=change_data["old_text"],
                            new_text=change_data["new_text"],
                            similarity_score=change_data["similarity_score"],
                            change_summary=change_data["summary"]
                        )
                        db.add(change)
            
            # Update status to completed
            version.processing_status = "completed"
            await db.commit()
            
            logger.info(f"Successfully processed document version {version_id}")
            
        except Exception as e:
            logger.error(f"Error processing document version {version_id}: {str(e)}")
            # Update status to failed
            await db.execute(
                update(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .values(processing_status="failed")
            )
            await db.commit()