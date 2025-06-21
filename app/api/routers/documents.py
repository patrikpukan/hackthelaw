from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
import os
import hashlib
import aiofiles
from pathlib import Path

from app.db.connection import get_db
from app.db.models import Document, ProcessingTask, DocumentVersion, DocumentChange
from app.utils.config import settings
from app.core.ingest.extractors import DocumentExtractor
from app.core.ingest.processors import DocumentProcessor
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file."""
    # Check filename exists
    if not file.filename:
        return False
        
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        return False
    
    # Check file size
    if not file.size or file.size > settings.max_file_size:
        return False
    
    return True


async def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file."""
    hash_sha256 = hashlib.sha256()
    async with aiofiles.open(file_path, 'rb') as f:
        async for chunk in f:
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload a legal document for processing."""
    
    # Validate file
    if not await validate_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type or size"
        )
    
    # Create upload directory if not exists
    upload_dir = Path(settings.upload_path)
    upload_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{file.filename}_{hash(file.filename)}_{file.size}{file_ext}"
    file_path = upload_dir / unique_filename
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Calculate file hash
        file_hash = await calculate_file_hash(str(file_path))
        
        # Check if document already exists
        existing_doc = await db.execute(
            text("SELECT id FROM documents WHERE document_hash = :hash"),
            {"hash": file_hash}
        )
        if existing_doc.scalar():
            # Remove duplicate file
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document already exists"
            )
        
        # Create document record
        document = Document(
            filename=file.filename,
            file_type=file_ext,
            file_size=file.size,
            document_hash=file_hash,
            processing_status="pending"
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Create processing task
        task = ProcessingTask(
            task_type="document_processing",
            document_id=document.id,
            status="pending"
        )
        
        db.add(task)
        await db.commit()
        
        # Trigger background processing task
        try:
            from app.workers.simple_worker import process_document_task
            process_document_task.delay(str(document.id), str(file_path))
            logger.info(f"Started background processing for document {document.id}")
        except Exception as e:
            logger.warning(f"Failed to start background processing: {e}")
            # Don't fail the upload if we can't start processing
        
        return {
            "message": "Document uploaded successfully",
            "document_id": str(document.id),
            "filename": file.filename,
            "status": "processing"
        }
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            os.remove(file_path)
        
        # Rollback database changes
        await db.rollback()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post("/upload-version/{document_family_id}", status_code=status.HTTP_201_CREATED)
async def upload_document_version(
    document_family_id: str,
    file: UploadFile = File(...),
    version_description: Optional[str] = None,
    version_tag: Optional[str] = None,
    author: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Upload a new version of an existing document."""

    # Validate file
    if not await validate_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type or size"
        )

    try:
        # Check if document family exists
        family_check = await db.execute(
            text("SELECT id FROM documents WHERE document_family_id = :family_id OR id = :family_id LIMIT 1"),
            {"family_id": document_family_id}
        )
        if not family_check.scalar():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document family not found"
            )

        # Get the latest version number for this family
        version_result = await db.execute(
            text("""
            SELECT MAX(version_number) as max_version
            FROM documents
            WHERE document_family_id = :family_id OR id = :family_id
            """),
            {"family_id": document_family_id}
        )
        max_version = version_result.scalar() or 0
        new_version_number = max_version + 1

        # Create upload directory if not exists
        upload_dir = Path(settings.upload_path)
        upload_dir.mkdir(exist_ok=True)

        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{file.filename}_v{new_version_number}_{hash(file.filename)}_{file.size}{file_ext}"
        file_path = upload_dir / unique_filename

        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Calculate file hash
        file_hash = await calculate_file_hash(str(file_path))

        # Check if this exact version already exists
        existing_doc = await db.execute(
            text("SELECT id FROM documents WHERE document_hash = :hash"),
            {"hash": file_hash}
        )
        if existing_doc.scalar():
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This exact document version already exists"
            )

        # Mark all other versions in this family as not latest
        await db.execute(
            text("""
            UPDATE documents
            SET is_latest_version = FALSE
            WHERE document_family_id = :family_id OR id = :family_id
            """),
            {"family_id": document_family_id}
        )

        # Create new document version record
        document = Document(
            filename=file.filename,
            file_type=file_ext,
            file_size=file.size,
            document_hash=file_hash,
            processing_status="pending",
            document_family_id=document_family_id,
            version_number=new_version_number,
            is_latest_version=True,
            version_description=version_description
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        # Create version metadata if provided
        if version_tag or author:
            doc_version = DocumentVersion(
                document_id=document.id,
                version_tag=version_tag,
                author=author,
                change_summary=version_description
            )
            db.add(doc_version)
            await db.commit()

        # Create processing task
        task = ProcessingTask(
            task_type="document_processing",
            document_id=document.id,
            status="pending"
        )

        db.add(task)
        await db.commit()

        # Trigger background processing task
        try:
            from app.workers.simple_worker import process_document_task
            process_document_task.delay(str(document.id), str(file_path))
            logger.info(f"Started background processing for document version {document.id}")
        except Exception as e:
            logger.warning(f"Failed to start background processing: {e}")

        return {
            "message": "Document version uploaded successfully",
            "document_id": str(document.id),
            "document_family_id": document_family_id,
            "version_number": new_version_number,
            "filename": file.filename,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        if 'file_path' in locals() and file_path.exists():
            os.remove(file_path)

        # Rollback database changes
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document version: {str(e)}"
        )


@router.get("/")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List uploaded documents."""
    
    try:
        result = await db.execute(
            text("""
            SELECT d.id, d.filename, d.file_type, d.upload_date as uploaded_at, d.file_size,
                   d.processing_status, COUNT(e.id) as chunk_count
            FROM documents d
            LEFT JOIN embeddings e ON d.id = e.document_id
            GROUP BY d.id, d.filename, d.file_type, d.upload_date, d.file_size, d.processing_status
            ORDER BY d.upload_date DESC
            OFFSET :skip LIMIT :limit
            """),
            {"skip": skip, "limit": limit}
        )
        
        documents = []
        for row in result.fetchall():
            documents.append({
                "id": str(row.id),
                "filename": row.filename,
                "file_type": row.file_type,
                "uploaded_at": row.uploaded_at.isoformat(),
                "file_size": row.file_size,
                "processing_status": row.processing_status,
                "chunk_count": row.chunk_count or 0
            })
        
        return {
            "documents": documents,
            "total": len(documents),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch documents: {str(e)}"
        )


@router.get("/family/{family_id}/versions")
async def get_document_versions(
    family_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all versions of a document family."""

    try:
        result = await db.execute(
            text("""
            SELECT d.id, d.filename, d.version_number, d.is_latest_version,
                   d.version_description, d.upload_date, d.processing_status,
                   d.file_size, dv.version_tag, dv.author, dv.approval_status,
                   dv.is_published, dv.change_summary
            FROM documents d
            LEFT JOIN document_versions dv ON d.id = dv.document_id
            WHERE d.document_family_id = :family_id OR d.id = :family_id
            ORDER BY d.version_number DESC
            """),
            {"family_id": family_id}
        )

        versions = []
        for row in result.fetchall():
            versions.append({
                "id": str(row.id),
                "filename": row.filename,
                "version_number": row.version_number,
                "is_latest_version": row.is_latest_version,
                "version_description": row.version_description,
                "upload_date": row.upload_date.isoformat(),
                "processing_status": row.processing_status,
                "file_size": row.file_size,
                "version_tag": row.version_tag,
                "author": row.author,
                "approval_status": row.approval_status,
                "is_published": row.is_published,
                "change_summary": row.change_summary
            })

        if not versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document family not found"
            )

        return {
            "family_id": family_id,
            "versions": versions,
            "total_versions": len(versions),
            "latest_version": next((v for v in versions if v["is_latest_version"]), None)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch document versions: {str(e)}"
        )


@router.post("/detect-version")
async def detect_document_version(
    file: UploadFile = File(...),
    similarity_threshold: float = 0.7,
    db: AsyncSession = Depends(get_db)
):
    """Detect if uploaded document is a new version of an existing document."""

    # Validate file
    if not await validate_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type or size"
        )

    try:
        # Save file temporarily for analysis
        upload_dir = Path(settings.upload_path) / "temp"
        upload_dir.mkdir(exist_ok=True)

        file_ext = Path(file.filename).suffix.lower()
        temp_filename = f"temp_{hash(file.filename)}_{file.size}{file_ext}"
        temp_file_path = upload_dir / temp_filename

        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Extract text from uploaded file
        try:
            from app.core.ingest.extractors import DocumentExtractorFactory
            extractor = DocumentExtractorFactory.get_extractor(str(temp_file_path))
            extracted_data = extractor.extract(str(temp_file_path))
            new_text = extracted_data.get('text', '')
        except Exception as e:
            logger.warning(f"Failed to extract text from uploaded file: {e}")
            new_text = ""

        # Get all existing documents for comparison
        result = await db.execute(
            text("""
            SELECT id, filename, extracted_text, document_family_id, version_number
            FROM documents
            WHERE extracted_text IS NOT NULL
            AND processing_status = 'completed'
            ORDER BY upload_date DESC
            """)
        )

        potential_matches = []

        for row in result.fetchall():
            if not row.extracted_text or not new_text:
                continue

            # Calculate similarity using simple text comparison
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, row.extracted_text, new_text).ratio()

            if similarity >= similarity_threshold:
                potential_matches.append({
                    "document_id": str(row.id),
                    "filename": row.filename,
                    "similarity_score": similarity,
                    "document_family_id": str(row.document_family_id) if row.document_family_id else str(row.id),
                    "current_version": row.version_number
                })

        # Clean up temporary file
        if temp_file_path.exists():
            os.remove(temp_file_path)

        # Sort by similarity score
        potential_matches.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "filename": file.filename,
            "is_new_version": len(potential_matches) > 0,
            "potential_matches": potential_matches[:5],  # Return top 5 matches
            "similarity_threshold": similarity_threshold,
            "recommendation": {
                "action": "upload_version" if potential_matches else "upload_new",
                "target_family_id": potential_matches[0]["document_family_id"] if potential_matches else None,
                "confidence": potential_matches[0]["similarity_score"] if potential_matches else 0.0
            }
        }

    except Exception as e:
        # Clean up temporary file on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect document version: {str(e)}"
        )


@router.get("/version-management")
async def version_management_interface():
    """Serve the version management interface."""
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
    from fastapi import Request

    templates = Jinja2Templates(directory="app/templates")

    # Create a mock request object for template rendering
    class MockRequest:
        def __init__(self):
            self.url = type('obj', (object,), {'path': '/version-management'})()

    request = MockRequest()

    return templates.TemplateResponse("version_management.html", {"request": request})


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get document details by ID."""
    
    try:
        result = await db.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": document_id}
        )
        
        document = result.fetchone()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "id": str(document.id),
            "filename": document.filename,
            "file_type": document.file_type,
            "upload_date": document.upload_date.isoformat(),
            "file_size": document.file_size,
            "processing_status": document.processing_status,
            "document_hash": document.document_hash,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch document: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and all related records."""
    
    try:
        # Check if document exists
        result = await db.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": document_id}
        )
        
        document = result.fetchone()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete in correct order to avoid foreign key violations
        
        # 1. Delete processing tasks first
        await db.execute(
            text("DELETE FROM processing_tasks WHERE document_id = :id"),
            {"id": document_id}
        )
        logger.info(f"Deleted processing tasks for document {document_id}")
        
        # 2. Delete embeddings
        await db.execute(
            text("DELETE FROM embeddings WHERE document_id = :id"),
            {"id": document_id}
        )
        logger.info(f"Deleted embeddings for document {document_id}")
        
        # 3. Delete clause versions
        await db.execute(
            text("DELETE FROM clause_versions WHERE document_id = :id"),
            {"id": document_id}
        )
        logger.info(f"Deleted clause versions for document {document_id}")
        
        # 4. Finally delete the document itself
        await db.execute(
            text("DELETE FROM documents WHERE id = :id"),
            {"id": document_id}
        )
        logger.info(f"Deleted document {document_id}")
        
        await db.commit()
        
        # TODO: Delete file from storage
        # TODO: Remove embeddings from vector database (Weaviate)
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/{document_id}/status")
async def get_processing_status(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get document processing status."""
    
    try:
        result = await db.execute(
            text("""
            SELECT d.processing_status, t.status as task_status, t.progress, t.error_message
            FROM documents d
            LEFT JOIN processing_tasks t ON d.id = t.document_id
            WHERE d.id = :id
            ORDER BY t.created_at DESC
            LIMIT 1
            """),
            {"id": document_id}
        )
        
        row = result.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "document_id": document_id,
            "processing_status": row.processing_status,
            "task_status": row.task_status,
            "progress": row.progress or 0.0,
            "error_message": row.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch processing status: {str(e)}"
        ) 