from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from app.db.connection import get_db

router = APIRouter()


class ConflictResponse(BaseModel):
    id: str
    conflict_type: str
    severity: str
    description: str
    status: str
    clauses: List[dict]


@router.get("/")
async def list_conflicts(
    status_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """List detected conflicts with optional filters."""
    
    try:
        # Build query with filters
        query = """
            SELECT dc.id, dc.conflict_type, dc.severity, dc.description,
                   dc.status, dc.detected_at,
                   cv1.clause_text as clause1_text, cv1.clause_type as clause1_type,
                   cv2.clause_text as clause2_text, cv2.clause_type as clause2_type,
                   d1.filename as doc1_filename, d2.filename as doc2_filename
            FROM detected_conflicts dc
            JOIN clause_versions cv1 ON dc.clause_1_id = cv1.id
            JOIN clause_versions cv2 ON dc.clause_2_id = cv2.id
            JOIN documents d1 ON cv1.document_id = d1.id
            JOIN documents d2 ON cv2.document_id = d2.id
            WHERE 1=1
        """
        
        params = {"skip": skip, "limit": limit}
        
        if status_filter:
            query += " AND dc.status = :status"
            params["status"] = status_filter
        
        if severity_filter:
            query += " AND dc.severity = :severity"
            params["severity"] = severity_filter
        
        query += " ORDER BY dc.detected_at DESC OFFSET :skip LIMIT :limit"
        
        result = await db.execute(query, params)
        
        conflicts = []
        for row in result.fetchall():
            conflicts.append({
                "id": str(row.id),
                "conflict_type": row.conflict_type,
                "severity": row.severity,
                "description": row.description,
                "status": row.status,
                "detected_at": row.detected_at.isoformat(),
                "clauses": [
                    {
                        "text": row.clause1_text,
                        "type": row.clause1_type,
                        "document": row.doc1_filename
                    },
                    {
                        "text": row.clause2_text,
                        "type": row.clause2_type,
                        "document": row.doc2_filename
                    }
                ]
            })
        
        return {
            "conflicts": conflicts,
            "total": len(conflicts),
            "skip": skip,
            "limit": limit,
            "filters": {
                "status": status_filter,
                "severity": severity_filter
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conflicts: {str(e)}"
        )


@router.get("/{conflict_id}")
async def get_conflict_details(
    conflict_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific conflict."""
    
    try:
        result = await db.execute(
            """
            SELECT dc.*, 
                   cv1.clause_text as clause1_text, cv1.clause_type as clause1_type,
                   cv2.clause_text as clause2_text, cv2.clause_type as clause2_type,
                   d1.filename as doc1_filename, d2.filename as doc2_filename
            FROM detected_conflicts dc
            JOIN clause_versions cv1 ON dc.clause_1_id = cv1.id
            JOIN clause_versions cv2 ON dc.clause_2_id = cv2.id
            JOIN documents d1 ON cv1.document_id = d1.id
            JOIN documents d2 ON cv2.document_id = d2.id
            WHERE dc.id = :conflict_id
            """,
            {"conflict_id": conflict_id}
        )
        
        row = result.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conflict not found"
            )
        
        return {
            "id": str(row.id),
            "conflict_type": row.conflict_type,
            "severity": row.severity,
            "description": row.description,
            "suggested_resolution": row.suggested_resolution,
            "status": row.status,
            "detected_at": row.detected_at.isoformat(),
            "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
            "clauses": [
                {
                    "id": str(row.clause_1_id),
                    "text": row.clause1_text,
                    "type": row.clause1_type,
                    "document": row.doc1_filename
                },
                {
                    "id": str(row.clause_2_id),
                    "text": row.clause2_text,
                    "type": row.clause2_type,
                    "document": row.doc2_filename
                }
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conflict details: {str(e)}"
        )


@router.patch("/{conflict_id}/status")
async def update_conflict_status(
    conflict_id: str,
    new_status: str,
    db: AsyncSession = Depends(get_db)
):
    """Update the status of a conflict (resolve, dismiss, etc.)."""
    
    valid_statuses = ["open", "resolved", "dismissed"]
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )
    
    try:
        # Check if conflict exists
        result = await db.execute(
            "SELECT id FROM detected_conflicts WHERE id = :id",
            {"id": conflict_id}
        )
        
        if not result.scalar():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conflict not found"
            )
        
        # Update status
        update_query = "UPDATE detected_conflicts SET status = :status"
        params = {"id": conflict_id, "status": new_status}
        
        if new_status == "resolved":
            update_query += ", resolved_at = NOW()"
        
        update_query += " WHERE id = :id"
        
        await db.execute(update_query, params)
        await db.commit()
        
        return {
            "message": f"Conflict status updated to {new_status}",
            "conflict_id": conflict_id,
            "new_status": new_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conflict status: {str(e)}"
        )


@router.get("/stats/summary")
async def get_conflict_statistics(
    db: AsyncSession = Depends(get_db)
):
    """Get summary statistics about conflicts."""
    
    try:
        result = await db.execute(
            """
            SELECT 
                status,
                severity,
                COUNT(*) as count
            FROM detected_conflicts 
            GROUP BY status, severity
            ORDER BY status, severity
            """
        )
        
        stats = {
            "by_status": {},
            "by_severity": {},
            "total": 0
        }
        
        for row in result.fetchall():
            status_key = row.status
            severity_key = row.severity
            count = row.count
            
            if status_key not in stats["by_status"]:
                stats["by_status"][status_key] = 0
            stats["by_status"][status_key] += count
            
            if severity_key not in stats["by_severity"]:
                stats["by_severity"][severity_key] = 0
            stats["by_severity"][severity_key] += count
            
            stats["total"] += count
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conflict statistics: {str(e)}"
        ) 