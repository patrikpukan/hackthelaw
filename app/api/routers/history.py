from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.db.connection import get_db
from app.core.history.diff_generator import generate_document_diff, generate_semantic_diff
from app.core.cache.version_cache import get_diff_cache, get_search_cache, get_metadata_cache

router = APIRouter()


class ClauseHistoryResponse(BaseModel):
    clause_id: str
    versions: List[dict]


@router.get("/clauses/{clause_id}")
async def get_clause_history(
    clause_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get change history for a specific clause."""
    
    try:
        result = await db.execute(
            """
            SELECT cv.id, cv.clause_text, cv.clause_type, cv.version_number,
                   cv.change_type, cv.change_description, cv.confidence_score,
                   cv.created_at, d.filename
            FROM clause_versions cv
            JOIN documents d ON cv.document_id = d.id
            WHERE cv.clause_id = :clause_id
            ORDER BY cv.version_number ASC
            """,
            {"clause_id": clause_id}
        )
        
        versions = []
        for row in result.fetchall():
            versions.append({
                "id": str(row.id),
                "clause_text": row.clause_text,
                "clause_type": row.clause_type,
                "version_number": row.version_number,
                "change_type": row.change_type,
                "change_description": row.change_description,
                "confidence_score": row.confidence_score,
                "created_at": row.created_at.isoformat(),
                "document_filename": row.filename
            })
        
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clause not found"
            )
        
        return ClauseHistoryResponse(
            clause_id=clause_id,
            versions=versions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch clause history: {str(e)}"
        )


@router.get("/documents/{document_id}/changes")
async def get_document_changes(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all changes detected in a specific document."""
    
    try:
        result = await db.execute(
            """
            SELECT cv.clause_id, cv.clause_type, cv.change_type,
                   cv.change_description, cv.confidence_score, cv.created_at
            FROM clause_versions cv
            WHERE cv.document_id = :document_id
            AND cv.change_type IS NOT NULL
            ORDER BY cv.created_at DESC
            """,
            {"document_id": document_id}
        )
        
        changes = []
        for row in result.fetchall():
            changes.append({
                "clause_id": str(row.clause_id),
                "clause_type": row.clause_type,
                "change_type": row.change_type,
                "change_description": row.change_description,
                "confidence_score": row.confidence_score,
                "detected_at": row.created_at.isoformat()
            })
        
        return {
            "document_id": document_id,
            "changes": changes,
            "total_changes": len(changes)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch document changes: {str(e)}"
        )


@router.get("/recent")
async def get_recent_changes(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get recent clause changes across all documents."""
    
    try:
        result = await db.execute(
            """
            SELECT cv.clause_id, cv.clause_type, cv.change_type,
                   cv.change_description, cv.confidence_score, cv.created_at,
                   d.filename
            FROM clause_versions cv
            JOIN documents d ON cv.document_id = d.id
            WHERE cv.change_type IS NOT NULL
            ORDER BY cv.created_at DESC
            LIMIT :limit
            """,
            {"limit": limit}
        )
        
        changes = []
        for row in result.fetchall():
            changes.append({
                "clause_id": str(row.clause_id),
                "clause_type": row.clause_type,
                "change_type": row.change_type,
                "change_description": row.change_description,
                "confidence_score": row.confidence_score,
                "detected_at": row.created_at.isoformat(),
                "document_filename": row.filename
            })
        
        return {
            "recent_changes": changes,
            "total": len(changes)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch recent changes: {str(e)}"
        )


@router.get("/compare/{version1_id}/{version2_id}")
async def compare_clause_versions(
    version1_id: str,
    version2_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Compare two versions of a clause."""
    
    try:
        result = await db.execute(
            """
            SELECT cv.id, cv.clause_text, cv.version_number, cv.created_at,
                   d.filename
            FROM clause_versions cv
            JOIN documents d ON cv.document_id = d.id
            WHERE cv.id IN (:version1, :version2)
            ORDER BY cv.version_number ASC
            """,
            {"version1": version1_id, "version2": version2_id}
        )
        
        versions = []
        for row in result.fetchall():
            versions.append({
                "id": str(row.id),
                "clause_text": row.clause_text,
                "version_number": row.version_number,
                "created_at": row.created_at.isoformat(),
                "document_filename": row.filename
            })
        
        if len(versions) != 2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both clause versions not found"
            )
        
        # Generate detailed diff analysis
        text1 = versions[0]["clause_text"]
        text2 = versions[1]["clause_text"]

        diff_result = generate_document_diff(text1, text2)

        return {
            "version1": versions[0],
            "version2": versions[1],
            "diff": diff_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare clause versions: {str(e)}"
        )


@router.get("/documents/compare/{document1_id}/{document2_id}")
async def compare_document_versions(
    document1_id: str,
    document2_id: str,
    comparison_type: str = "full",  # 'full', 'semantic', 'side_by_side'
    db: AsyncSession = Depends(get_db)
):
    """Compare two document versions with comprehensive diff analysis."""

    try:
        # Get both documents
        result = await db.execute(
            text("""
            SELECT id, filename, extracted_text, version_number, upload_date,
                   document_family_id, version_description
            FROM documents
            WHERE id IN (:doc1, :doc2)
            ORDER BY version_number ASC
            """),
            {"doc1": document1_id, "doc2": document2_id}
        )

        documents = []
        for row in result.fetchall():
            documents.append({
                "id": str(row.id),
                "filename": row.filename,
                "extracted_text": row.extracted_text,
                "version_number": row.version_number,
                "upload_date": row.upload_date.isoformat(),
                "document_family_id": str(row.document_family_id) if row.document_family_id else None,
                "version_description": row.version_description
            })

        if len(documents) != 2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both documents not found"
            )

        # Check if documents have extracted text
        if not documents[0]["extracted_text"] or not documents[1]["extracted_text"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Documents must be processed before comparison"
            )

        text1 = documents[0]["extracted_text"]
        text2 = documents[1]["extracted_text"]

        # Check cache first
        diff_cache = get_diff_cache()
        cached_diff = await diff_cache.get_diff(document1_id, document2_id, comparison_type)

        if cached_diff:
            diff_result = cached_diff
        else:
            # Generate appropriate comparison based on type
            if comparison_type == "semantic":
                diff_result = generate_semantic_diff(text1, text2)
            else:
                diff_result = generate_document_diff(text1, text2)

            # Cache the result
            await diff_cache.cache_diff(document1_id, document2_id, diff_result, comparison_type)

        return {
            "document1": documents[0],
            "document2": documents[1],
            "comparison_type": comparison_type,
            "diff": diff_result,
            "metadata": {
                "same_family": documents[0]["document_family_id"] == documents[1]["document_family_id"],
                "version_gap": abs(documents[1]["version_number"] - documents[0]["version_number"]) if documents[0]["version_number"] and documents[1]["version_number"] else None
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare documents: {str(e)}"
        )


@router.get("/families")
async def list_document_families(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """List all document families with version counts."""

    try:
        result = await db.execute(
            text("""
            SELECT
                COALESCE(document_family_id, id) as family_id,
                COUNT(*) as version_count,
                MAX(version_number) as latest_version,
                MIN(upload_date) as first_uploaded,
                MAX(upload_date) as last_updated,
                STRING_AGG(DISTINCT filename, ', ') as filenames
            FROM documents
            GROUP BY COALESCE(document_family_id, id)
            ORDER BY last_updated DESC
            OFFSET :skip LIMIT :limit
            """),
            {"skip": skip, "limit": limit}
        )

        families = []
        for row in result.fetchall():
            families.append({
                "family_id": str(row.family_id),
                "version_count": row.version_count,
                "latest_version": row.latest_version,
                "first_uploaded": row.first_uploaded.isoformat(),
                "last_updated": row.last_updated.isoformat(),
                "filenames": row.filenames
            })

        return {
            "families": families,
            "total": len(families),
            "skip": skip,
            "limit": limit
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list document families: {str(e)}"
        )


@router.get("/families/{family_id}/timeline")
async def get_family_timeline(
    family_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get chronological timeline of changes for a document family."""

    try:
        # Check cache first
        metadata_cache = get_metadata_cache()
        cached_timeline = await metadata_cache.get_family_versions(family_id)

        if cached_timeline:
            timeline = cached_timeline
        else:
            # Get all versions in the family
            result = await db.execute(
                text("""
                SELECT d.id, d.filename, d.version_number, d.upload_date,
                       d.version_description, d.processing_status,
                       dv.version_tag, dv.author, dv.change_summary,
                       COUNT(dc.id) as change_count
                FROM documents d
                LEFT JOIN document_versions dv ON d.id = dv.document_id
                LEFT JOIN document_changes dc ON d.id = dc.to_document_id
                WHERE d.document_family_id = :family_id OR d.id = :family_id
                GROUP BY d.id, d.filename, d.version_number, d.upload_date,
                         d.version_description, d.processing_status,
                         dv.version_tag, dv.author, dv.change_summary
                ORDER BY d.version_number ASC
                """),
                {"family_id": family_id}
            )

            timeline = []
            for row in result.fetchall():
                timeline.append({
                    "document_id": str(row.id),
                    "filename": row.filename,
                    "version_number": row.version_number,
                    "upload_date": row.upload_date.isoformat(),
                    "version_description": row.version_description,
                    "processing_status": row.processing_status,
                    "version_tag": row.version_tag,
                    "author": row.author,
                    "change_summary": row.change_summary,
                    "change_count": row.change_count
                })

            # Cache the timeline
            await metadata_cache.cache_family_versions(family_id, timeline)

        if not timeline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document family not found"
            )

        return {
            "family_id": family_id,
            "timeline": timeline,
            "total_versions": len(timeline)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get family timeline: {str(e)}"
        )


@router.post("/families/{family_id}/merge-versions")
async def merge_document_versions(
    family_id: str,
    source_version_id: str,
    target_version_id: str,
    merge_strategy: str = "manual",  # 'manual', 'auto_accept_newer', 'auto_accept_older'
    db: AsyncSession = Depends(get_db)
):
    """Merge changes between two document versions (placeholder for future implementation)."""

    # This is a complex feature that would require sophisticated merge algorithms
    # For now, return a placeholder response

    return {
        "message": "Document version merging is not yet implemented",
        "family_id": family_id,
        "source_version_id": source_version_id,
        "target_version_id": target_version_id,
        "merge_strategy": merge_strategy,
        "status": "pending_implementation"
    }


@router.patch("/documents/{document_id}/version-metadata")
async def update_version_metadata(
    document_id: str,
    version_tag: Optional[str] = None,
    version_notes: Optional[str] = None,
    author: Optional[str] = None,
    approval_status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Update metadata for a document version."""

    try:
        # Check if document exists
        doc_result = await db.execute(
            text("SELECT id FROM documents WHERE id = :id"),
            {"id": document_id}
        )

        if not doc_result.scalar():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check if version metadata exists
        version_result = await db.execute(
            text("SELECT id FROM document_versions WHERE document_id = :id"),
            {"id": document_id}
        )

        version_id = version_result.scalar()

        if version_id:
            # Update existing version metadata
            update_fields = []
            params = {"id": version_id}

            if version_tag is not None:
                update_fields.append("version_tag = :version_tag")
                params["version_tag"] = version_tag

            if version_notes is not None:
                update_fields.append("version_notes = :version_notes")
                params["version_notes"] = version_notes

            if author is not None:
                update_fields.append("author = :author")
                params["author"] = author

            if approval_status is not None:
                update_fields.append("approval_status = :approval_status")
                params["approval_status"] = approval_status

            if update_fields:
                await db.execute(
                    text(f"UPDATE document_versions SET {', '.join(update_fields)} WHERE id = :id"),
                    params
                )
        else:
            # Create new version metadata
            await db.execute(
                text("""
                INSERT INTO document_versions (document_id, version_tag, version_notes, author, approval_status)
                VALUES (:document_id, :version_tag, :version_notes, :author, :approval_status)
                """),
                {
                    "document_id": document_id,
                    "version_tag": version_tag,
                    "version_notes": version_notes,
                    "author": author,
                    "approval_status": approval_status or "draft"
                }
            )

        await db.commit()

        return {
            "message": "Version metadata updated successfully",
            "document_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update version metadata: {str(e)}"
        )


@router.get("/search/versions")
async def search_across_versions(
    query: str,
    family_id: Optional[str] = None,
    version_range: Optional[str] = None,  # e.g., "1-3" or "latest"
    search_type: str = "all",  # 'all', 'latest', 'specific'
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Search across document versions with version-aware filtering."""

    try:
        # Build base query
        base_query = """
        SELECT d.id, d.filename, d.version_number, d.extracted_text,
               d.document_family_id, d.upload_date, d.is_latest_version
        FROM documents d
        WHERE d.extracted_text IS NOT NULL
        AND d.processing_status = 'completed'
        """

        params = {"query": f"%{query}%", "limit": limit}

        # Add family filter
        if family_id:
            base_query += " AND (d.document_family_id = :family_id OR d.id = :family_id)"
            params["family_id"] = family_id

        # Add version filtering
        if search_type == "latest":
            base_query += " AND d.is_latest_version = TRUE"
        elif version_range and version_range != "all":
            if "-" in version_range:
                start_version, end_version = map(int, version_range.split("-"))
                base_query += " AND d.version_number BETWEEN :start_version AND :end_version"
                params["start_version"] = start_version
                params["end_version"] = end_version
            elif version_range.isdigit():
                base_query += " AND d.version_number = :version_number"
                params["version_number"] = int(version_range)

        # Add text search
        base_query += " AND d.extracted_text ILIKE :query"
        base_query += " ORDER BY d.upload_date DESC LIMIT :limit"

        result = await db.execute(text(base_query), params)

        search_results = []
        for row in result.fetchall():
            # Find text snippets containing the query
            text = row.extracted_text
            query_lower = query.lower()
            text_lower = text.lower()

            snippets = []
            start = 0
            while True:
                pos = text_lower.find(query_lower, start)
                if pos == -1:
                    break

                # Extract snippet with context
                snippet_start = max(0, pos - 100)
                snippet_end = min(len(text), pos + len(query) + 100)
                snippet = text[snippet_start:snippet_end]

                snippets.append({
                    "text": snippet,
                    "position": pos,
                    "highlighted": True
                })

                start = pos + 1
                if len(snippets) >= 3:  # Limit snippets per document
                    break

            search_results.append({
                "document_id": str(row.id),
                "filename": row.filename,
                "version_number": row.version_number,
                "document_family_id": str(row.document_family_id) if row.document_family_id else None,
                "upload_date": row.upload_date.isoformat(),
                "is_latest_version": row.is_latest_version,
                "snippets": snippets,
                "match_count": len(snippets)
            })

        return {
            "query": query,
            "search_type": search_type,
            "family_id": family_id,
            "version_range": version_range,
            "results": search_results,
            "total_matches": len(search_results)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search across versions: {str(e)}"
        )


@router.get("/search/advanced")
async def advanced_cross_version_search(
    query: str,
    search_scope: str = "all",  # 'all', 'latest', 'specific_versions', 'date_range', 'family'
    family_ids: Optional[str] = None,  # Comma-separated family IDs
    version_ids: Optional[str] = None,  # Comma-separated version IDs
    date_from: Optional[str] = None,  # ISO date string
    date_to: Optional[str] = None,  # ISO date string
    version_range: Optional[str] = None,  # e.g., "1-3" or "latest-2"
    include_metadata: bool = True,
    include_snippets: bool = True,
    snippet_length: int = 200,
    max_snippets_per_doc: int = 3,
    similarity_threshold: float = 0.1,
    group_by_family: bool = False,
    sort_by: str = "relevance",  # 'relevance', 'date', 'version', 'family'
    sort_order: str = "desc",  # 'asc', 'desc'
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Advanced cross-version search with comprehensive filtering and grouping options."""

    try:
        # Check cache first for search results
        search_cache = get_search_cache()
        search_params_for_cache = {
            "search_scope": search_scope,
            "family_ids": family_ids,
            "version_ids": version_ids,
            "date_from": date_from,
            "date_to": date_to,
            "version_range": version_range,
            "include_metadata": include_metadata,
            "include_snippets": include_snippets,
            "snippet_length": snippet_length,
            "max_snippets_per_doc": max_snippets_per_doc,
            "group_by_family": group_by_family,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
            "offset": offset
        }

        cached_results = await search_cache.get_search_results(query, search_params_for_cache)
        if cached_results:
            return cached_results

        # Build dynamic query based on search scope
        base_query = """
        SELECT d.id, d.filename, d.extracted_text, d.version_number,
               d.document_family_id, d.upload_date, d.is_latest_version,
               d.version_description, d.processing_status, d.file_size,
               dv.version_tag, dv.author, dv.approval_status, dv.change_summary
        FROM documents d
        LEFT JOIN document_versions dv ON d.id = dv.document_id
        WHERE d.extracted_text IS NOT NULL
        AND d.processing_status = 'completed'
        AND d.extracted_text ILIKE :query
        """

        params = {
            "query": f"%{query}%",
            "limit": limit,
            "offset": offset
        }

        # Add scope-specific filters
        if search_scope == "latest":
            base_query += " AND d.is_latest_version = TRUE"

        elif search_scope == "specific_versions" and version_ids:
            version_list = [v.strip() for v in version_ids.split(",")]
            base_query += " AND d.id = ANY(:version_ids)"
            params["version_ids"] = version_list

        elif search_scope == "family" and family_ids:
            family_list = [f.strip() for f in family_ids.split(",")]
            base_query += " AND (d.document_family_id = ANY(:family_ids) OR d.id = ANY(:family_ids))"
            params["family_ids"] = family_list

        elif search_scope == "date_range":
            if date_from:
                base_query += " AND d.upload_date >= :date_from"
                params["date_from"] = date_from
            if date_to:
                base_query += " AND d.upload_date <= :date_to"
                params["date_to"] = date_to

        # Add version range filter
        if version_range:
            if "-" in version_range:
                if version_range.startswith("latest"):
                    # Handle "latest-N" format
                    try:
                        n = int(version_range.split("-")[1])
                        base_query += """
                        AND d.version_number >= (
                            SELECT MAX(version_number) - :version_offset
                            FROM documents d2
                            WHERE d2.document_family_id = d.document_family_id
                            OR (d.document_family_id IS NULL AND d2.id = d.id)
                        )
                        """
                        params["version_offset"] = n
                    except (ValueError, IndexError):
                        pass
                else:
                    # Handle "N-M" format
                    try:
                        start_version, end_version = map(int, version_range.split("-"))
                        base_query += " AND d.version_number BETWEEN :start_version AND :end_version"
                        params["start_version"] = start_version
                        params["end_version"] = end_version
                    except (ValueError, IndexError):
                        pass
            elif version_range.isdigit():
                base_query += " AND d.version_number = :specific_version"
                params["specific_version"] = int(version_range)

        # Add sorting
        if sort_by == "date":
            base_query += " ORDER BY d.upload_date"
        elif sort_by == "version":
            base_query += " ORDER BY d.version_number"
        elif sort_by == "family":
            base_query += " ORDER BY d.document_family_id, d.version_number"
        else:  # relevance (default)
            # Simple relevance scoring based on query frequency
            base_query = base_query.replace(
                "SELECT d.id",
                """SELECT d.id,
                (LENGTH(d.extracted_text) - LENGTH(REPLACE(LOWER(d.extracted_text), LOWER(:query_clean), '')))
                / LENGTH(:query_clean) as relevance_score, d.id"""
            )
            base_query += " ORDER BY relevance_score"
            params["query_clean"] = query.replace("%", "")

        if sort_order == "asc":
            base_query += " ASC"
        else:
            base_query += " DESC"

        base_query += " LIMIT :limit OFFSET :offset"

        # Execute search
        result = await db.execute(text(base_query), params)
        documents = result.fetchall()

        # Process results
        search_results = []
        family_groups = {}

        for doc in documents:
            # Extract snippets if requested
            snippets = []
            if include_snippets and doc.extracted_text:
                snippets = extract_advanced_snippets(
                    doc.extracted_text,
                    query,
                    snippet_length,
                    max_snippets_per_doc
                )

            # Calculate relevance score if not already done
            relevance_score = getattr(doc, 'relevance_score', None)
            if relevance_score is None:
                relevance_score = calculate_simple_relevance(doc.extracted_text, query)

            doc_result = {
                "document_id": str(doc.id),
                "filename": doc.filename,
                "version_number": doc.version_number,
                "document_family_id": str(doc.document_family_id) if doc.document_family_id else None,
                "is_latest_version": doc.is_latest_version,
                "upload_date": doc.upload_date.isoformat(),
                "relevance_score": float(relevance_score) if relevance_score else 0.0,
                "snippets": snippets,
                "snippet_count": len(snippets)
            }

            # Add metadata if requested
            if include_metadata:
                doc_result.update({
                    "version_description": doc.version_description,
                    "processing_status": doc.processing_status,
                    "file_size": doc.file_size,
                    "version_tag": doc.version_tag,
                    "author": doc.author,
                    "approval_status": doc.approval_status,
                    "change_summary": doc.change_summary
                })

            search_results.append(doc_result)

            # Group by family if requested
            if group_by_family:
                family_id = doc_result["document_family_id"] or doc_result["document_id"]
                if family_id not in family_groups:
                    family_groups[family_id] = {
                        "family_id": family_id,
                        "family_name": doc.filename.split("_v")[0] if "_v" in doc.filename else doc.filename,
                        "documents": [],
                        "total_versions": 0,
                        "latest_version": 0,
                        "total_snippets": 0,
                        "avg_relevance": 0.0
                    }

                family_groups[family_id]["documents"].append(doc_result)
                family_groups[family_id]["total_versions"] += 1
                family_groups[family_id]["latest_version"] = max(
                    family_groups[family_id]["latest_version"],
                    doc.version_number or 0
                )
                family_groups[family_id]["total_snippets"] += len(snippets)

        # Calculate family-level statistics
        if group_by_family:
            for family in family_groups.values():
                if family["documents"]:
                    family["avg_relevance"] = sum(
                        doc["relevance_score"] for doc in family["documents"]
                    ) / len(family["documents"])

        # Prepare response
        response_data = {
            "query": query,
            "search_scope": search_scope,
            "total_results": len(search_results),
            "results": search_results,
            "search_metadata": {
                "family_ids": family_ids.split(",") if family_ids else None,
                "version_ids": version_ids.split(",") if version_ids else None,
                "date_range": {
                    "from": date_from,
                    "to": date_to
                } if date_from or date_to else None,
                "version_range": version_range,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "include_metadata": include_metadata,
                "include_snippets": include_snippets
            }
        }

        if group_by_family:
            response_data["family_groups"] = list(family_groups.values())
            response_data["total_families"] = len(family_groups)

        # Cache the search results
        await search_cache.cache_search_results(query, search_params_for_cache, response_data)

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced search failed: {str(e)}"
        )


def extract_advanced_snippets(text: str, query: str, snippet_length: int, max_snippets: int) -> List[dict]:
    """Extract text snippets containing the search query with advanced features."""

    if not text or not query:
        return []

    text_lower = text.lower()
    query_lower = query.lower()
    snippets = []

    start = 0
    while len(snippets) < max_snippets:
        pos = text_lower.find(query_lower, start)
        if pos == -1:
            break

        # Calculate snippet boundaries
        snippet_start = max(0, pos - snippet_length // 2)
        snippet_end = min(len(text), pos + len(query) + snippet_length // 2)

        # Adjust to word boundaries
        if snippet_start > 0:
            space_pos = text.find(' ', snippet_start)
            if space_pos != -1 and space_pos < pos:
                snippet_start = space_pos + 1

        if snippet_end < len(text):
            space_pos = text.rfind(' ', snippet_start, snippet_end)
            if space_pos != -1 and space_pos > pos + len(query):
                snippet_end = space_pos

        snippet_text = text[snippet_start:snippet_end]

        # Avoid duplicate snippets
        if not any(abs(s["position"] - pos) < snippet_length // 4 for s in snippets):
            snippets.append({
                "text": snippet_text,
                "position": pos,
                "start_char": snippet_start,
                "end_char": snippet_end,
                "highlighted_text": highlight_query_in_text(snippet_text, query)
            })

        start = pos + 1

    return snippets


def highlight_query_in_text(text: str, query: str) -> str:
    """Highlight query terms in text."""

    import re

    # Escape special regex characters in query
    escaped_query = re.escape(query)

    # Create regex pattern for case-insensitive matching
    pattern = re.compile(f'({escaped_query})', re.IGNORECASE)

    # Replace matches with highlighted version
    highlighted = pattern.sub(r'<mark>\1</mark>', text)

    return highlighted


def calculate_simple_relevance(text: str, query: str) -> float:
    """Calculate simple relevance score based on query frequency and position."""

    if not text or not query:
        return 0.0

    text_lower = text.lower()
    query_lower = query.lower()

    # Count occurrences
    occurrences = text_lower.count(query_lower)
    if occurrences == 0:
        return 0.0

    # Calculate frequency score
    frequency_score = occurrences / len(text.split())

    # Boost score if query appears early in text
    first_occurrence = text_lower.find(query_lower)
    position_score = 1.0 - (first_occurrence / len(text)) if first_occurrence != -1 else 0.0

    # Combine scores
    relevance = (frequency_score * 0.7) + (position_score * 0.3)

    return min(relevance, 1.0)  # Cap at 1.0


@router.get("/performance/stats")
async def get_performance_stats(db: AsyncSession = Depends(get_db)):
    """Get performance statistics for the versioning system."""

    try:
        # Get cache statistics
        from app.core.cache.version_cache import get_version_cache
        cache = get_version_cache()
        cache_stats = cache.get_stats()

        # Get database statistics
        db_stats_result = await db.execute(
            text("""
            SELECT
                (SELECT COUNT(*) FROM documents) as total_documents,
                (SELECT COUNT(*) FROM documents WHERE processing_status = 'completed') as processed_documents,
                (SELECT COUNT(DISTINCT document_family_id) FROM documents WHERE document_family_id IS NOT NULL) as total_families,
                (SELECT COUNT(*) FROM document_changes) as total_changes,
                (SELECT COUNT(*) FROM search_history) as total_searches,
                (SELECT AVG(version_number) FROM documents WHERE version_number IS NOT NULL) as avg_versions_per_family
            """)
        )

        db_stats = db_stats_result.fetchone()

        # Get recent activity
        recent_activity_result = await db.execute(
            text("""
            SELECT
                COUNT(*) FILTER (WHERE upload_date >= NOW() - INTERVAL '24 hours') as documents_last_24h,
                COUNT(*) FILTER (WHERE upload_date >= NOW() - INTERVAL '7 days') as documents_last_7d,
                COUNT(*) FILTER (WHERE upload_date >= NOW() - INTERVAL '30 days') as documents_last_30d
            FROM documents
            """)
        )

        recent_activity = recent_activity_result.fetchone()

        # Get index usage statistics (PostgreSQL specific)
        index_stats_result = await db.execute(
            text("""
            SELECT
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('documents', 'document_versions', 'document_changes')
            ORDER BY idx_scan DESC
            LIMIT 10
            """)
        )

        index_stats = []
        for row in index_stats_result.fetchall():
            index_stats.append({
                "index_name": row.indexname,
                "scans": row.idx_scan,
                "tuples_read": row.idx_tup_read,
                "tuples_fetched": row.idx_tup_fetch
            })

        return {
            "cache_statistics": cache_stats,
            "database_statistics": {
                "total_documents": db_stats.total_documents,
                "processed_documents": db_stats.processed_documents,
                "total_families": db_stats.total_families,
                "total_changes": db_stats.total_changes,
                "total_searches": db_stats.total_searches,
                "avg_versions_per_family": float(db_stats.avg_versions_per_family) if db_stats.avg_versions_per_family else 0.0
            },
            "recent_activity": {
                "documents_last_24h": recent_activity.documents_last_24h,
                "documents_last_7d": recent_activity.documents_last_7d,
                "documents_last_30d": recent_activity.documents_last_30d
            },
            "index_performance": index_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@router.post("/performance/cache/clear")
async def clear_performance_cache():
    """Clear all performance caches."""

    try:
        from app.core.cache.version_cache import get_version_cache

        # Clear all caches
        cache = get_version_cache()
        await cache.clear()

        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/performance/cache/warm/{family_id}")
async def warm_cache_for_family(family_id: str, db: AsyncSession = Depends(get_db)):
    """Pre-warm cache for a specific document family."""

    try:
        from app.core.cache.version_cache import warm_cache_for_family

        await warm_cache_for_family(family_id, db)

        return {
            "message": f"Cache warmed for family {family_id}",
            "family_id": family_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to warm cache: {str(e)}"
        )