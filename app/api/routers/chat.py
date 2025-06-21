from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import List, Optional
import uuid

from app.db.connection import get_db
from app.db.models import ChatSession, ChatMessage
from app.core.chat.generator import RAGGenerator
from app.core.chat.iterative_rag import IterativeRAGGenerator
from app.core.chat.search_cache import get_search_cache
from app.core.chat.adaptive_limits import create_adaptive_limits_manager
from app.core.chat.enhanced_retriever import EnhancedDocumentRetriever
from app.core.chat.llm_client import LLMClientFactory
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def resolve_version_aware_documents(request: 'ChatRequest', db: AsyncSession) -> List[str]:
    """Resolve document IDs based on version-aware search parameters."""

    if request.specific_versions:
        # Use specific version IDs
        return request.specific_versions

    if request.document_ids:
        # Use provided document IDs, but filter by version mode
        if request.version_mode == "latest":
            # Get latest versions of the provided documents
            result = await db.execute(
                text("""
                SELECT DISTINCT COALESCE(d.document_family_id, d.id) as family_id
                FROM documents d
                WHERE d.id = ANY(:doc_ids)
                """),
                {"doc_ids": request.document_ids}
            )

            family_ids = [str(row.family_id) for row in result.fetchall()]

            # Get latest versions for each family
            latest_result = await db.execute(
                text("""
                SELECT d.id
                FROM documents d
                WHERE (d.document_family_id = ANY(:family_ids) OR d.id = ANY(:family_ids))
                AND d.is_latest_version = TRUE
                AND d.processing_status = 'processed'
                """),
                {"family_ids": family_ids}
            )

            return [str(row.id) for row in latest_result.fetchall()]
        else:
            # Return all provided documents
            return request.document_ids

    if request.family_ids:
        # Get documents from specific families
        if request.version_mode == "latest":
            result = await db.execute(
                text("""
                SELECT d.id
                FROM documents d
                WHERE (d.document_family_id = ANY(:family_ids) OR d.id = ANY(:family_ids))
                AND d.is_latest_version = TRUE
                AND d.processing_status = 'completed'
                """),
                {"family_ids": request.family_ids}
            )
        else:
            result = await db.execute(
                text("""
                SELECT d.id
                FROM documents d
                WHERE (d.document_family_id = ANY(:family_ids) OR d.id = ANY(:family_ids))
                AND d.processing_status = 'completed'
                """),
                {"family_ids": request.family_ids}
            )

        return [str(row.id) for row in result.fetchall()]

    # Default: get all documents based on version mode
    if request.version_mode == "latest":
        result = await db.execute(
            text("""
            SELECT d.id
            FROM documents d
            WHERE d.is_latest_version = TRUE
            AND d.processing_status = 'completed'
            """)
        )
    else:
        result = await db.execute(
            text("""
            SELECT d.id
            FROM documents d
            WHERE d.processing_status = 'completed'
            """)
        )

    return [str(row.id) for row in result.fetchall()]


async def add_version_context(response_data: dict, document_ids: List[str], db: AsyncSession) -> dict:
    """Add version context information to the response."""

    if not document_ids:
        return response_data

    # Get version information for the documents used
    result = await db.execute(
        text("""
        SELECT d.id, d.filename, d.version_number, d.document_family_id,
               d.is_latest_version, d.upload_date, dv.version_tag, dv.author
        FROM documents d
        LEFT JOIN document_versions dv ON d.id = dv.document_id
        WHERE d.id = ANY(:doc_ids)
        """),
        {"doc_ids": document_ids}
    )

    version_info = []
    for row in result.fetchall():
        version_info.append({
            "document_id": str(row.id),
            "filename": row.filename,
            "version_number": row.version_number,
            "family_id": str(row.document_family_id) if row.document_family_id else None,
            "is_latest": row.is_latest_version,
            "upload_date": row.upload_date.isoformat(),
            "version_tag": row.version_tag,
            "author": row.author
        })

    response_data["version_context"] = {
        "documents_used": version_info,
        "total_documents": len(version_info),
        "latest_only": all(doc["is_latest"] for doc in version_info),
        "families_represented": len(set(doc["family_id"] for doc in version_info if doc["family_id"]))
    }

    return response_data


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_provider: Optional[str] = None  # Will use configured default if None
    document_ids: Optional[List[str]] = []
    use_iterative_rag: bool = True
    max_iterations: int = 3
    # Version-aware search parameters
    version_mode: str = "latest"  # 'latest', 'all', 'specific', 'family'
    specific_versions: Optional[List[str]] = None  # Specific version IDs
    family_ids: Optional[List[str]] = None  # Document family IDs
    include_version_context: bool = True  # Include version info in responses

    # Multi-stage legal analysis parameters
    detailed_analysis_mode: bool = False  # Enable comprehensive legal document analysis
    enable_contradiction_detection: bool = True  # Detect contradictions across documents
    enable_temporal_tracking: bool = True  # Track legal positions over time
    enable_cross_document_reasoning: bool = True  # Enable cross-document relationship analysis
    max_analysis_tokens: int = 300000  # Maximum tokens for final analysis stage


class ChatResponse(BaseModel):
    response: str  # Changed from 'message' to 'response' for frontend compatibility
    session_id: str
    sources: List[dict] = []
    metadata: Optional[dict] = None

    # Legacy fields for backward compatibility
    message: Optional[str] = None
    model_used: Optional[str] = None
    usage: Optional[dict] = None
    rag_approach: str = "traditional"
    iterations_used: Optional[int] = None
    total_chunks_found: Optional[int] = None

    # Enhanced analysis results for detailed mode
    legal_analysis: Optional[dict] = None  # Comprehensive legal insights
    contradictions: Optional[List[dict]] = None  # Detected contradictions
    temporal_analysis: Optional[dict] = None  # Legal position evolution
    cross_document_relationships: Optional[List[dict]] = None  # Document relationships
    processing_stages: Optional[List[dict]] = None  # Multi-stage processing details
    token_usage: Optional[dict] = None  # Detailed token usage across stages


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Process a chat query using RAG."""
    
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id:
            # Create new session
            session = ChatSession()
            db.add(session)
            await db.commit()
            await db.refresh(session)
            session_id = str(session.id)
        else:
            # Verify session exists
            result = await db.execute(
                text("SELECT id FROM chat_sessions WHERE id = :id"),
                {"id": session_id}
            )
            if not result.scalar():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
        
        # Get conversation history
        history_result = await db.execute(
            text("""
                SELECT message_type, content 
                FROM chat_messages 
                WHERE session_id = :session_id 
                ORDER BY created_at ASC
            """),
            {"session_id": session_id}
        )
        
        conversation_history = []
        for row in history_result.fetchall():
            if row.message_type == "user":
                conversation_history.append({"user_message": row.content})
            elif row.message_type == "assistant" and conversation_history:
                conversation_history[-1]["assistant_message"] = row.content
        
        # Resolve document IDs based on version mode
        resolved_document_ids = await resolve_version_aware_documents(request, db)

        # Initialize LLM client with configured default
        from app.utils.config import settings
        default_provider = getattr(settings, 'llm_provider', 'groq')
        llm_client = LLMClientFactory.create_client(request.llm_provider or default_provider)

        # Choose RAG approach based on request
        if request.detailed_analysis_mode:
            # Use Multi-Stage RAG for comprehensive legal analysis
            from app.core.analysis.multi_stage_rag_generator import MultiStageRAGGenerator, MultiStageConfig

            config = MultiStageConfig(
                enable_contradiction_detection=request.enable_contradiction_detection,
                enable_temporal_tracking=request.enable_temporal_tracking,
                enable_cross_document_reasoning=request.enable_cross_document_reasoning,
                max_final_tokens=request.max_analysis_tokens
            )

            multi_stage_generator = MultiStageRAGGenerator(db_session=db, config=config)

            multi_stage_result = await multi_stage_generator.generate_comprehensive_analysis(
                query=request.message,
                document_ids=resolved_document_ids,
                session_id=session_id
            )

            # Convert multi-stage result to standard response format
            rag_response = {
                'response': multi_stage_result.final_response,
                'sources': [],  # Could be enhanced to extract sources from stages
                'metadata': multi_stage_result.processing_metadata,
                'legal_analysis': multi_stage_result.legal_analysis.__dict__ if multi_stage_result.legal_analysis else None,
                'contradictions': [c.__dict__ for c in multi_stage_result.legal_analysis.contradictions] if multi_stage_result.legal_analysis else None,
                'temporal_analysis': multi_stage_result.legal_analysis.temporal_analysis if multi_stage_result.legal_analysis else None,
                'cross_document_relationships': multi_stage_result.legal_analysis.cross_document_insights if multi_stage_result.legal_analysis else None,
                'processing_stages': [s.__dict__ for s in multi_stage_result.stage_results],
                'token_usage': multi_stage_result.total_token_usage,
                'fallback_used': multi_stage_result.fallback_used
            }

        elif request.use_iterative_rag:
            # Use Iterative RAG for complex queries
            retriever = EnhancedDocumentRetriever()
            rag_generator = IterativeRAGGenerator(llm_client=llm_client, retriever=retriever)

            rag_response = await rag_generator.generate_response(
                query=request.message,
                session_id=session_id,
                max_iterations=request.max_iterations,
                document_ids=resolved_document_ids
            )
        else:
            # Use Traditional RAG for simple queries
            rag_generator = RAGGenerator(llm_client=llm_client)

            rag_response = await rag_generator.generate_response(
                query=request.message,
                session_id=session_id,
                conversation_history=conversation_history,
                document_ids=resolved_document_ids
            )

        # Add version context if requested
        if request.include_version_context:
            rag_response = await add_version_context(rag_response, resolved_document_ids, db)
        
        # Prepare metadata
        metadata = {
            "model_used": rag_response.get("model_used"),
            "usage": rag_response.get("usage"),
            "rag_approach": "iterative" if request.use_iterative_rag else "traditional",
            "iterations_used": rag_response.get("iterations_used"),
            "search_stats": {
                "total_chunks": rag_response.get("total_chunks_found"),
                "documents_searched": len(resolved_document_ids),
                "iterations_used": rag_response.get("iterations_used"),
                "search_time": rag_response.get("search_time")
            },
            "version_info": {
                "version_mode": request.version_mode,
                "documents_resolved": len(resolved_document_ids),
                "original_document_count": len(request.document_ids or []),
                "include_version_context": request.include_version_context,
                "version_context": rag_response.get("version_context", {})
            }
        }
        
        return ChatResponse(
            response=rag_response.get("response", rag_response.get("content", "")),
            message=rag_response.get("response", rag_response.get("content", "")),  # Legacy compatibility
            session_id=session_id,
            sources=rag_response.get("sources", []),
            metadata=metadata,
            model_used=rag_response.get("model_used"),
            usage=rag_response.get("usage"),
            rag_approach="multi_stage" if request.detailed_analysis_mode else ("iterative" if request.use_iterative_rag else "traditional"),
            iterations_used=rag_response.get("iterations_used"),
            total_chunks_found=rag_response.get("total_chunks_found"),

            # Enhanced analysis results for detailed mode
            legal_analysis=rag_response.get('legal_analysis'),
            contradictions=rag_response.get('contradictions'),
            temporal_analysis=rag_response.get('temporal_analysis'),
            cross_document_relationships=rag_response.get('cross_document_relationships'),
            processing_stages=rag_response.get('processing_stages'),
            token_usage=rag_response.get('token_usage')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}"
        )


@router.post("/iterative", response_model=ChatResponse)
async def chat_iterative_rag(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """Process a chat query using Iterative RAG specifically."""
    
    try:
        # Force iterative RAG
        request.use_iterative_rag = True
        
        # Get or create session
        session_id = request.session_id
        if not session_id:
            # Create new session
            session = ChatSession()
            db.add(session)
            await db.commit()
            await db.refresh(session)
            session_id = str(session.id)
        else:
            # Verify session exists
            result = await db.execute(
                text("SELECT id FROM chat_sessions WHERE id = :id"),
                {"id": session_id}
            )
            if not result.scalar():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
        
        # Initialize LLM client
        llm_client = LLMClientFactory.create_client(request.llm_provider or "groq")
        
        # Use Iterative RAG with enhanced features
        retriever = EnhancedDocumentRetriever()
        
        rag_generator = IterativeRAGGenerator(
            llm_client=llm_client,
            retriever=retriever
        )
        
        rag_response = await rag_generator.generate_response(
            query=request.message,
            session_id=session_id,
            max_iterations=request.max_iterations,
            document_ids=request.document_ids or []
        )
        
        # Prepare enhanced metadata
        metadata = {
            "model_used": rag_response.get("model_used"),
            "usage": rag_response.get("usage"),
            "rag_approach": "iterative",
            "iterations_used": rag_response.get("iterations_used"),
            "search_stats": {
                "total_chunks": rag_response.get("total_chunks_found"),
                "documents_searched": len(request.document_ids or []),
                "iterations_used": rag_response.get("iterations_used"),
                "search_time": rag_response.get("search_time"),
                "cache_hits": rag_response.get("cache_hits", 0),
                "query_complexity": rag_response.get("query_complexity"),
                "efficiency_score": rag_response.get("efficiency_score")
            }
        }
        
        return ChatResponse(
            response=rag_response["content"],
            message=rag_response["content"],  # Legacy compatibility
            session_id=session_id,
            sources=rag_response.get("sources", []),
            metadata=metadata,
            model_used=rag_response.get("model_used"),
            usage=rag_response.get("usage"),
            rag_approach="iterative",
            iterations_used=rag_response.get("iterations_used"),
            total_chunks_found=rag_response.get("total_chunks_found")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Iterative RAG error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process iterative RAG query: {str(e)}"
        )


@router.get("/sessions")
async def list_chat_sessions(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """List chat sessions."""
    
    try:
        result = await db.execute(
            text("""
                SELECT id, session_name, created_at, updated_at
                FROM chat_sessions 
                ORDER BY updated_at DESC 
                OFFSET :skip LIMIT :limit
            """),
            {"skip": skip, "limit": limit}
        )
        
        sessions = []
        for row in result.fetchall():
            sessions.append({
                "id": str(row.id),
                "session_name": row.session_name,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat()
            })
        
        return {
            "sessions": sessions,
            "total": len(sessions),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}/messages")
async def get_chat_history(
    session_id: str,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get chat history for a session."""
    
    try:
        result = await db.execute(
            text("""
                SELECT message_type, content, created_at, metadata
                FROM chat_messages 
                WHERE session_id = :session_id
                ORDER BY created_at ASC
                OFFSET :skip LIMIT :limit
            """),
            {"session_id": session_id, "skip": skip, "limit": limit}
        )
        
        messages = []
        for row in result.fetchall():
            messages.append({
                "type": row.message_type,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
                "metadata": row.metadata
            })
        
        return {
            "session_id": session_id,
            "messages": messages,
            "total": len(messages),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat history: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a chat session and its messages."""
    
    try:
        # Check if session exists
        result = await db.execute(
            text("SELECT id FROM chat_sessions WHERE id = :id"),
            {"id": session_id}
        )
        
        if not result.scalar():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Delete session (cascade will delete messages)
        await db.execute(
            text("DELETE FROM chat_sessions WHERE id = :id"),
            {"id": session_id}
        )
        await db.commit()
        
        return {"message": "Chat session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat session: {str(e)}"
        )


@router.get("/providers")
async def get_available_providers():
    """Get list of available and configured LLM providers."""

    from app.utils.config import settings

    providers = []

    # Check Groq
    if getattr(settings, 'groq_api_key', None):
        providers.append({
            "id": "groq",
            "name": "Groq (Fast)",
            "description": "Fast inference with Llama models",
            "available": True
        })
    else:
        providers.append({
            "id": "groq",
            "name": "Groq (Fast)",
            "description": "Fast inference with Llama models",
            "available": False,
            "reason": "API key not configured"
        })

    # Check OpenAI
    if getattr(settings, 'openai_api_key', None):
        providers.append({
            "id": "openai",
            "name": "OpenAI",
            "description": "GPT models via OpenAI API",
            "available": True
        })
    else:
        providers.append({
            "id": "openai",
            "name": "OpenAI",
            "description": "GPT models via OpenAI API",
            "available": False,
            "reason": "API key not configured"
        })

    # Check Vertex AI
    if getattr(settings, 'vertex_ai_project_id', None):
        providers.append({
            "id": "vertexai",
            "name": "Vertex AI (Gemini)",
            "description": "Google's Gemini models via Vertex AI",
            "available": True,
            "model": getattr(settings, 'vertex_ai_model', 'gemini-2.5-pro'),
            "location": getattr(settings, 'vertex_ai_location', 'us-central1')
        })
    else:
        providers.append({
            "id": "vertexai",
            "name": "Vertex AI (Gemini)",
            "description": "Google's Gemini models via Vertex AI",
            "available": False,
            "reason": "Project ID not configured"
        })

    # Mock is always available
    providers.append({
        "id": "mock",
        "name": "Test Mode",
        "description": "Mock responses for testing",
        "available": True
    })

    return {
        "providers": providers,
        "default": getattr(settings, 'llm_provider', 'groq')
    }


@router.post("/test-llm")
async def test_llm_connection(llm_provider: str = "groq"):
    """Test LLM connection and response."""

    try:
        llm_client = LLMClientFactory.create_client(llm_provider)

        # Test with a simple query
        from app.core.chat.llm_client import ChatMessage
        test_messages = [
            ChatMessage(role="user", content="Hello! Please respond with a brief greeting.")
        ]

        response = await llm_client.chat_completion(test_messages)

        return {
            "status": "success",
            "provider": llm_provider,
            "model": response.model,
            "response": response.content,
            "usage": response.usage
        }

    except Exception as e:
        return {
            "status": "error",
            "provider": llm_provider,
            "error": str(e)
        }


class QueryAnalysisRequest(BaseModel):
    query: str


@router.post("/analyze-query")
async def analyze_query(request: QueryAnalysisRequest):
    """Analyze query and recommend RAG approach."""
    
    def classify_query_type(query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "definition", "meaning"]):
            return "definition"
        elif any(word in query_lower for word in ["who are", "parties", "stakeholders"]):
            return "entity_identification"
        elif any(word in query_lower for word in ["obligations", "requirements", "must", "shall"]):
            return "obligation_inquiry"
        elif any(word in query_lower for word in ["rights", "permissions", "allowed", "can"]):
            return "rights_inquiry"
        elif any(word in query_lower for word in ["subject", "topic", "about", "main", "key"]):
            return "conceptual_overview"
        else:
            return "general"
    
    def needs_iterative_approach(query: str) -> bool:
        """Determine if query needs iterative approach."""
        query_type = classify_query_type(query)
        return query_type in ["conceptual_overview", "general", "entity_identification"]
    
    def estimate_iterations(query: str) -> int:
        """Estimate how many iterations might be needed."""
        query_type = classify_query_type(query)
        
        if query_type in ["conceptual_overview", "general"]:
            return 3
        elif query_type in ["entity_identification", "obligation_inquiry"]:
            return 2
        else:
            return 1
    
    query_type = classify_query_type(request.query)
    recommended_approach = "iterative" if needs_iterative_approach(request.query) else "traditional"
    
    return {
        "query": request.query,
        "query_type": query_type,
        "recommended_approach": recommended_approach,
        "estimated_iterations": estimate_iterations(request.query),
        "reasoning": {
            "traditional_rag": "Best for exact definitions, direct quotes, specific clauses",
            "iterative_rag": "Best for conceptual questions, entity analysis, complex topics"
        }
    }


@router.get("/cache-stats")
async def get_cache_stats():
    """Get search cache performance statistics."""
    
    try:
        cache = get_search_cache()
        stats = cache.get_stats()
        top_queries = cache.get_top_queries(10)
        
        return {
            "cache_stats": {
                "total_queries": stats.total_queries,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "similar_hits": stats.similar_hits,
                "hit_rate": stats.hit_rate,
                "total_entries": stats.total_entries,
                "cache_size_mb": stats.cache_size_mb,
                "avg_results_per_query": stats.avg_results_per_query
            },
            "top_queries": [
                {"query": q, "hits": h, "last_accessed": t}
                for q, h, t in top_queries
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


@router.post("/cache/clear")
async def clear_cache():
    """Clear the search cache."""
    
    try:
        cache = get_search_cache()
        cache.clear()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post("/cache/optimize")
async def optimize_cache():
    """Optimize the search cache by removing low-value entries."""
    
    try:
        cache = get_search_cache()
        cache.optimize()
        
        return {"message": "Cache optimized successfully"}
        
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize cache")


@router.post("/query-complexity")
async def analyze_query_complexity(request: QueryAnalysisRequest):
    """Analyze query complexity and provide detailed resource estimates."""
    
    try:
        limits_manager = create_adaptive_limits_manager()
        complexity = limits_manager.analyze_query_complexity(request.query)
        initial_limits = limits_manager.get_initial_limits(request.query)
        
        # Estimate resource cost
        from app.core.chat.adaptive_limits import estimate_resource_cost
        resource_estimate = estimate_resource_cost(initial_limits, initial_limits.max_iterations)
        
        return {
            "query": request.query,
            "complexity": complexity.value,
            "adaptive_limits": {
                "max_iterations": initial_limits.max_iterations,
                "max_chunks_per_iteration": initial_limits.max_chunks_per_iteration,
                "min_confidence_threshold": initial_limits.min_confidence_threshold,
                "similarity_threshold": initial_limits.similarity_threshold,
                "early_stop_threshold": initial_limits.early_stop_threshold
            },
            "resource_estimate": resource_estimate,
            "recommendations": {
                "approach": "iterative" if complexity.value in ["complex", "very_complex"] else "traditional",
                "confidence": 0.9 if complexity.value == "very_complex" else 
                           0.7 if complexity.value == "complex" else
                           0.6 if complexity.value == "moderate" else 0.8
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing query complexity: {e}")
        raise HTTPException(status_code=500, detail="Query complexity analysis failed") 