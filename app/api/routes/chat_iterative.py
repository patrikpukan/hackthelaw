from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.core.chat.iterative_rag import IterativeRAGGenerator
from app.core.chat.llm_client import LLMClientFactory
from app.core.chat.enhanced_retriever import EnhancedDocumentRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Iterative Chat"])


class IterativeChatRequest(BaseModel):
    """Request model for iterative chat."""
    query: str
    session_id: Optional[str] = None
    max_iterations: int = 3
    max_chunks_per_iteration: int = 5
    document_ids: Optional[List[str]] = None


class IterativeChatResponse(BaseModel):
    """Response model for iterative chat."""
    content: str
    sources: List[Dict[str, Any]]
    query: str
    iterations_used: int
    total_chunks_found: int
    model_used: Optional[str] = None
    timestamp: str
    error: bool = False


@router.post("/iterative", response_model=IterativeChatResponse)
async def iterative_chat(request: IterativeChatRequest):
    """
    Enhanced chat endpoint using Iterative RAG.
    
    This endpoint implements:
    1. Query rewriting for better document search
    2. Iterative retrieval with self-assessment
    3. Adaptive search strategies
    4. Comprehensive response generation
    """
    
    try:
        logger.info(f"Iterative chat request: {request.query[:100]}...")
        
        # Initialize components
        llm_client = LLMClientFactory.get_default_client()
        retriever = EnhancedDocumentRetriever()
        rag_generator = IterativeRAGGenerator(llm_client, retriever)
        
        # Generate response using iterative approach
        result = await rag_generator.generate_response(
            query=request.query,
            session_id=request.session_id,
            max_iterations=request.max_iterations,
            max_chunks_per_iteration=request.max_chunks_per_iteration,
            document_ids=request.document_ids
        )
        
        # Convert to response model
        return IterativeChatResponse(
            content=result["content"],
            sources=result["sources"],
            query=result["query"],
            iterations_used=result.get("iterations_used", 1),
            total_chunks_found=result.get("total_chunks_found", 0),
            model_used=result.get("model_used"),
            timestamp=result["timestamp"],
            error=result.get("error", False)
        )
        
    except Exception as e:
        logger.error(f"Error in iterative chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing iterative chat request: {str(e)}"
        )


@router.get("/compare")
async def compare_approaches():
    """
    Compare different RAG approaches.
    """
    
    return {
        "approaches": {
            "traditional_rag": {
                "description": "Single search query, direct retrieval",
                "pros": ["Fast", "Simple", "Predictable"],
                "cons": ["Limited to exact phrases", "Poor conceptual understanding", "Single chance search"]
            },
            "iterative_rag": {
                "description": "Query rewriting, multiple iterations, self-assessment",
                "pros": [
                    "Handles conceptual questions",
                    "Multiple search strategies", 
                    "Self-correcting",
                    "Better coverage"
                ],
                "cons": ["Slower", "More complex", "Higher API costs"]
            }
        },
        "use_cases": {
            "traditional_rag": [
                "Exact clause lookup",
                "Specific term definitions",
                "Direct phrase matching"
            ],
            "iterative_rag": [
                "\"What is the main subject of this document?\"",
                "\"Who are the key stakeholders?\"",
                "\"What are the compliance requirements?\"",
                "\"How does this agreement protect intellectual property?\""
            ]
        }
    }


class QueryAnalysisRequest(BaseModel):
    """Request for query analysis."""
    query: str


@router.post("/analyze-query")
async def analyze_query(request: QueryAnalysisRequest):
    """
    Analyze a query and show how it would be processed.
    """
    
    try:
        # Initialize components
        llm_client = LLMClientFactory.get_default_client()
        rag_generator = IterativeRAGGenerator(llm_client, None)
        
        # Analyze query
        search_queries = await rag_generator._rewrite_query(request.query)
        
        return {
            "original_query": request.query,
            "rewritten_queries": search_queries,
            "query_type": _classify_query_type(request.query),
            "expected_iterations": _estimate_iterations(request.query),
            "recommended_approach": "iterative_rag" if _needs_iterative_approach(request.query) else "traditional_rag"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing query: {str(e)}"
        )


def _classify_query_type(query: str) -> str:
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
    elif any(word in query_lower for word in ["subject", "topic", "about", "main"]):
        return "conceptual_overview"
    else:
        return "general"


def _estimate_iterations(query: str) -> int:
    """Estimate how many iterations might be needed."""
    
    query_type = _classify_query_type(query)
    
    if query_type in ["conceptual_overview", "general"]:
        return 2-3
    elif query_type in ["entity_identification", "obligation_inquiry"]:
        return 1-2
    else:
        return 1


def _needs_iterative_approach(query: str) -> bool:
    """Determine if query needs iterative approach."""
    
    query_type = _classify_query_type(query)
    return query_type in ["conceptual_overview", "general", "entity_identification"] 