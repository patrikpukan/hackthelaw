from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.connection import async_session

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Document retrieval system for finding relevant chunks."""
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        limit: int = 5,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query with aggressive fallback strategies."""
        
        try:
            logger.info(f"Searching for query: '{query}' with document_ids: {document_ids}")
            
            # Strategy 1: Enhanced text-based search
            results = await self._text_based_search(
                query, limit, document_ids, chunk_types
            )
            
            if len(results) >= 3:  # Good enough
                logger.info(f"Text search found {len(results)} relevant chunks")
                return results
            
            # Strategy 2: Aggressive keyword search if insufficient results
            additional_results = await self._aggressive_keyword_search(
                query, limit, document_ids, chunk_types
            )
            
            # Combine and deduplicate
            combined_results = results + additional_results
            seen_ids = set()
            unique_results = []
            
            for result in combined_results:
                chunk_id = result['chunk']['id']
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_results.append(result)
                    if len(unique_results) >= limit:
                        break
            
            if len(unique_results) >= 2:  # Still acceptable
                logger.info(f"Combined search found {len(unique_results)} relevant chunks")
                return unique_results
            
            # Strategy 3: Desperate fallback - return ANY chunks from the document
            if document_ids:
                fallback_results = await self._fallback_any_chunks(document_ids, limit)
                if fallback_results:
                    logger.info(f"Fallback found {len(fallback_results)} chunks")
                    return fallback_results
            
            logger.warning(f"No results found for query: '{query}'")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    async def _text_based_search(
        self, 
        query: str, 
        limit: int,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Simple text-based search using PostgreSQL text search."""
        
        async with async_session() as session:
            # Build ULTRA-AGGRESSIVE search query - multiple strategies
            where_conditions = [
                "(e.chunk_text ILIKE :search_pattern OR e.chunk_text ILIKE :search_pattern2 OR e.chunk_text ILIKE :search_pattern3 OR e.chunk_text ILIKE :search_pattern4 OR e.chunk_text ILIKE :search_pattern5)"
            ]
            
            # Multiple search patterns for better matching + translation keywords
            query_words = query.lower().split()
            
            # English synonyms and related terms for better matching
            synonym_map = {
                "date": ["time", "deadline", "due", "when", "timing", "schedule", "period", "day", "month", "year"],
                "delivery": ["shipping", "transport", "send", "dispatch", "supply", "provide", "fulfill", "deliver"],
                "goods": ["items", "products", "merchandise", "cargo", "materials", "supplies", "commodities", "stuff"],
                "payment": ["pay", "fee", "cost", "price", "amount", "sum", "money", "compensation", "bill"],
                "contract": ["agreement", "deal", "arrangement", "terms", "conditions", "document"],
                "party": ["side", "participant", "entity", "organization", "company", "parties"],
                "penalty": ["fine", "charge", "fee", "punishment", "sanction", "forfeit"],
                "liability": ["responsibility", "obligation", "duty", "accountable", "liable"],
                "termination": ["end", "cancel", "expire", "conclude", "finish", "terminate"],
                "inspection": ["review", "check", "examine", "verify", "audit", "inspect"],
                "warranty": ["guarantee", "assurance", "protection", "coverage", "warrantee"],
                "compliance": ["adherence", "conformity", "following", "accordance", "comply"],
                "deadline": ["due", "date", "time", "limit", "cutoff", "expiry"],
                "obligation": ["duty", "responsibility", "requirement", "must", "shall"],
                "fulfill": ["complete", "satisfy", "meet", "accomplish", "deliver"],
                "amount": ["sum", "total", "cost", "price", "value", "quantity"],
                "clause": ["section", "provision", "term", "condition", "article"]
            }
            
            # Extract key terms from query (remove common question words)
            stop_words = {"what", "is", "the", "a", "an", "when", "where", "how", "why", "which", "who", "whom", "whose"}
            key_words = [word for word in query_words if word not in stop_words and len(word) > 2]
            
            # Build extended search terms
            extended_terms = []
            for word in key_words:
                extended_terms.append(word)
                if word in synonym_map:
                    extended_terms.extend(synonym_map[word][:5])  # Limit to top 5 synonyms
            
            # If no key words found, use original query
            if not key_words:
                key_words = query_words
                extended_terms = query_words
            
            # Create ULTRA-comprehensive search patterns (5 different strategies)
            params = {
                "search_pattern": f"%{query}%",  # Exact query
                "search_pattern2": f"%{'%'.join(key_words[:3])}%" if len(key_words) > 1 else f"%{query}%",  # Key words
                "search_pattern3": f"%{'%'.join(extended_terms[:3])}%" if len(extended_terms) > 1 else f"%{query}%",  # Synonyms
                "search_pattern4": f"%{' '.join(query.split()[:2])}%" if len(query.split()) > 1 else f"%{query}%",  # First 2 words
                "search_pattern5": f"%{query.split()[-1]}%" if len(query.split()) > 1 else f"%{query}%",  # Last word (often most important)
                "limit": limit
            }
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            if chunk_types:
                where_conditions.append("e.chunk_type = ANY(:chunk_types)")
                params["chunk_types"] = chunk_types
            
            # Simple text similarity search
            query_sql = f"""
                SELECT 
                    e.id as embedding_id,
                    e.chunk_id,
                    e.chunk_text,
                    e.chunk_type,
                    e.start_char,
                    e.end_char,
                    d.id as document_id,
                    d.filename,
                    d.file_type,
                    d.upload_date,
                    -- ULTRA-Enhanced similarity score with 5 patterns + loose matching
                    (
                        CASE 
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:search_pattern) THEN 1.0    -- Exact query match
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:search_pattern2) THEN 0.9   -- Key words match
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:search_pattern3) THEN 0.8   -- Synonyms match
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:search_pattern4) THEN 0.7   -- First words match
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:search_pattern5) THEN 0.6   -- Last word match
                            WHEN LOWER(e.chunk_text) LIKE LOWER(:loose_pattern) THEN 0.5     -- Loose pattern
                            ELSE 0.3  -- Any match at all gets some score
                        END
                    ) as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY similarity_score DESC, e.created_at DESC
                LIMIT :limit
            """
            
            # Add loose pattern for broader matching
            params["loose_pattern"] = f"%{' '.join(query.split()[:3])}%"
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            # Format results
            results = []
            for row in rows:
                chunk_data = {
                    "id": str(row.embedding_id),
                    "chunk_id": row.chunk_id,
                    "text": row.chunk_text,
                    "chunk_type": row.chunk_type,
                    "start_char": row.start_char,
                    "end_char": row.end_char
                }
                
                document_data = {
                    "id": str(row.document_id),
                    "filename": row.filename,
                    "file_type": row.file_type,
                    "upload_date": row.upload_date.isoformat() if row.upload_date else None
                }
                
                results.append({
                    "chunk": chunk_data,
                    "document": document_data,
                    "similarity_score": float(row.similarity_score)
                })
            
            return results
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        
        async with async_session() as session:
            query_sql = """
                SELECT 
                    e.id as embedding_id,
                    e.chunk_id,
                    e.chunk_text,
                    e.chunk_type,
                    e.start_char,
                    e.end_char,
                    d.filename,
                    d.file_type
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE d.id = :document_id
                ORDER BY e.start_char ASC
            """
            
            result = await session.execute(text(query_sql), {"document_id": document_id})
            rows = result.fetchall()
            
            chunks = []
            for row in rows:
                chunk = {
                    "id": str(row.embedding_id),
                    "chunk_id": row.chunk_id,
                    "text": row.chunk_text,
                    "chunk_type": row.chunk_type,
                    "start_char": row.start_char,
                    "end_char": row.end_char,
                    "document_filename": row.filename,
                    "document_type": row.file_type
                }
                chunks.append(chunk)
            
            return chunks
    
    async def find_similar_clauses(
        self, 
        clause_text: str, 
        clause_type: str = None,
        exclude_document_id: str = None,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find similar clauses across documents."""
        
        # Extract key terms from clause text for matching
        key_terms = self._extract_key_terms(clause_text)
        
        if not key_terms:
            return []
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": 10}
            
            # Build search pattern from key terms
            search_patterns = []
            for i, term in enumerate(key_terms[:3]):  # Use top 3 terms
                pattern_key = f"term_{i}"
                search_patterns.append(f"LOWER(e.chunk_text) LIKE LOWER(:{pattern_key})")
                params[pattern_key] = f"%{term}%"
            
            if search_patterns:
                where_conditions.append(f"({' OR '.join(search_patterns)})")
            
            if clause_type:
                where_conditions.append("e.chunk_type = :clause_type")
                params["clause_type"] = clause_type
            
            if exclude_document_id:
                where_conditions.append("d.id != :exclude_document_id")
                params["exclude_document_id"] = exclude_document_id
            
            query_sql = f"""
                SELECT 
                    e.chunk_text,
                    e.chunk_type,
                    d.id as document_id,
                    d.filename,
                    -- Simple similarity calculation
                    (
                        LENGTH(e.chunk_text) - LENGTH(REPLACE(LOWER(e.chunk_text), LOWER(:main_term), ''))
                    ) / LENGTH(:main_term)::float as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY similarity_score DESC
                LIMIT :limit
            """
            
            params["main_term"] = key_terms[0] if key_terms else ""
            
            result = await session.execute(text(query_sql), params)
            rows = result.fetchall()
            
            similar_clauses = []
            for row in rows:
                if row.similarity_score >= similarity_threshold:
                    similar_clauses.append({
                        "clause_text": row.chunk_text,
                        "clause_type": row.chunk_type,
                        "document_id": str(row.document_id),
                        "document_filename": row.filename,
                        "similarity_score": float(row.similarity_score)
                    })
            
            return similar_clauses
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for similarity matching."""
        
        # Simple keyword extraction (in production, use more sophisticated NLP)
        import re
        
        # Common legal stopwords to exclude
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Extract words (3+ characters, excluding common stopwords)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Return top 10 terms
    
    async def _debug_document_status(self, document_ids: List[str]):
        """Debug method to check document and embedding status."""
        
        async with async_session() as session:
            for doc_id in document_ids:
                # Check if document exists and its status
                doc_result = await session.execute(
                    text("SELECT id, filename, processing_status, extracted_text FROM documents WHERE id = :id"),
                    {"id": doc_id}
                )
                doc_row = doc_result.fetchone()
                
                if doc_row:
                    logger.info(f"Document {doc_id}: {doc_row.filename}, status: {doc_row.processing_status}")
                    logger.info(f"Has extracted text: {bool(doc_row.extracted_text)}")
                    
                    # Check embeddings/chunks
                    chunk_result = await session.execute(
                        text("SELECT COUNT(*) FROM embeddings WHERE document_id = :id"),
                        {"id": doc_id}
                    )
                    chunk_count = chunk_result.scalar()
                    logger.info(f"Document {doc_id} has {chunk_count} chunks in embeddings table")
                    
                    if chunk_count > 0:
                        # Show sample chunks
                        sample_result = await session.execute(
                            text("SELECT chunk_text FROM embeddings WHERE document_id = :id LIMIT 2"),
                            {"id": doc_id}
                        )
                        sample_chunks = sample_result.fetchall()
                        for i, chunk in enumerate(sample_chunks):
                            logger.info(f"Sample chunk {i+1}: {chunk.chunk_text[:100]}...")
                else:
                    logger.error(f"Document {doc_id} not found in database") 