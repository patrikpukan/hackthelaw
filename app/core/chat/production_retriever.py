from typing import List, Dict, Any, Optional
import logging
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.connection import async_session

logger = logging.getLogger(__name__)


class ProductionDocumentRetriever:
    """Production-ready document retrieval system optimized for English queries."""
    
    def __init__(self):
        self.similarity_threshold = 0.3  # Lower threshold for more results
        
        # Comprehensive English synonyms and related terms
        self.synonym_map = {
            'date': ['time', 'deadline', 'due', 'when', 'timing', 'schedule', 'period', 'day', 'month', 'year'],
            'delivery': ['shipping', 'transport', 'send', 'dispatch', 'supply', 'provide', 'fulfill'],
            'goods': ['items', 'products', 'merchandise', 'cargo', 'materials', 'supplies', 'commodities'],
            'payment': ['pay', 'fee', 'cost', 'price', 'amount', 'sum', 'money', 'compensation'],
            'contract': ['agreement', 'deal', 'arrangement', 'terms', 'conditions'],
            'party': ['side', 'participant', 'entity', 'organization', 'company'],
            'penalty': ['fine', 'charge', 'fee', 'punishment', 'sanction'],
            'liability': ['responsibility', 'obligation', 'duty', 'accountable'],
            'termination': ['end', 'cancel', 'expire', 'conclude', 'finish'],
            'renewal': ['extend', 'continue', 'restart', 'refresh'],
            'inspection': ['review', 'check', 'examine', 'verify', 'audit'],
            'warranty': ['guarantee', 'assurance', 'protection', 'coverage'],
            'compliance': ['adherence', 'conformity', 'following', 'accordance'],
        }
        
        # Legal document patterns for better extraction
        self.legal_patterns = {
            'dates': [
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b',
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{2,4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
                r'\bwithin\s+\d+\s+(?:days?|months?|years?|weeks?)\b',
                r'\b\d+\s+(?:days?|months?|years?|weeks?)\s+(?:after|before|from)\b'
            ],
            'amounts': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|eur|euros?|pounds?|gbp)\b',
                r'\b(?:hundred|thousand|million|billion)\s+(?:dollars?|euros?|pounds?)\b'
            ],
            'obligations': [
                r'\b(?:shall|must|will|required to|obligated to|responsible for)\b',
                r'\b(?:deliver|provide|pay|complete|fulfill|perform)\b',
                r'\bno later than\b',
                r'\bby the date\b'
            ]
        }
        
        # Stop words - common words that don't add semantic meaning
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'if', 'then', 'else', 'so', 'because', 'as', 'since', 'while', 'until', 'unless'
        }
    
    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        limit: int = 5,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Production-ready retrieval with multiple fallback strategies."""
        
        try:
            logger.info(f"Production search for query: '{query}' with document_ids: {document_ids}")
            
            # Strategy 1: Smart semantic search
            results = await self._semantic_search(query, limit, document_ids, chunk_types)
            
            if len(results) >= 3:  # Good enough results
                logger.info(f"Semantic search found {len(results)} results")
                return results
            
            # Strategy 2: Aggressive keyword matching
            additional_results = await self._aggressive_keyword_search(query, limit, document_ids, chunk_types)
            results.extend(additional_results)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_results = []
            for result in results:
                chunk_id = result['chunk']['id']
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_results.append(result)
                    if len(unique_results) >= limit:
                        break
            
            if len(unique_results) >= 2:  # Still good
                logger.info(f"Combined search found {len(unique_results)} results")
                return unique_results
            
            # Strategy 3: Desperate fallback - find anything related
            fallback_results = await self._fallback_search(query, limit, document_ids, chunk_types)
            unique_results.extend(fallback_results)
            
            # Final deduplication
            final_seen = set()
            final_results = []
            for result in unique_results:
                chunk_id = result['chunk']['id']
                if chunk_id not in final_seen:
                    final_seen.add(chunk_id)
                    final_results.append(result)
                    if len(final_results) >= limit:
                        break
            
            logger.info(f"Final search found {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in production retrieval: {e}")
            return []
    
    async def _semantic_search(
        self, 
        query: str, 
        limit: int,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Smart semantic search with intent detection."""
        
        # Extract meaningful terms
        query_terms = self._extract_meaningful_terms(query)
        logger.info(f"Extracted terms: {query_terms}")
        
        if not query_terms:
            return []
        
        # Generate search variations
        search_patterns = self._generate_search_patterns(query_terms)
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": limit}
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            if chunk_types:
                where_conditions.append("e.chunk_type = ANY(:chunk_types)")
                params["chunk_types"] = chunk_types
            
            # Build search conditions
            search_conditions = []
            param_count = 0
            
            # Exact phrase (highest priority)
            search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
            params[f"pattern_{param_count}"] = f"%{query}%"
            param_count += 1
            
            # All terms present
            if len(query_terms) > 1:
                all_terms_condition = " AND ".join([f"e.chunk_text ILIKE :pattern_{param_count + i}" 
                                                   for i in range(len(query_terms))])
                search_conditions.append(f"({all_terms_condition})")
                for i, term in enumerate(query_terms):
                    params[f"pattern_{param_count + i}"] = f"%{term}%"
                param_count += len(query_terms)
            
            # Any important term
            for pattern in search_patterns[:10]:  # Limit patterns
                search_conditions.append(f"e.chunk_text ILIKE :pattern_{param_count}")
                params[f"pattern_{param_count}"] = f"%{pattern}%"
                param_count += 1
            
            base_where = " AND ".join(where_conditions) if where_conditions else "1=1"
            search_where = " OR ".join(search_conditions)
            
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
                    -- Advanced similarity scoring
                    (
                        CASE 
                            WHEN e.chunk_text ILIKE :pattern_0 THEN 1.0
                            WHEN {' AND '.join([f"e.chunk_text ILIKE :pattern_{i+1}" for i in range(len(query_terms))])} THEN 0.9
                            ELSE 0.7
                        END
                        + 
                        -- Length bonus (shorter chunks often more relevant)
                        CASE 
                            WHEN LENGTH(e.chunk_text) < 200 THEN 0.1
                            WHEN LENGTH(e.chunk_text) < 500 THEN 0.05
                            ELSE 0
                        END
                    ) as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {base_where} AND ({search_where})
                ORDER BY similarity_score DESC, e.created_at DESC
                LIMIT :limit
            """
            
            result = await session.execute(text(query_sql), params)
            return self._format_results(result.fetchall())
    
    async def _aggressive_keyword_search(
        self, 
        query: str, 
        limit: int,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Aggressive keyword matching for when semantic search fails."""
        
        # Extract every possible keyword
        words = re.findall(r'\b\w{3,}\b', query.lower())  # Words 3+ chars
        keywords = [word for word in words if word not in self.stop_words]
        
        if not keywords:
            return []
        
        # Add synonyms for each keyword
        expanded_keywords = set(keywords)
        for keyword in keywords:
            if keyword in self.synonym_map:
                expanded_keywords.update(self.synonym_map[keyword])
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": limit}
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            # Create OR conditions for all keywords
            keyword_conditions = []
            for i, keyword in enumerate(list(expanded_keywords)[:20]):  # Limit to 20
                keyword_conditions.append(f"e.chunk_text ILIKE :kw_{i}")
                params[f"kw_{i}"] = f"%{keyword}%"
            
            base_where = " AND ".join(where_conditions) if where_conditions else "1=1"
            keyword_where = " OR ".join(keyword_conditions)
            
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
                    -- Count keyword matches for scoring
                    (
                        {' + '.join([f"CASE WHEN e.chunk_text ILIKE :kw_{i} THEN 1 ELSE 0 END" 
                                   for i in range(len(list(expanded_keywords)[:20]))])}
                    ) * 0.1 as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {base_where} AND ({keyword_where})
                ORDER BY similarity_score DESC, e.created_at DESC
                LIMIT :limit
            """
            
            result = await session.execute(text(query_sql), params)
            return self._format_results(result.fetchall())
    
    async def _fallback_search(
        self, 
        query: str, 
        limit: int,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Last resort - find any chunks that might be relevant."""
        
        # Extract any meaningful characters
        chars = re.findall(r'\w{2,}', query.lower())
        if not chars:
            # Absolute fallback - return any chunks
            return await self._get_any_chunks(limit, document_ids)
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": limit}
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            # Very loose matching
            char_conditions = []
            for i, char in enumerate(chars[:10]):  # Limit chars
                char_conditions.append(f"e.chunk_text ILIKE :char_{i}")
                params[f"char_{i}"] = f"%{char}%"
            
            base_where = " AND ".join(where_conditions) if where_conditions else "1=1"
            char_where = " OR ".join(char_conditions) if char_conditions else "1=1"
            
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
                    0.3 as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {base_where} AND ({char_where})
                ORDER BY e.created_at DESC
                LIMIT :limit
            """
            
            result = await session.execute(text(query_sql), params)
            return self._format_results(result.fetchall())
    
    async def _get_any_chunks(
        self, 
        limit: int,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Emergency fallback - return any available chunks."""
        
        async with async_session() as session:
            where_conditions = []
            params = {"limit": limit}
            
            if document_ids:
                where_conditions.append("d.id = ANY(:document_ids)")
                params["document_ids"] = document_ids
            
            base_where = " AND ".join(where_conditions) if where_conditions else "1=1"
            
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
                    0.2 as similarity_score
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                WHERE {base_where}
                ORDER BY LENGTH(e.chunk_text) DESC  -- Prefer longer, more informative chunks
                LIMIT :limit
            """
            
            result = await session.execute(text(query_sql), params)
            return self._format_results(result.fetchall())
    
    def _extract_meaningful_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        
        # Clean and split
        words = re.findall(r'\b\w{3,}\b', query.lower())
        
        # Remove stop words
        meaningful_words = [word for word in words if word not in self.stop_words]
        
        # Prioritize legal/business terms
        legal_terms = ['contract', 'delivery', 'payment', 'date', 'goods', 'party', 'liability', 'penalty']
        prioritized = []
        
        # Add legal terms first
        for word in meaningful_words:
            if word in legal_terms or any(word in self.synonym_map.get(term, []) for term in legal_terms):
                prioritized.append(word)
        
        # Add other meaningful terms
        for word in meaningful_words:
            if word not in prioritized:
                prioritized.append(word)
        
        return prioritized[:5]  # Limit to top 5 terms
    
    def _generate_search_patterns(self, terms: List[str]) -> List[str]:
        """Generate search patterns from terms."""
        
        patterns = []
        
        # Individual terms
        patterns.extend(terms)
        
        # Add synonyms
        for term in terms:
            if term in self.synonym_map:
                patterns.extend(self.synonym_map[term][:3])  # Top 3 synonyms
        
        # Combined terms (pairs)
        if len(terms) > 1:
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    patterns.append(f"{terms[i]} {terms[j]}")
        
        return patterns
    
    def _format_results(self, rows) -> List[Dict[str, Any]]:
        """Format database results into standard format."""
        
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