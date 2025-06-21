"""
Enhanced Legal Search System

Advanced search capabilities specifically designed for legal documents with
legal-specific embeddings, semantic similarity improvements, and sophisticated
query understanding.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Different search strategies for legal documents."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LEGAL_CONCEPT_MATCHING = "legal_concept_matching"
    CLAUSE_TYPE_SEARCH = "clause_type_search"
    TEMPORAL_SEARCH = "temporal_search"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class SearchQuery:
    """Represents a search query with legal context."""
    query_text: str
    legal_context: Optional[str] = None
    document_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    clause_types: Optional[List[str]] = None
    parties_involved: Optional[List[str]] = None
    max_results: int = 10
    min_relevance_score: float = 0.3


@dataclass
class SearchResult:
    """Enhanced search result with legal metadata."""
    document_id: str
    chunk_id: str
    content: str
    relevance_score: float
    legal_concepts: List[str]
    clause_type: Optional[str]
    document_metadata: Dict[str, Any]
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    legal_significance: Optional[str] = None


class EnhancedLegalSearchEngine:
    """
    Advanced search engine specifically designed for legal documents.
    
    Features:
    - Legal-specific semantic embeddings
    - Clause-aware search strategies
    - Legal concept understanding
    - Context-aware result ranking
    - Multi-strategy search fusion
    """
    
    def __init__(self, embedding_model=None, vector_store=None):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.legal_concepts = self._initialize_legal_concepts()
        self.clause_patterns = self._initialize_clause_patterns()
        self.legal_synonyms = self._initialize_legal_synonyms()
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform enhanced legal search with multiple strategies.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        
        logger.info(f"Enhanced legal search: {query.query_text[:100]}...")
        
        # Analyze query to determine best search strategy
        query_analysis = await self._analyze_search_query(query)
        
        # Execute multiple search strategies
        search_results = []
        
        # 1. Semantic similarity search
        semantic_results = await self._semantic_search(query, query_analysis)
        search_results.extend(semantic_results)
        
        # 2. Legal concept matching
        concept_results = await self._legal_concept_search(query, query_analysis)
        search_results.extend(concept_results)
        
        # 3. Clause-specific search
        clause_results = await self._clause_type_search(query, query_analysis)
        search_results.extend(clause_results)
        
        # 4. Temporal search if date context exists
        if query.date_range or query_analysis.get('temporal_context'):
            temporal_results = await self._temporal_search(query, query_analysis)
            search_results.extend(temporal_results)
        
        # Merge and rank results
        final_results = await self._merge_and_rank_results(search_results, query, query_analysis)
        
        # Enhance results with legal context
        enhanced_results = await self._enhance_results_with_legal_context(final_results, query)
        
        logger.info(f"Enhanced search completed: {len(enhanced_results)} results")
        return enhanced_results[:query.max_results]
    
    async def _analyze_search_query(self, query: SearchQuery) -> Dict[str, Any]:
        """Analyze search query to understand legal intent and context."""
        
        query_text = query.query_text.lower()
        
        # Detect legal concepts in query
        detected_concepts = []
        for concept, keywords in self.legal_concepts.items():
            if any(keyword in query_text for keyword in keywords):
                detected_concepts.append(concept)
        
        # Detect clause types
        detected_clause_types = []
        for clause_type, patterns in self.clause_patterns.items():
            if any(pattern in query_text for pattern in patterns):
                detected_clause_types.append(clause_type)
        
        # Analyze query complexity
        query_complexity = self._assess_query_complexity(query_text)
        
        # Detect temporal context
        temporal_context = self._detect_temporal_context(query_text)
        
        # Determine primary search intent
        search_intent = self._determine_search_intent(query_text, detected_concepts)
        
        return {
            'detected_concepts': detected_concepts,
            'detected_clause_types': detected_clause_types,
            'query_complexity': query_complexity,
            'temporal_context': temporal_context,
            'search_intent': search_intent,
            'expanded_terms': self._expand_query_terms(query_text)
        }
    
    async def _semantic_search(self, query: SearchQuery, analysis: Dict[str, Any]) -> List[SearchResult]:
        """Perform semantic similarity search using legal embeddings."""
        
        # Expand query with legal synonyms
        expanded_query = self._expand_with_legal_synonyms(query.query_text, analysis)
        
        # Generate embeddings for expanded query
        if self.embedding_model:
            query_embedding = await self._generate_legal_embedding(expanded_query)
            
            # Search vector store
            if self.vector_store:
                similar_chunks = await self._vector_similarity_search(
                    query_embedding, 
                    query.max_results * 2  # Get more for filtering
                )
                
                # Convert to SearchResult objects
                results = []
                for chunk in similar_chunks:
                    if chunk.get('similarity_score', 0) >= query.min_relevance_score:
                        result = SearchResult(
                            document_id=chunk.get('document_id', ''),
                            chunk_id=chunk.get('chunk_id', ''),
                            content=chunk.get('content', ''),
                            relevance_score=chunk.get('similarity_score', 0),
                            legal_concepts=analysis.get('detected_concepts', []),
                            clause_type=chunk.get('clause_type'),
                            document_metadata=chunk.get('metadata', {})
                        )
                        results.append(result)
                
                return results
        
        # Fallback to keyword-based search if no embedding model
        return await self._keyword_fallback_search(query, analysis)
    
    async def _legal_concept_search(self, query: SearchQuery, analysis: Dict[str, Any]) -> List[SearchResult]:
        """Search based on legal concepts and terminology."""
        
        detected_concepts = analysis.get('detected_concepts', [])
        if not detected_concepts:
            return []
        
        results = []
        
        # Search for documents containing the same legal concepts
        for concept in detected_concepts:
            concept_keywords = self.legal_concepts.get(concept, [])
            
            # Simulate concept-based search
            # In a real implementation, this would query the database
            concept_results = await self._search_by_concept(concept, concept_keywords, query)
            results.extend(concept_results)
        
        return results
    
    async def _clause_type_search(self, query: SearchQuery, analysis: Dict[str, Any]) -> List[SearchResult]:
        """Search specifically within clause types."""
        
        clause_types = analysis.get('detected_clause_types', []) or query.clause_types or []
        if not clause_types:
            return []
        
        results = []
        
        for clause_type in clause_types:
            # Search within specific clause type
            clause_results = await self._search_within_clause_type(clause_type, query)
            results.extend(clause_results)
        
        return results
    
    async def _temporal_search(self, query: SearchQuery, analysis: Dict[str, Any]) -> List[SearchResult]:
        """Search with temporal context and date filtering."""
        
        temporal_context = analysis.get('temporal_context', {})
        date_range = query.date_range
        
        if not temporal_context and not date_range:
            return []
        
        # Implement temporal search logic
        # This would filter documents by date ranges and temporal relationships
        results = []
        
        # Simulate temporal search
        if date_range:
            start_date, end_date = date_range
            # Search documents within date range
            temporal_results = await self._search_by_date_range(start_date, end_date, query)
            results.extend(temporal_results)
        
        return results
    
    async def _merge_and_rank_results(
        self, 
        all_results: List[SearchResult], 
        query: SearchQuery, 
        analysis: Dict[str, Any]
    ) -> List[SearchResult]:
        """Merge results from different search strategies and rank them."""
        
        # Remove duplicates based on chunk_id
        unique_results = {}
        for result in all_results:
            key = f"{result.document_id}_{result.chunk_id}"
            if key not in unique_results or result.relevance_score > unique_results[key].relevance_score:
                unique_results[key] = result
        
        results_list = list(unique_results.values())
        
        # Enhanced ranking based on multiple factors
        for result in results_list:
            enhanced_score = self._calculate_enhanced_relevance_score(result, query, analysis)
            result.relevance_score = enhanced_score
        
        # Sort by enhanced relevance score
        results_list.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results_list
    
    async def _enhance_results_with_legal_context(
        self, 
        results: List[SearchResult], 
        query: SearchQuery
    ) -> List[SearchResult]:
        """Enhance search results with additional legal context."""
        
        for result in results:
            # Add legal significance assessment
            result.legal_significance = self._assess_legal_significance(result.content)
            
            # Add context before and after (if available)
            context = await self._get_surrounding_context(result.document_id, result.chunk_id)
            result.context_before = context.get('before')
            result.context_after = context.get('after')
            
            # Enhance legal concepts detection
            result.legal_concepts = self._detect_legal_concepts_in_content(result.content)
        
        return results
    
    def _initialize_legal_concepts(self) -> Dict[str, List[str]]:
        """Initialize legal concepts and their keywords."""
        
        return {
            'contract_formation': [
                'offer', 'acceptance', 'consideration', 'mutual assent', 'agreement'
            ],
            'termination': [
                'terminate', 'termination', 'end', 'expire', 'dissolution', 'breach'
            ],
            'payment_obligations': [
                'payment', 'pay', 'compensation', 'salary', 'fee', 'remuneration'
            ],
            'confidentiality': [
                'confidential', 'proprietary', 'trade secret', 'non-disclosure', 'nda'
            ],
            'liability': [
                'liability', 'liable', 'responsible', 'damages', 'indemnify', 'indemnification'
            ],
            'intellectual_property': [
                'intellectual property', 'patent', 'copyright', 'trademark', 'trade secret'
            ],
            'governing_law': [
                'governing law', 'jurisdiction', 'applicable law', 'venue', 'forum'
            ],
            'dispute_resolution': [
                'arbitration', 'mediation', 'dispute resolution', 'litigation', 'court'
            ]
        }
    
    def _initialize_clause_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for different clause types."""
        
        return {
            'termination_clause': [
                'termination', 'terminate', 'end this agreement', 'expire'
            ],
            'payment_clause': [
                'payment terms', 'compensation', 'salary', 'fee schedule'
            ],
            'confidentiality_clause': [
                'confidentiality', 'non-disclosure', 'proprietary information'
            ],
            'liability_clause': [
                'limitation of liability', 'damages', 'indemnification'
            ],
            'governing_law_clause': [
                'governing law', 'jurisdiction', 'applicable law'
            ]
        }
    
    def _initialize_legal_synonyms(self) -> Dict[str, List[str]]:
        """Initialize legal synonyms for query expansion."""
        
        return {
            'agreement': ['contract', 'accord', 'covenant', 'pact'],
            'terminate': ['end', 'conclude', 'dissolve', 'cancel'],
            'payment': ['compensation', 'remuneration', 'fee', 'salary'],
            'confidential': ['proprietary', 'private', 'secret', 'restricted'],
            'liable': ['responsible', 'accountable', 'answerable'],
            'damages': ['compensation', 'restitution', 'remedy', 'relief']
        }
