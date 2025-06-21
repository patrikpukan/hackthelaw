"""
Legal Memory System

An intelligent legal memory system that provides sophisticated recall capabilities
for legal documents, precedents, and clause evolution tracking.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of legal memory queries."""
    PRECEDENT_SEARCH = "precedent_search"
    SIMILAR_TERMS = "similar_terms"
    CLAUSE_EVOLUTION = "clause_evolution"
    CONTRADICTION_CHECK = "contradiction_check"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class LegalPrecedent:
    """Represents a legal precedent found in the document corpus."""
    id: str
    document_id: str
    clause_text: str
    clause_type: str
    context: str
    relevance_score: float
    date: datetime
    parties_involved: List[str]
    outcome: Optional[str] = None
    tags: List[str] = None


@dataclass
class MemoryQuery:
    """Represents a query to the legal memory system."""
    query_text: str
    query_type: QueryType
    context: Dict[str, Any]
    filters: Dict[str, Any]
    max_results: int = 10


@dataclass
class MemoryResponse:
    """Response from the legal memory system."""
    query: MemoryQuery
    results: List[Dict[str, Any]]
    confidence_score: float
    reasoning: str
    suggestions: List[str]
    processing_time_ms: int


class LegalMemorySystem:
    """
    Intelligent legal memory system that provides sophisticated recall capabilities.
    
    Key Features:
    - Precedent identification and matching
    - Similar terms detection across document corpus
    - Clause evolution tracking over time
    - Intelligent query understanding and response
    - Context-aware legal reasoning
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.memory_index = {}
        self.clause_patterns = self._initialize_clause_patterns()
        self.legal_concepts = self._initialize_legal_concepts()
        
    async def query_memory(self, query: MemoryQuery) -> MemoryResponse:
        """
        Query the legal memory system with intelligent understanding.
        
        Args:
            query: MemoryQuery object containing the query details
            
        Returns:
            MemoryResponse with relevant results and reasoning
        """
        
        start_time = datetime.now()
        logger.info(f"Processing legal memory query: {query.query_text[:100]}...")
        
        # Parse and understand the query
        parsed_query = await self._parse_legal_query(query)
        
        # Route to appropriate handler based on query type
        if query.query_type == QueryType.PRECEDENT_SEARCH:
            results = await self._search_precedents(parsed_query)
        elif query.query_type == QueryType.SIMILAR_TERMS:
            results = await self._find_similar_terms(parsed_query)
        elif query.query_type == QueryType.CLAUSE_EVOLUTION:
            results = await self._track_clause_evolution(parsed_query)
        elif query.query_type == QueryType.CONTRADICTION_CHECK:
            results = await self._check_contradictions(parsed_query)
        elif query.query_type == QueryType.RISK_ASSESSMENT:
            results = await self._assess_risks(parsed_query)
        else:
            results = await self._general_search(parsed_query)
        
        # Generate reasoning and suggestions
        reasoning = self._generate_reasoning(query, results)
        suggestions = self._generate_suggestions(query, results)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(results, query)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return MemoryResponse(
            query=query,
            results=results,
            confidence_score=confidence,
            reasoning=reasoning,
            suggestions=suggestions,
            processing_time_ms=processing_time
        )
    
    async def answer_complex_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Answer complex legal questions like 'Have we agreed to similar terms before?'
        
        Args:
            question: Natural language question
            context: Optional context information
            
        Returns:
            Comprehensive answer with supporting evidence
        """
        
        # Analyze the question to determine intent
        query_analysis = self._analyze_question_intent(question)
        
        # Create appropriate memory query
        memory_query = MemoryQuery(
            query_text=question,
            query_type=query_analysis['primary_intent'],
            context=context or {},
            filters=query_analysis.get('filters', {}),
            max_results=query_analysis.get('max_results', 10)
        )
        
        # Query the memory system
        memory_response = await self.query_memory(memory_query)
        
        # Generate comprehensive answer
        answer = await self._generate_comprehensive_answer(question, memory_response, context)
        
        return answer
    
    async def _parse_legal_query(self, query: MemoryQuery) -> Dict[str, Any]:
        """Parse and understand a legal query."""
        
        query_text = query.query_text.lower()
        
        # Extract key legal concepts
        concepts = []
        for concept, keywords in self.legal_concepts.items():
            if any(keyword in query_text for keyword in keywords):
                concepts.append(concept)
        
        # Extract entities (simplified)
        entities = self._extract_query_entities(query_text)
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query, concepts)
        
        return {
            'original_query': query.query_text,
            'concepts': concepts,
            'entities': entities,
            'search_strategy': search_strategy,
            'filters': query.filters
        }
    
    async def _search_precedents(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for legal precedents based on the query."""
        
        # This would typically query a database of legal precedents
        # For now, we'll simulate with a basic implementation
        
        precedents = []
        concepts = parsed_query.get('concepts', [])
        
        # Simulate precedent search
        if 'employment' in concepts:
            precedents.append({
                'type': 'precedent',
                'document_id': 'emp_001',
                'clause_text': 'Employee termination with 30 days notice',
                'relevance_score': 0.85,
                'date': '2023-06-15',
                'context': 'Employment agreement with similar role'
            })
        
        if 'payment' in concepts:
            precedents.append({
                'type': 'precedent',
                'document_id': 'pay_002',
                'clause_text': 'Monthly salary payment on the 15th of each month',
                'relevance_score': 0.78,
                'date': '2023-08-20',
                'context': 'Service agreement with similar payment structure'
            })
        
        return precedents
    
    async def _find_similar_terms(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar terms across the document corpus."""
        
        similar_terms = []
        
        # Extract key terms from query
        query_terms = parsed_query['original_query'].split()
        
        # Simulate finding similar terms
        for term in query_terms:
            if len(term) > 3:  # Skip short words
                similar_terms.append({
                    'type': 'similar_term',
                    'original_term': term,
                    'similar_terms': [f"{term}_variant1", f"{term}_variant2"],
                    'documents_found': ['doc_001', 'doc_002'],
                    'frequency': 5
                })
        
        return similar_terms
    
    async def _track_clause_evolution(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Track how clauses have evolved over time."""
        
        evolution_data = []
        
        # Simulate clause evolution tracking
        concepts = parsed_query.get('concepts', [])
        
        if 'termination' in concepts:
            evolution_data.append({
                'type': 'clause_evolution',
                'clause_type': 'termination',
                'timeline': [
                    {
                        'date': '2022-01-01',
                        'version': 'v1',
                        'text': 'Termination with 60 days notice',
                        'document': 'contract_v1.pdf'
                    },
                    {
                        'date': '2023-01-01',
                        'version': 'v2',
                        'text': 'Termination with 30 days notice',
                        'document': 'contract_v2.pdf'
                    }
                ],
                'trend': 'notice_period_decreased',
                'impact_assessment': 'Reduced employee protection'
            })
        
        return evolution_data
    
    async def _check_contradictions(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for contradictions related to the query."""
        
        contradictions = []
        
        # Simulate contradiction detection
        contradictions.append({
            'type': 'contradiction',
            'contradiction_type': 'payment_terms',
            'document1': 'contract_a.pdf',
            'document2': 'contract_b.pdf',
            'description': 'Different payment schedules specified',
            'severity': 'medium',
            'recommendation': 'Standardize payment terms across agreements'
        })
        
        return contradictions
    
    async def _assess_risks(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess legal risks related to the query."""
        
        risks = []
        concepts = parsed_query.get('concepts', [])
        
        if 'liability' in concepts:
            risks.append({
                'type': 'risk_assessment',
                'risk_category': 'liability',
                'risk_level': 'high',
                'description': 'Unlimited liability exposure in current agreements',
                'mitigation_suggestions': [
                    'Add liability caps',
                    'Include indemnification clauses',
                    'Consider insurance requirements'
                ]
            })
        
        return risks
    
    async def _general_search(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform general search across legal documents."""
        
        # Simulate general search results
        return [
            {
                'type': 'general_result',
                'document_id': 'doc_001',
                'relevance_score': 0.75,
                'snippet': 'Relevant text snippet from document...',
                'context': 'General legal document search result'
            }
        ]

    def _analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the intent of a natural language question."""

        question_lower = question.lower()

        # Intent patterns
        intent_patterns = {
            QueryType.PRECEDENT_SEARCH: [
                'have we', 'did we', 'previous', 'before', 'similar', 'precedent'
            ],
            QueryType.SIMILAR_TERMS: [
                'similar terms', 'same conditions', 'equivalent', 'comparable'
            ],
            QueryType.CLAUSE_EVOLUTION: [
                'how has', 'evolution', 'changed', 'over time', 'history'
            ],
            QueryType.CONTRADICTION_CHECK: [
                'conflict', 'contradict', 'inconsistent', 'different'
            ],
            QueryType.RISK_ASSESSMENT: [
                'risk', 'danger', 'liability', 'exposure', 'vulnerable'
            ]
        }

        # Determine primary intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in question_lower)
            intent_scores[intent] = score

        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else QueryType.PRECEDENT_SEARCH

        # Extract filters from question
        filters = {}
        if 'employment' in question_lower:
            filters['document_type'] = 'employment'
        if 'payment' in question_lower or 'salary' in question_lower:
            filters['clause_type'] = 'payment'

        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'filters': filters,
            'max_results': 10
        }

    def _extract_query_entities(self, query_text: str) -> Dict[str, List[str]]:
        """Extract entities from query text."""
        import re

        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        monetary_amounts = re.findall(money_pattern, query_text)

        # Extract dates
        date_pattern = r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
        dates = re.findall(date_pattern, query_text, re.IGNORECASE)

        # Extract company-like terms
        company_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Inc\.|LLC|Corp\.|Corporation|Company)\b'
        companies = re.findall(company_pattern, query_text)

        return {
            'monetary_amounts': monetary_amounts,
            'dates': dates,
            'companies': companies
        }

    def _determine_search_strategy(self, query: MemoryQuery, concepts: List[str]) -> str:
        """Determine the best search strategy for the query."""

        if query.query_type == QueryType.PRECEDENT_SEARCH:
            return 'semantic_similarity'
        elif query.query_type == QueryType.SIMILAR_TERMS:
            return 'term_matching'
        elif query.query_type == QueryType.CLAUSE_EVOLUTION:
            return 'temporal_analysis'
        else:
            return 'hybrid_search'

    def _generate_reasoning(self, query: MemoryQuery, results: List[Dict[str, Any]]) -> str:
        """Generate reasoning for the memory system response."""

        if not results:
            return "No relevant precedents or similar terms found in the document corpus."

        result_types = [r.get('type', 'unknown') for r in results]
        unique_types = list(set(result_types))

        reasoning = f"Found {len(results)} relevant results based on your query. "

        if 'precedent' in unique_types:
            precedent_count = sum(1 for r in results if r.get('type') == 'precedent')
            reasoning += f"Identified {precedent_count} precedents with similar legal concepts. "

        if 'similar_term' in unique_types:
            term_count = sum(1 for r in results if r.get('type') == 'similar_term')
            reasoning += f"Found {term_count} instances of similar terms across documents. "

        if 'clause_evolution' in unique_types:
            reasoning += "Tracked clause evolution over time showing changes in legal positions. "

        reasoning += "Results are ranked by relevance and legal significance."

        return reasoning

    def _generate_suggestions(self, query: MemoryQuery, results: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on the query and results."""

        suggestions = []

        if not results:
            suggestions.append("Try broadening your search terms")
            suggestions.append("Check if documents are properly indexed")
            return suggestions

        if query.query_type == QueryType.PRECEDENT_SEARCH:
            suggestions.append("Review precedent documents for applicable clauses")
            suggestions.append("Consider consulting with legal counsel for interpretation")

        if query.query_type == QueryType.SIMILAR_TERMS:
            suggestions.append("Standardize terminology across similar agreements")
            suggestions.append("Create a legal glossary for consistent usage")

        if query.query_type == QueryType.CONTRADICTION_CHECK:
            suggestions.append("Resolve contradictions before finalizing agreements")
            suggestions.append("Implement document review process to prevent conflicts")

        # Add general suggestions
        suggestions.append("Document findings for future reference")
        suggestions.append("Consider creating templates based on successful precedents")

        return suggestions

    def _calculate_confidence(self, results: List[Dict[str, Any]], query: MemoryQuery) -> float:
        """Calculate confidence score for the memory system response."""

        if not results:
            return 0.0

        # Base confidence on number and quality of results
        result_count_factor = min(len(results) / 10, 1.0)  # Normalize to 0-1

        # Factor in relevance scores if available
        relevance_scores = [r.get('relevance_score', 0.5) for r in results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5

        # Adjust based on query type specificity
        query_specificity = 0.8 if query.filters else 0.6

        # Calculate overall confidence
        confidence = (result_count_factor * 0.4) + (avg_relevance * 0.4) + (query_specificity * 0.2)

        return round(confidence, 3)

    async def _generate_comprehensive_answer(
        self,
        question: str,
        memory_response: MemoryResponse,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive answer to a complex legal question."""

        # Analyze the results to form an answer
        results = memory_response.results

        if not results:
            return {
                'answer': "Based on my analysis of the document corpus, I could not find specific precedents or similar terms related to your question.",
                'confidence': 0.1,
                'supporting_evidence': [],
                'recommendations': [
                    "Consider expanding the search criteria",
                    "Review document indexing and processing",
                    "Consult with legal experts for manual review"
                ]
            }

        # Construct answer based on result types
        answer_parts = []
        supporting_evidence = []

        precedents = [r for r in results if r.get('type') == 'precedent']
        if precedents:
            answer_parts.append(f"Yes, I found {len(precedents)} similar precedents in your document corpus.")
            for precedent in precedents[:3]:  # Top 3
                supporting_evidence.append({
                    'type': 'precedent',
                    'document': precedent.get('document_id'),
                    'relevance': precedent.get('relevance_score', 0),
                    'text': precedent.get('clause_text', '')
                })

        similar_terms = [r for r in results if r.get('type') == 'similar_term']
        if similar_terms:
            answer_parts.append(f"Found {len(similar_terms)} instances of similar terms.")

        # Generate recommendations
        recommendations = memory_response.suggestions

        # Combine answer parts
        full_answer = " ".join(answer_parts) if answer_parts else "No specific matches found."

        return {
            'answer': full_answer,
            'confidence': memory_response.confidence_score,
            'supporting_evidence': supporting_evidence,
            'recommendations': recommendations,
            'reasoning': memory_response.reasoning,
            'processing_time_ms': memory_response.processing_time_ms
        }

    def _initialize_clause_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for recognizing different types of clauses."""

        return {
            'termination': [
                'terminate', 'termination', 'end', 'expire', 'dissolution'
            ],
            'payment': [
                'payment', 'pay', 'compensation', 'salary', 'fee', 'remuneration'
            ],
            'confidentiality': [
                'confidential', 'proprietary', 'trade secret', 'non-disclosure'
            ],
            'liability': [
                'liability', 'liable', 'responsible', 'damages', 'indemnify'
            ],
            'intellectual_property': [
                'intellectual property', 'patent', 'copyright', 'trademark'
            ]
        }

    def _initialize_legal_concepts(self) -> Dict[str, List[str]]:
        """Initialize legal concepts and their associated keywords."""

        return {
            'employment': [
                'employee', 'employer', 'employment', 'job', 'position', 'work'
            ],
            'contract': [
                'contract', 'agreement', 'covenant', 'terms', 'conditions'
            ],
            'payment': [
                'payment', 'salary', 'compensation', 'fee', 'remuneration'
            ],
            'termination': [
                'termination', 'terminate', 'end', 'expire', 'breach'
            ],
            'liability': [
                'liability', 'damages', 'indemnification', 'responsible'
            ],
            'confidentiality': [
                'confidential', 'proprietary', 'trade secret', 'disclosure'
            ]
        }
