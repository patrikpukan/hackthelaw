"""
Multi-Document Legal Analysis System

This module provides sophisticated cross-document analysis capabilities for legal documents,
including contradiction detection, temporal tracking, and semantic clustering.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ContradictionSeverity(Enum):
    """Severity levels for detected contradictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DocumentRelationship:
    """Represents a relationship between two documents."""
    doc1_id: str
    doc2_id: str
    relationship_type: str
    confidence_score: float
    description: str
    detected_at: datetime


@dataclass
class Contradiction:
    """Represents a detected contradiction between documents."""
    id: str
    doc1_id: str
    doc2_id: str
    clause1_text: str
    clause2_text: str
    contradiction_type: str
    severity: ContradictionSeverity
    confidence_score: float
    description: str
    detected_at: datetime


@dataclass
class ClauseEvolution:
    """Tracks how a clause evolves across document versions."""
    clause_type: str
    documents: List[Dict[str, Any]]
    evolution_timeline: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]


class MultiDocumentAnalyzer:
    """
    Advanced multi-document analysis system for legal documents.
    
    Provides capabilities for:
    - Cross-document contradiction detection
    - Temporal tracking of legal positions
    - Semantic clustering of related content
    - Document relationship mapping
    - Clause evolution analysis
    """
    
    def __init__(self):
        self.contradiction_rules = self._initialize_contradiction_rules()
        self.semantic_similarity_threshold = 0.7
        self.temporal_analysis_enabled = True
        
    async def analyze_document_corpus(
        self, 
        documents: List[Dict[str, Any]],
        analysis_options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis across a corpus of legal documents.
        
        Args:
            documents: List of document dictionaries with metadata and content
            analysis_options: Optional configuration for specific analysis types
            
        Returns:
            Comprehensive analysis results including contradictions, relationships, etc.
        """
        
        logger.info(f"Starting multi-document analysis for {len(documents)} documents")
        
        options = analysis_options or {
            'detect_contradictions': True,
            'analyze_relationships': True,
            'track_clause_evolution': True,
            'semantic_clustering': True
        }
        
        results = {
            'analysis_summary': {
                'total_documents': len(documents),
                'analysis_timestamp': datetime.now().isoformat(),
                'options_used': options
            }
        }
        
        # 1. Document relationship analysis
        if options.get('analyze_relationships', True):
            logger.info("Analyzing document relationships...")
            relationships = await self._analyze_document_relationships(documents)
            results['document_relationships'] = relationships
        
        # 2. Contradiction detection
        if options.get('detect_contradictions', True):
            logger.info("Detecting contradictions...")
            contradictions = await self._detect_contradictions(documents)
            results['contradictions'] = contradictions
        
        # 3. Clause evolution tracking
        if options.get('track_clause_evolution', True):
            logger.info("Tracking clause evolution...")
            evolution_analysis = await self._track_clause_evolution(documents)
            results['clause_evolution'] = evolution_analysis
        
        # 4. Semantic clustering
        if options.get('semantic_clustering', True):
            logger.info("Performing semantic clustering...")
            clusters = await self._perform_semantic_clustering(documents)
            results['semantic_clusters'] = clusters
        
        # 5. Generate insights and recommendations
        insights = self._generate_analysis_insights(results)
        results['insights'] = insights
        
        logger.info("Multi-document analysis completed")
        return results
    
    async def _analyze_document_relationships(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[DocumentRelationship]:
        """Analyze relationships between documents."""
        
        relationships = []
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                relationship = await self._detect_document_relationship(doc1, doc2)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    async def _detect_document_relationship(
        self, 
        doc1: Dict[str, Any], 
        doc2: Dict[str, Any]
    ) -> Optional[DocumentRelationship]:
        """Detect relationship between two specific documents."""
        
        # Extract key features for comparison
        doc1_features = self._extract_document_features(doc1)
        doc2_features = self._extract_document_features(doc2)
        
        # Calculate similarity scores
        content_similarity = self._calculate_content_similarity(doc1_features, doc2_features)
        structural_similarity = self._calculate_structural_similarity(doc1_features, doc2_features)
        temporal_relationship = self._analyze_temporal_relationship(doc1, doc2)
        
        # Determine relationship type and confidence
        relationship_type, confidence = self._classify_relationship(
            content_similarity, structural_similarity, temporal_relationship
        )
        
        if confidence > 0.3:  # Threshold for meaningful relationships
            return DocumentRelationship(
                doc1_id=doc1['id'],
                doc2_id=doc2['id'],
                relationship_type=relationship_type,
                confidence_score=confidence,
                description=f"{relationship_type} relationship with {confidence:.2f} confidence",
                detected_at=datetime.now()
            )
        
        return None
    
    async def _detect_contradictions(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Contradiction]:
        """Detect contradictions between documents."""
        
        contradictions = []
        
        # Extract clauses from all documents
        all_clauses = []
        for doc in documents:
            doc_clauses = self._extract_clauses_for_comparison(doc)
            all_clauses.extend(doc_clauses)
        
        # Compare clauses pairwise for contradictions
        for i, clause1 in enumerate(all_clauses):
            for clause2 in all_clauses[i+1:]:
                if clause1['document_id'] != clause2['document_id']:  # Different documents
                    contradiction = await self._check_clause_contradiction(clause1, clause2)
                    if contradiction:
                        contradictions.append(contradiction)
        
        return contradictions
    
    async def _track_clause_evolution(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, ClauseEvolution]:
        """Track how clauses evolve across documents and time."""
        
        # Group documents by type and sort by date
        document_groups = self._group_documents_by_type(documents)
        
        evolution_tracking = {}
        
        for doc_type, doc_group in document_groups.items():
            # Sort by date
            sorted_docs = sorted(doc_group, key=lambda x: x.get('date', datetime.min))
            
            # Track clause evolution within this document type
            clause_evolution = await self._analyze_clause_evolution_in_group(sorted_docs)
            evolution_tracking[doc_type] = clause_evolution
        
        return evolution_tracking
    
    async def _perform_semantic_clustering(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform semantic clustering of document content."""
        
        # Extract semantic features from documents
        document_vectors = []
        for doc in documents:
            vector = await self._extract_semantic_vector(doc)
            document_vectors.append({
                'document_id': doc['id'],
                'vector': vector,
                'metadata': doc.get('metadata', {})
            })
        
        # Perform clustering
        clusters = self._cluster_documents(document_vectors)
        
        # Analyze clusters for insights
        cluster_analysis = self._analyze_clusters(clusters, documents)
        
        return {
            'clusters': clusters,
            'cluster_analysis': cluster_analysis,
            'total_clusters': len(clusters)
        }
    
    def _initialize_contradiction_rules(self) -> Dict[str, Any]:
        """Initialize rules for detecting contradictions."""
        
        return {
            'payment_terms': {
                'keywords': ['payment', 'pay', 'compensation', 'salary', 'fee'],
                'contradiction_patterns': [
                    'different_amounts',
                    'different_schedules',
                    'conflicting_conditions'
                ]
            },
            'termination_clauses': {
                'keywords': ['termination', 'terminate', 'end', 'expire'],
                'contradiction_patterns': [
                    'different_notice_periods',
                    'conflicting_conditions',
                    'different_severance'
                ]
            },
            'confidentiality': {
                'keywords': ['confidential', 'proprietary', 'trade secret'],
                'contradiction_patterns': [
                    'different_scope',
                    'conflicting_duration',
                    'different_exceptions'
                ]
            }
        }

    def _extract_document_features(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from a document for comparison."""

        content = document.get('content', '')
        metadata = document.get('metadata', {})

        # Extract structural features
        word_count = len(content.split())
        section_count = content.lower().count('section')
        clause_indicators = ['shall', 'must', 'will', 'agrees', 'covenant']
        clause_density = sum(1 for indicator in clause_indicators if indicator in content.lower())

        # Extract semantic features
        legal_terms = self._extract_legal_terms(content)
        key_entities = self._extract_key_entities_simple(content)

        return {
            'word_count': word_count,
            'section_count': section_count,
            'clause_density': clause_density,
            'legal_terms': legal_terms,
            'key_entities': key_entities,
            'document_type': metadata.get('type', 'unknown'),
            'date': metadata.get('date'),
            'parties': metadata.get('parties', [])
        }

    def _calculate_content_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate content similarity between two documents."""

        # Compare legal terms overlap
        terms1 = set(features1.get('legal_terms', []))
        terms2 = set(features2.get('legal_terms', []))

        if terms1 and terms2:
            term_overlap = len(terms1.intersection(terms2)) / len(terms1.union(terms2))
        else:
            term_overlap = 0.0

        # Compare entities overlap
        entities1 = set(features1.get('key_entities', {}).get('companies', []))
        entities2 = set(features2.get('key_entities', {}).get('companies', []))

        if entities1 and entities2:
            entity_overlap = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
        else:
            entity_overlap = 0.0

        # Weighted combination
        content_similarity = (term_overlap * 0.7) + (entity_overlap * 0.3)
        return content_similarity

    def _calculate_structural_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity between documents."""

        # Compare document types
        type_match = 1.0 if features1.get('document_type') == features2.get('document_type') else 0.0

        # Compare structural metrics
        word_ratio = min(features1.get('word_count', 0), features2.get('word_count', 0)) / \
                    max(features1.get('word_count', 1), features2.get('word_count', 1))

        section_ratio = min(features1.get('section_count', 0), features2.get('section_count', 0)) / \
                       max(features1.get('section_count', 1), features2.get('section_count', 1))

        clause_ratio = min(features1.get('clause_density', 0), features2.get('clause_density', 0)) / \
                      max(features1.get('clause_density', 1), features2.get('clause_density', 1))

        # Weighted combination
        structural_similarity = (type_match * 0.5) + (word_ratio * 0.2) + \
                               (section_ratio * 0.15) + (clause_ratio * 0.15)

        return structural_similarity

    def _analyze_temporal_relationship(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal relationship between documents."""

        date1 = doc1.get('metadata', {}).get('date')
        date2 = doc2.get('metadata', {}).get('date')

        if not date1 or not date2:
            return {'has_temporal_data': False}

        # Convert to datetime if strings
        if isinstance(date1, str):
            try:
                date1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            except:
                return {'has_temporal_data': False}

        if isinstance(date2, str):
            try:
                date2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            except:
                return {'has_temporal_data': False}

        time_diff = abs((date2 - date1).days)

        # Classify temporal relationship
        if time_diff == 0:
            relationship = 'same_date'
        elif time_diff <= 30:
            relationship = 'close_temporal'
        elif time_diff <= 365:
            relationship = 'same_year'
        else:
            relationship = 'distant_temporal'

        return {
            'has_temporal_data': True,
            'time_difference_days': time_diff,
            'temporal_relationship': relationship,
            'chronological_order': 'doc1_first' if date1 < date2 else 'doc2_first'
        }

    def _classify_relationship(
        self,
        content_sim: float,
        structural_sim: float,
        temporal_rel: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Classify the relationship type and confidence between documents."""

        # Calculate overall similarity
        overall_similarity = (content_sim * 0.6) + (structural_sim * 0.4)

        # Adjust based on temporal relationship
        temporal_boost = 0.0
        if temporal_rel.get('has_temporal_data'):
            if temporal_rel.get('temporal_relationship') == 'close_temporal':
                temporal_boost = 0.1
            elif temporal_rel.get('temporal_relationship') == 'same_year':
                temporal_boost = 0.05

        final_confidence = min(overall_similarity + temporal_boost, 1.0)

        # Classify relationship type
        if final_confidence > 0.8:
            relationship_type = 'amendment_or_version'
        elif final_confidence > 0.6:
            relationship_type = 'related_agreement'
        elif final_confidence > 0.4:
            relationship_type = 'similar_type'
        else:
            relationship_type = 'weak_relation'

        return relationship_type, final_confidence

    def _extract_legal_terms(self, content: str) -> List[str]:
        """Extract legal terms from document content."""

        legal_terms = [
            'agreement', 'contract', 'party', 'parties', 'whereas', 'therefore',
            'covenant', 'warrant', 'represent', 'indemnify', 'liability',
            'termination', 'breach', 'default', 'confidential', 'proprietary',
            'governing law', 'jurisdiction', 'arbitration', 'mediation',
            'force majeure', 'assignment', 'amendment', 'waiver'
        ]

        content_lower = content.lower()
        found_terms = [term for term in legal_terms if term in content_lower]

        return found_terms

    def _extract_key_entities_simple(self, content: str) -> Dict[str, List[str]]:
        """Simple extraction of key entities from content."""
        import re

        # Extract company names
        company_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Inc\.|LLC|Corp\.|Corporation|Company)\b'
        companies = re.findall(company_pattern, content)

        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts = re.findall(money_pattern, content)

        # Extract dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, content, re.IGNORECASE)

        return {
            'companies': list(set(companies)),
            'monetary_amounts': list(set(amounts)),
            'dates': list(set(dates))
        }

    def _extract_clauses_for_comparison(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract clauses from a document for contradiction analysis."""

        content = document.get('content', '')
        doc_id = document.get('id', '')

        # Define clause patterns
        clause_patterns = {
            'payment': r'(?:payment|pay|compensation|salary|fee).*?(?:\.|;|\n)',
            'termination': r'(?:termination|terminate|end|expire).*?(?:\.|;|\n)',
            'confidentiality': r'(?:confidential|proprietary|trade secret).*?(?:\.|;|\n)',
            'liability': r'(?:liability|liable|responsible|damages).*?(?:\.|;|\n)',
            'governing_law': r'(?:governing law|jurisdiction|applicable law).*?(?:\.|;|\n)'
        }

        clauses = []
        import re

        for clause_type, pattern in clause_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clauses.append({
                    'document_id': doc_id,
                    'clause_type': clause_type,
                    'text': match.group(0).strip(),
                    'start_position': match.start(),
                    'end_position': match.end()
                })

        return clauses

    async def _check_clause_contradiction(
        self,
        clause1: Dict[str, Any],
        clause2: Dict[str, Any]
    ) -> Optional[Contradiction]:
        """Check if two clauses contradict each other."""

        # Only compare clauses of the same type
        if clause1['clause_type'] != clause2['clause_type']:
            return None

        clause_type = clause1['clause_type']
        text1 = clause1['text'].lower()
        text2 = clause2['text'].lower()

        # Apply type-specific contradiction detection
        contradiction_detected = False
        contradiction_description = ""
        severity = ContradictionSeverity.LOW

        if clause_type == 'payment':
            contradiction_detected, contradiction_description, severity = \
                self._detect_payment_contradiction(text1, text2)
        elif clause_type == 'termination':
            contradiction_detected, contradiction_description, severity = \
                self._detect_termination_contradiction(text1, text2)
        elif clause_type == 'confidentiality':
            contradiction_detected, contradiction_description, severity = \
                self._detect_confidentiality_contradiction(text1, text2)

        if contradiction_detected:
            return Contradiction(
                id=f"contradiction_{clause1['document_id']}_{clause2['document_id']}_{clause_type}",
                doc1_id=clause1['document_id'],
                doc2_id=clause2['document_id'],
                clause1_text=clause1['text'],
                clause2_text=clause2['text'],
                contradiction_type=clause_type,
                severity=severity,
                confidence_score=0.8,  # Default confidence
                description=contradiction_description,
                detected_at=datetime.now()
            )

        return None

    def _detect_payment_contradiction(self, text1: str, text2: str) -> Tuple[bool, str, ContradictionSeverity]:
        """Detect contradictions in payment clauses."""
        import re

        # Extract monetary amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?'
        amounts1 = re.findall(amount_pattern, text1)
        amounts2 = re.findall(amount_pattern, text2)

        if amounts1 and amounts2:
            # Check for different amounts
            if set(amounts1) != set(amounts2):
                return True, f"Different payment amounts: {amounts1} vs {amounts2}", ContradictionSeverity.HIGH

        # Check for different payment schedules
        schedule_keywords = ['monthly', 'weekly', 'annually', 'quarterly', 'biweekly']
        schedule1 = [kw for kw in schedule_keywords if kw in text1]
        schedule2 = [kw for kw in schedule_keywords if kw in text2]

        if schedule1 and schedule2 and set(schedule1) != set(schedule2):
            return True, f"Different payment schedules: {schedule1} vs {schedule2}", ContradictionSeverity.MEDIUM

        return False, "", ContradictionSeverity.LOW
