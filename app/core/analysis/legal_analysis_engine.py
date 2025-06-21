"""
Legal Analysis Engine for Multi-Stage Legal Document Analysis

This module provides comprehensive legal analysis capabilities including contradiction detection,
temporal tracking of legal positions, semantic clustering, and cross-document relationship identification.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import re

from app.core.analysis.document_context_processor import ProcessedContext, DocumentSection
from app.core.analysis.intelligent_compressor import CompressionResult
from app.core.chat.llm_client import LLMClient, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    """Represents a detected contradiction between documents or sections."""
    id: str
    type: str  # 'direct', 'implicit', 'temporal'
    severity: str  # 'critical', 'major', 'minor'
    description: str
    source1: Dict[str, str]  # document_id, section, content
    source2: Dict[str, str]  # document_id, section, content
    confidence: float
    legal_implications: List[str]


@dataclass
class TemporalPosition:
    """Represents a legal position at a specific point in time."""
    position_id: str
    topic: str
    position_text: str
    document_id: str
    section_id: str
    timestamp: Optional[datetime]
    version_info: Optional[str]
    confidence: float


@dataclass
class SemanticCluster:
    """Represents a cluster of semantically related content."""
    cluster_id: str
    topic: str
    sections: List[str]  # section identifiers
    key_concepts: List[str]
    cluster_summary: str
    coherence_score: float


@dataclass
class LegalAnalysisResult:
    """Complete result of legal analysis."""
    contradictions: List[Contradiction]
    temporal_analysis: Dict[str, List[TemporalPosition]]
    semantic_clusters: List[SemanticCluster]
    cross_document_insights: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    analysis_metadata: Dict[str, Any]


class LegalAnalysisEngine:
    """
    Comprehensive legal analysis engine for complex document analysis.
    
    The engine provides:
    1. Contradiction detection across documents
    2. Temporal tracking of legal positions
    3. Semantic clustering of related content
    4. Cross-document relationship analysis
    5. Risk assessment and recommendations
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
        # Legal concept patterns for analysis
        self.legal_patterns = {
            'obligations': [
                r'\b(?:shall|must|required to|obligated to|duty to)\b',
                r'\b(?:covenant|undertake|agree to)\b'
            ],
            'rights': [
                r'\b(?:right to|entitled to|may|permitted to)\b',
                r'\b(?:privilege|authority|power to)\b'
            ],
            'prohibitions': [
                r'\b(?:shall not|must not|prohibited|forbidden)\b',
                r'\b(?:may not|cannot|restricted from)\b'
            ],
            'conditions': [
                r'\b(?:if|unless|provided that|subject to)\b',
                r'\b(?:conditional upon|contingent on)\b'
            ],
            'temporal': [
                r'\b(?:within|by|before|after|during)\b.*?(?:days?|weeks?|months?|years?)',
                r'\b(?:deadline|due date|expiry|expiration)\b'
            ]
        }
    
    async def analyze_legal_content(
        self, 
        processed_context: ProcessedContext,
        compression_result: CompressionResult,
        query: str,
        analysis_options: Dict[str, bool]
    ) -> LegalAnalysisResult:
        """
        Perform comprehensive legal analysis on processed content.
        
        Args:
            processed_context: Processed document context
            compression_result: Compressed content result
            query: Original user query
            analysis_options: Options for different analysis types
            
        Returns:
            LegalAnalysisResult with comprehensive analysis
        """
        logger.info("Starting comprehensive legal analysis")
        
        contradictions = []
        temporal_analysis = {}
        semantic_clusters = []
        cross_document_insights = []
        
        # Step 1: Contradiction Detection
        if analysis_options.get('enable_contradiction_detection', True):
            contradictions = await self._detect_contradictions(processed_context)
        
        # Step 2: Temporal Analysis
        if analysis_options.get('enable_temporal_tracking', True):
            temporal_analysis = await self._analyze_temporal_positions(processed_context)
        
        # Step 3: Semantic Clustering
        if analysis_options.get('enable_semantic_clustering', True):
            semantic_clusters = await self._perform_semantic_clustering(processed_context)
        
        # Step 4: Cross-Document Analysis
        if analysis_options.get('enable_cross_document_reasoning', True):
            cross_document_insights = await self._analyze_cross_document_relationships(
                processed_context, contradictions
            )
        
        # Step 5: Risk Assessment
        risk_assessment = await self._assess_legal_risks(
            contradictions, temporal_analysis, cross_document_insights
        )
        
        # Step 6: Generate Recommendations
        recommendations = await self._generate_recommendations(
            contradictions, temporal_analysis, risk_assessment, query
        )
        
        analysis_metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sections_analyzed': len(processed_context.sections),
            'contradictions_found': len(contradictions),
            'temporal_positions': sum(len(positions) for positions in temporal_analysis.values()),
            'semantic_clusters': len(semantic_clusters),
            'cross_document_insights': len(cross_document_insights)
        }
        
        return LegalAnalysisResult(
            contradictions=contradictions,
            temporal_analysis=temporal_analysis,
            semantic_clusters=semantic_clusters,
            cross_document_insights=cross_document_insights,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            analysis_metadata=analysis_metadata
        )
    
    async def _detect_contradictions(
        self, 
        processed_context: ProcessedContext
    ) -> List[Contradiction]:
        """Detect contradictions between documents and sections."""
        
        contradictions = []
        sections = processed_context.sections
        
        # Compare each pair of sections for contradictions
        for i, section1 in enumerate(sections):
            for j, section2 in enumerate(sections[i+1:], i+1):
                contradiction = await self._analyze_section_contradiction(section1, section2)
                if contradiction:
                    contradictions.append(contradiction)
        
        # Sort by severity and confidence
        contradictions.sort(key=lambda x: (
            {'critical': 3, 'major': 2, 'minor': 1}[x.severity],
            x.confidence
        ), reverse=True)
        
        logger.info(f"Detected {len(contradictions)} potential contradictions")
        return contradictions[:20]  # Limit to top 20
    
    async def _analyze_section_contradiction(
        self, 
        section1: DocumentSection, 
        section2: DocumentSection
    ) -> Optional[Contradiction]:
        """Analyze two sections for potential contradictions."""
        
        # Skip if same document and adjacent sections (likely related)
        if (section1.document_id == section2.document_id and 
            abs(section1.start_position - section2.start_position) < 1000):
            return None
        
        # Use LLM to detect contradictions
        prompt = f"""Analyze these two legal document sections for contradictions:

SECTION 1:
Document: {section1.document_title}
Section: {section1.section_title}
Content: {section1.content[:1000]}

SECTION 2:
Document: {section2.document_title}
Section: {section2.section_title}
Content: {section2.content[:1000]}

Identify any contradictions, conflicts, or inconsistencies between these sections.
Consider:
1. Direct contradictions (opposite statements)
2. Implicit conflicts (incompatible requirements)
3. Temporal conflicts (timeline inconsistencies)

Respond in JSON format:
{{
    "has_contradiction": true/false,
    "type": "direct|implicit|temporal",
    "severity": "critical|major|minor",
    "description": "detailed description",
    "confidence": 0.0-1.0,
    "legal_implications": ["implication1", "implication2"]
}}"""

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            analysis = json.loads(response.content)
            
            if analysis.get('has_contradiction', False):
                return Contradiction(
                    id=f"contradiction_{section1.document_id}_{section2.document_id}_{hash(section1.section_title + section2.section_title)}",
                    type=analysis.get('type', 'unknown'),
                    severity=analysis.get('severity', 'minor'),
                    description=analysis.get('description', ''),
                    source1={
                        'document_id': section1.document_id,
                        'document_title': section1.document_title,
                        'section': section1.section_title,
                        'content': section1.content[:500]
                    },
                    source2={
                        'document_id': section2.document_id,
                        'document_title': section2.document_title,
                        'section': section2.section_title,
                        'content': section2.content[:500]
                    },
                    confidence=analysis.get('confidence', 0.5),
                    legal_implications=analysis.get('legal_implications', [])
                )
        
        except Exception as e:
            logger.error(f"Error analyzing contradiction: {e}")
        
        return None
    
    async def _analyze_temporal_positions(
        self, 
        processed_context: ProcessedContext
    ) -> Dict[str, List[TemporalPosition]]:
        """Analyze temporal evolution of legal positions."""
        
        temporal_positions = defaultdict(list)
        
        # Extract temporal markers and associated positions
        for marker in processed_context.temporal_markers:
            # Find sections related to this temporal marker
            related_sections = await self._find_sections_for_temporal_marker(
                marker, processed_context.sections
            )
            
            for section in related_sections:
                position = await self._extract_temporal_position(section, marker)
                if position:
                    topic = position.topic
                    temporal_positions[topic].append(position)
        
        # Sort positions by timestamp for each topic
        for topic in temporal_positions:
            temporal_positions[topic].sort(
                key=lambda x: x.timestamp or datetime.min
            )
        
        logger.info(f"Analyzed temporal positions for {len(temporal_positions)} topics")
        return dict(temporal_positions)
    
    async def _find_sections_for_temporal_marker(
        self, 
        marker: Dict[str, Any], 
        sections: List[DocumentSection]
    ) -> List[DocumentSection]:
        """Find sections related to a temporal marker."""
        
        related_sections = []
        marker_source = marker.get('source_section', '')
        
        for section in sections:
            # Direct match by source section
            if section.section_title == marker_source:
                related_sections.append(section)
            # Content-based matching
            elif marker.get('description', '').lower() in section.content.lower():
                related_sections.append(section)
        
        return related_sections
    
    async def _extract_temporal_position(
        self, 
        section: DocumentSection, 
        marker: Dict[str, Any]
    ) -> Optional[TemporalPosition]:
        """Extract a temporal position from a section and marker."""
        
        # Simple implementation - could be enhanced with more sophisticated extraction
        try:
            timestamp = None
            if marker.get('date'):
                # Try to parse the date
                date_str = marker['date']
                # This is a simplified parser - would need more robust date parsing
                timestamp = datetime.now()  # Placeholder
            
            return TemporalPosition(
                position_id=f"pos_{section.document_id}_{hash(section.section_title)}",
                topic=marker.get('description', 'Unknown Topic'),
                position_text=section.content[:500],
                document_id=section.document_id,
                section_id=section.section_title,
                timestamp=timestamp,
                version_info=None,  # Could be enhanced with version tracking
                confidence=0.7  # Default confidence
            )
        
        except Exception as e:
            logger.error(f"Error extracting temporal position: {e}")
            return None

    async def _perform_semantic_clustering(
        self,
        processed_context: ProcessedContext
    ) -> List[SemanticCluster]:
        """Perform semantic clustering of related content."""

        clusters = []
        sections = processed_context.sections

        # Group sections by key concepts
        concept_groups = defaultdict(list)

        for section in sections:
            analysis = section.metadata.get('light_model_analysis', {})
            key_concepts = analysis.get('key_concepts', [])

            for concept in key_concepts:
                concept_groups[concept].append(section)

        # Create clusters for concepts with multiple sections
        cluster_id = 0
        for concept, related_sections in concept_groups.items():
            if len(related_sections) >= 2:  # At least 2 sections for a cluster
                cluster_summary = await self._generate_cluster_summary(
                    concept, related_sections
                )

                clusters.append(SemanticCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    topic=concept,
                    sections=[f"{s.document_id}:{s.section_title}" for s in related_sections],
                    key_concepts=[concept],
                    cluster_summary=cluster_summary,
                    coherence_score=len(related_sections) / len(sections)  # Simple coherence metric
                ))
                cluster_id += 1

        # Sort by coherence score
        clusters.sort(key=lambda x: x.coherence_score, reverse=True)

        logger.info(f"Created {len(clusters)} semantic clusters")
        return clusters[:15]  # Limit to top 15 clusters

    async def _generate_cluster_summary(
        self,
        concept: str,
        sections: List[DocumentSection]
    ) -> str:
        """Generate a summary for a semantic cluster."""

        # Simple summary generation
        section_titles = [s.section_title for s in sections]
        document_titles = list(set([s.document_title for s in sections]))

        summary = f"Concept '{concept}' appears across {len(sections)} sections "
        summary += f"in {len(document_titles)} document(s): {', '.join(document_titles[:3])}"

        if len(document_titles) > 3:
            summary += f" and {len(document_titles) - 3} others"

        return summary

    async def _analyze_cross_document_relationships(
        self,
        processed_context: ProcessedContext,
        contradictions: List[Contradiction]
    ) -> List[Dict[str, Any]]:
        """Analyze relationships across documents."""

        insights = []

        # Group sections by document
        doc_sections = defaultdict(list)
        for section in processed_context.sections:
            doc_sections[section.document_id].append(section)

        # Analyze relationships between document pairs
        doc_ids = list(doc_sections.keys())
        for i, doc1_id in enumerate(doc_ids):
            for doc2_id in doc_ids[i+1:]:
                relationship = await self._analyze_document_pair_relationship(
                    doc1_id, doc_sections[doc1_id],
                    doc2_id, doc_sections[doc2_id],
                    contradictions
                )
                if relationship:
                    insights.append(relationship)

        logger.info(f"Generated {len(insights)} cross-document insights")
        return insights

    async def _analyze_document_pair_relationship(
        self,
        doc1_id: str,
        doc1_sections: List[DocumentSection],
        doc2_id: str,
        doc2_sections: List[DocumentSection],
        contradictions: List[Contradiction]
    ) -> Optional[Dict[str, Any]]:
        """Analyze relationship between a pair of documents."""

        # Count contradictions between these documents
        doc_contradictions = [
            c for c in contradictions
            if ((c.source1['document_id'] == doc1_id and c.source2['document_id'] == doc2_id) or
                (c.source1['document_id'] == doc2_id and c.source2['document_id'] == doc1_id))
        ]

        # Find common concepts
        doc1_concepts = set()
        doc2_concepts = set()

        for section in doc1_sections:
            analysis = section.metadata.get('light_model_analysis', {})
            doc1_concepts.update(analysis.get('key_concepts', []))

        for section in doc2_sections:
            analysis = section.metadata.get('light_model_analysis', {})
            doc2_concepts.update(analysis.get('key_concepts', []))

        common_concepts = doc1_concepts.intersection(doc2_concepts)

        if len(common_concepts) > 0 or len(doc_contradictions) > 0:
            doc1_title = doc1_sections[0].document_title if doc1_sections else "Unknown"
            doc2_title = doc2_sections[0].document_title if doc2_sections else "Unknown"

            return {
                'document1': {'id': doc1_id, 'title': doc1_title},
                'document2': {'id': doc2_id, 'title': doc2_title},
                'common_concepts': list(common_concepts),
                'contradictions': len(doc_contradictions),
                'relationship_strength': len(common_concepts) / max(len(doc1_concepts), len(doc2_concepts), 1),
                'relationship_type': 'conflicting' if doc_contradictions else 'complementary'
            }

        return None

    async def _assess_legal_risks(
        self,
        contradictions: List[Contradiction],
        temporal_analysis: Dict[str, List[TemporalPosition]],
        cross_document_insights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess legal risks based on analysis results."""

        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'critical_issues': [],
            'risk_score': 0.0
        }

        risk_score = 0.0

        # Risk from contradictions
        critical_contradictions = [c for c in contradictions if c.severity == 'critical']
        major_contradictions = [c for c in contradictions if c.severity == 'major']

        if critical_contradictions:
            risk_score += len(critical_contradictions) * 0.3
            risk_assessment['risk_factors'].append(f"{len(critical_contradictions)} critical contradictions")
            risk_assessment['critical_issues'].extend([c.description for c in critical_contradictions])

        if major_contradictions:
            risk_score += len(major_contradictions) * 0.2
            risk_assessment['risk_factors'].append(f"{len(major_contradictions)} major contradictions")

        # Risk from temporal issues
        temporal_conflicts = 0
        for topic, positions in temporal_analysis.items():
            if len(positions) > 1:
                # Check for conflicting temporal positions
                temporal_conflicts += 1

        if temporal_conflicts > 0:
            risk_score += temporal_conflicts * 0.1
            risk_assessment['risk_factors'].append(f"{temporal_conflicts} temporal conflicts")

        # Risk from cross-document conflicts
        conflicting_relationships = [
            insight for insight in cross_document_insights
            if insight.get('relationship_type') == 'conflicting'
        ]

        if conflicting_relationships:
            risk_score += len(conflicting_relationships) * 0.15
            risk_assessment['risk_factors'].append(f"{len(conflicting_relationships)} conflicting document relationships")

        # Determine overall risk level
        risk_assessment['risk_score'] = min(risk_score, 1.0)

        if risk_score >= 0.7:
            risk_assessment['overall_risk_level'] = 'critical'
        elif risk_score >= 0.4:
            risk_assessment['overall_risk_level'] = 'high'
        elif risk_score >= 0.2:
            risk_assessment['overall_risk_level'] = 'medium'
        else:
            risk_assessment['overall_risk_level'] = 'low'

        return risk_assessment

    async def _generate_recommendations(
        self,
        contradictions: List[Contradiction],
        temporal_analysis: Dict[str, List[TemporalPosition]],
        risk_assessment: Dict[str, Any],
        query: str
    ) -> List[str]:
        """Generate recommendations based on analysis results."""

        recommendations = []

        # Recommendations for contradictions
        if contradictions:
            critical_contradictions = [c for c in contradictions if c.severity == 'critical']
            if critical_contradictions:
                recommendations.append(
                    f"URGENT: Resolve {len(critical_contradictions)} critical contradictions "
                    "that could lead to legal disputes or compliance issues."
                )

            recommendations.append(
                "Review and reconcile contradictory provisions across documents "
                "to ensure consistency and enforceability."
            )

        # Recommendations for temporal issues
        if temporal_analysis:
            recommendations.append(
                "Establish clear timelines and ensure all temporal requirements "
                "are consistent across related documents."
            )

        # Risk-based recommendations
        if risk_assessment['overall_risk_level'] in ['critical', 'high']:
            recommendations.append(
                "Immediate legal review recommended due to high risk factors identified."
            )

        # Query-specific recommendations
        if 'liability' in query.lower():
            recommendations.append(
                "Pay special attention to liability allocation and limitation clauses "
                "across all related documents."
            )

        if 'compliance' in query.lower():
            recommendations.append(
                "Ensure all compliance requirements are clearly defined and "
                "consistently applied across the document set."
            )

        # General recommendations
        recommendations.append(
            "Consider creating a master document or cross-reference matrix "
            "to track relationships between related provisions."
        )

        return recommendations[:10]  # Limit to top 10 recommendations
