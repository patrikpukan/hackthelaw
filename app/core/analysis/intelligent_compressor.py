"""
Intelligent Compression System for Multi-Stage Legal Analysis

This module compresses processed document context while preserving critical legal details,
relationships, and cross-document connections for the final analysis stage.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
from collections import defaultdict

from app.core.analysis.document_context_processor import ProcessedContext, DocumentSection
from app.core.chat.llm_client import LLMClient, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of intelligent compression process."""
    compressed_content: str
    compression_ratio: float
    preserved_elements: Dict[str, int]
    token_count: int
    critical_relationships: List[Dict[str, Any]]
    key_legal_points: List[str]
    compression_metadata: Dict[str, Any]


@dataclass
class CompressionStrategy:
    """Configuration for compression strategy."""
    target_token_count: int = 250000  # Leave room for final analysis
    preserve_legal_entities: bool = True
    preserve_temporal_markers: bool = True
    preserve_cross_references: bool = True
    preserve_contradictions: bool = True
    min_section_importance: float = 0.3
    max_summary_ratio: float = 0.4  # Max 40% of original content


class IntelligentCompressor:
    """
    Compresses processed document context while preserving critical legal information.
    
    The compressor:
    1. Identifies critical legal elements that must be preserved
    2. Summarizes less critical content
    3. Maintains cross-document relationships
    4. Ensures final output fits within token limits
    """
    
    def __init__(self, light_model_client: LLMClient):
        self.light_model_client = light_model_client
    
    async def compress_context(
        self, 
        processed_context: ProcessedContext,
        strategy: CompressionStrategy,
        query: str
    ) -> CompressionResult:
        """
        Compress processed context while preserving critical legal information.
        
        Args:
            processed_context: The processed document context
            strategy: Compression strategy configuration
            query: Original user query for relevance scoring
            
        Returns:
            CompressionResult with compressed content and metadata
        """
        logger.info(f"Starting intelligent compression of {len(processed_context.sections)} sections")
        
        # Step 1: Analyze and prioritize sections
        prioritized_sections = await self._prioritize_sections(
            processed_context.sections, query, strategy
        )
        
        # Step 2: Extract and preserve critical elements
        critical_elements = await self._extract_critical_elements(
            processed_context, strategy
        )
        
        # Step 3: Compress sections based on priority
        compressed_sections = await self._compress_sections(
            prioritized_sections, critical_elements, strategy
        )
        
        # Step 4: Build final compressed content
        compressed_content = await self._build_compressed_content(
            compressed_sections, critical_elements, processed_context
        )
        
        # Step 5: Calculate metrics and metadata
        original_token_count = self._estimate_token_count(
            "\n\n".join([s.content for s in processed_context.sections])
        )
        compressed_token_count = self._estimate_token_count(compressed_content)
        compression_ratio = compressed_token_count / original_token_count if original_token_count > 0 else 0
        
        preserved_elements = {
            'legal_entities': len(critical_elements.get('legal_entities', [])),
            'temporal_markers': len(critical_elements.get('temporal_markers', [])),
            'cross_references': len(critical_elements.get('cross_references', [])),
            'key_concepts': len(critical_elements.get('key_concepts', []))
        }
        
        compression_metadata = {
            'original_sections': len(processed_context.sections),
            'compressed_sections': len(compressed_sections),
            'original_token_count': original_token_count,
            'compression_strategy': strategy.__dict__,
            'critical_elements_preserved': sum(preserved_elements.values())
        }
        
        return CompressionResult(
            compressed_content=compressed_content,
            compression_ratio=compression_ratio,
            preserved_elements=preserved_elements,
            token_count=compressed_token_count,
            critical_relationships=processed_context.relationships,
            key_legal_points=critical_elements.get('key_legal_points', []),
            compression_metadata=compression_metadata
        )
    
    async def _prioritize_sections(
        self, 
        sections: List[DocumentSection], 
        query: str,
        strategy: CompressionStrategy
    ) -> List[Tuple[DocumentSection, float]]:
        """Prioritize sections based on importance and relevance."""
        
        prioritized = []
        
        for section in sections:
            # Calculate importance score
            importance_score = await self._calculate_section_importance(section, query)
            
            # Only include sections above minimum importance
            if importance_score >= strategy.min_section_importance:
                prioritized.append((section, importance_score))
        
        # Sort by importance (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Prioritized {len(prioritized)} sections out of {len(sections)}")
        return prioritized
    
    async def _calculate_section_importance(
        self, 
        section: DocumentSection, 
        query: str
    ) -> float:
        """Calculate importance score for a section."""
        
        score = 0.0
        
        # Base relevance score
        relevance = section.metadata.get('relevance_score', 0.0)
        score += relevance * 0.4
        
        # Light model analysis score
        analysis = section.metadata.get('light_model_analysis', {})
        if analysis.get('relevance_to_query'):
            score += analysis['relevance_to_query'] * 0.3
        
        # Section type importance
        type_weights = {
            'definition': 0.8,
            'obligation': 0.9,
            'rights': 0.9,
            'clause': 0.7,
            'general': 0.5
        }
        score += type_weights.get(section.section_type, 0.5) * 0.2
        
        # Content richness (entities, concepts, etc.)
        if analysis:
            content_richness = (
                len(analysis.get('legal_entities', [])) * 0.1 +
                len(analysis.get('key_concepts', [])) * 0.05 +
                len(analysis.get('temporal_markers', [])) * 0.15 +
                len(analysis.get('cross_references', [])) * 0.1
            )
            score += min(content_richness, 0.1)  # Cap at 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _extract_critical_elements(
        self, 
        processed_context: ProcessedContext,
        strategy: CompressionStrategy
    ) -> Dict[str, Any]:
        """Extract critical elements that must be preserved."""
        
        critical_elements = {}
        
        if strategy.preserve_legal_entities:
            critical_elements['legal_entities'] = processed_context.legal_entities
        
        if strategy.preserve_temporal_markers:
            critical_elements['temporal_markers'] = processed_context.temporal_markers
        
        if strategy.preserve_cross_references:
            critical_elements['cross_references'] = processed_context.cross_references
        
        # Extract key legal points from high-priority sections
        key_legal_points = []
        for section in processed_context.sections:
            analysis = section.metadata.get('light_model_analysis', {})
            obligations = analysis.get('obligations', [])
            rights = analysis.get('rights', [])
            key_legal_points.extend(obligations + rights)
        
        critical_elements['key_legal_points'] = list(set(key_legal_points))
        critical_elements['key_concepts'] = processed_context.key_concepts
        
        return critical_elements
    
    async def _compress_sections(
        self, 
        prioritized_sections: List[Tuple[DocumentSection, float]],
        critical_elements: Dict[str, Any],
        strategy: CompressionStrategy
    ) -> List[Dict[str, Any]]:
        """Compress sections based on priority and strategy."""
        
        compressed_sections = []
        current_token_count = 0
        
        for section, importance_score in prioritized_sections:
            if current_token_count >= strategy.target_token_count:
                break
            
            # Determine compression level based on importance
            if importance_score > 0.8:
                # High importance: minimal compression
                compression_level = "minimal"
                max_tokens = 2000
            elif importance_score > 0.6:
                # Medium importance: moderate compression
                compression_level = "moderate"
                max_tokens = 1000
            else:
                # Lower importance: aggressive compression
                compression_level = "aggressive"
                max_tokens = 500
            
            # Compress the section
            compressed_section = await self._compress_single_section(
                section, compression_level, max_tokens, critical_elements
            )
            
            section_tokens = self._estimate_token_count(compressed_section['content'])
            
            # Check if we can fit this section
            if current_token_count + section_tokens <= strategy.target_token_count:
                compressed_sections.append(compressed_section)
                current_token_count += section_tokens
            else:
                # Try with more aggressive compression
                if compression_level != "aggressive":
                    compressed_section = await self._compress_single_section(
                        section, "aggressive", max_tokens // 2, critical_elements
                    )
                    section_tokens = self._estimate_token_count(compressed_section['content'])
                    
                    if current_token_count + section_tokens <= strategy.target_token_count:
                        compressed_sections.append(compressed_section)
                        current_token_count += section_tokens
        
        logger.info(f"Compressed {len(compressed_sections)} sections, estimated tokens: {current_token_count}")
        return compressed_sections
    
    async def _compress_single_section(
        self, 
        section: DocumentSection,
        compression_level: str,
        max_tokens: int,
        critical_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compress a single section."""
        
        if compression_level == "minimal":
            # Just truncate if too long
            content = section.content
            if self._estimate_token_count(content) > max_tokens:
                # Keep first part and add summary
                truncated = content[:max_tokens * 3]  # Rough char to token ratio
                content = truncated + "\n[Content truncated - key points preserved above]"
        
        elif compression_level == "moderate":
            # Summarize while preserving key elements
            content = await self._summarize_section(section, max_tokens, critical_elements)
        
        else:  # aggressive
            # Extract only the most critical information
            content = await self._extract_key_points(section, max_tokens, critical_elements)
        
        return {
            'document_title': section.document_title,
            'section_title': section.section_title,
            'content': content,
            'compression_level': compression_level,
            'original_length': len(section.content),
            'compressed_length': len(content)
        }

    async def _summarize_section(
        self,
        section: DocumentSection,
        max_tokens: int,
        critical_elements: Dict[str, Any]
    ) -> str:
        """Summarize a section while preserving critical elements."""

        # Create prompt for summarization
        critical_entities = critical_elements.get('legal_entities', [])
        critical_concepts = critical_elements.get('key_concepts', [])

        prompt = f"""Summarize this legal document section while preserving all critical information:

Document: {section.document_title}
Section: {section.section_title}

CRITICAL ELEMENTS TO PRESERVE:
- Legal entities: {', '.join(critical_entities[:10])}
- Key concepts: {', '.join(critical_concepts[:10])}

CONTENT:
{section.content}

Requirements:
1. Preserve all legal entities, dates, and obligations mentioned
2. Maintain legal terminology and specific requirements
3. Keep cross-references to other sections/documents
4. Summarize background/explanatory text
5. Maximum length: {max_tokens * 3} characters

Provide a concise but legally complete summary:"""

        try:
            messages = [ChatMessage(role="user", content=prompt)]

            # Use higher token limit for summarization to avoid truncation
            summary_max_tokens = max(max_tokens, 1000)  # Ensure minimum of 1000 tokens for summaries

            response = await self.light_model_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=summary_max_tokens
            )

            # Enhanced validation for summary responses
            if response.finish_reason in ['length', 'max_tokens']:
                logger.warning(f"Summary for section '{section.section_title}' was truncated. "
                             f"Consider increasing max_tokens. Current length: {len(response.content)}")

            if not response.content or len(response.content.strip()) < 20:
                logger.warning(f"Summary for section '{section.section_title}' appears too short or empty. "
                             f"Content: '{response.content[:50]}...', "
                             f"Finish reason: {response.finish_reason}")

            return response.content
        except Exception as e:
            logger.error(f"Error summarizing section {section.section_title}: {e}")
            # Fallback to truncation
            return section.content[:max_tokens * 3] + "\n[Summary failed - content truncated]"

    async def _extract_key_points(
        self,
        section: DocumentSection,
        max_tokens: int,
        critical_elements: Dict[str, Any]
    ) -> str:
        """Extract only the most critical points from a section."""

        prompt = f"""Extract ONLY the most critical legal points from this section:

Document: {section.document_title}
Section: {section.section_title}

CONTENT:
{section.content}

Extract only:
1. Specific obligations and requirements
2. Rights and permissions
3. Deadlines and time periods
4. Legal entities and their roles
5. Cross-references to other documents

Format as bullet points. Maximum: {max_tokens * 3} characters."""

        try:
            messages = [ChatMessage(role="user", content=prompt)]

            # Use higher token limit for key point extraction
            extraction_max_tokens = max(max_tokens, 800)  # Ensure minimum of 800 tokens

            response = await self.light_model_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=extraction_max_tokens
            )

            # Enhanced validation for key point extraction
            if response.finish_reason in ['length', 'max_tokens']:
                logger.warning(f"Key point extraction for section '{section.section_title}' was truncated. "
                             f"Consider increasing max_tokens. Current length: {len(response.content)}")

            if not response.content or len(response.content.strip()) < 10:
                logger.warning(f"Key point extraction for section '{section.section_title}' appears too short or empty. "
                             f"Content: '{response.content[:50]}...', "
                             f"Finish reason: {response.finish_reason}")

            return response.content
        except Exception as e:
            logger.error(f"Error extracting key points from {section.section_title}: {e}")
            # Fallback to first part of content
            return section.content[:max_tokens * 2] + "\n[Key point extraction failed]"

    async def _build_compressed_content(
        self,
        compressed_sections: List[Dict[str, Any]],
        critical_elements: Dict[str, Any],
        processed_context: ProcessedContext
    ) -> str:
        """Build the final compressed content."""

        content_parts = []

        # Add critical elements summary
        content_parts.append("=== CRITICAL LEGAL ELEMENTS ===")

        if critical_elements.get('legal_entities'):
            content_parts.append(f"Legal Entities: {', '.join(critical_elements['legal_entities'][:20])}")

        if critical_elements.get('temporal_markers'):
            temporal_summary = []
            for marker in critical_elements['temporal_markers'][:10]:
                temporal_summary.append(f"{marker.get('type', 'date')}: {marker.get('date', 'N/A')} - {marker.get('description', 'N/A')}")
            content_parts.append(f"Key Dates/Deadlines:\n" + "\n".join(temporal_summary))

        if critical_elements.get('key_legal_points'):
            content_parts.append(f"Key Legal Points:\n" + "\n".join(f"- {point}" for point in critical_elements['key_legal_points'][:15]))

        # Add relationships summary
        if processed_context.relationships:
            content_parts.append("\n=== CROSS-DOCUMENT RELATIONSHIPS ===")
            for rel in processed_context.relationships[:10]:
                content_parts.append(f"- {rel.get('relationship_type', 'related')}: {rel.get('section1_id', '')} ↔ {rel.get('section2_id', '')}")

        # Add compressed sections
        content_parts.append("\n=== DOCUMENT SECTIONS ===")

        for section in compressed_sections:
            content_parts.append(f"\n--- {section['document_title']}: {section['section_title']} ---")
            content_parts.append(f"[Compression: {section['compression_level']}]")
            content_parts.append(section['content'])

        return "\n\n".join(content_parts)

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is a simplification - real tokenization would be more accurate
        return len(text) // 4
