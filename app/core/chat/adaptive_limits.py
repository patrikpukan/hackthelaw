"""
Adaptive limits system for efficient resource usage in RAG.
Dynamically adjusts iteration counts and chunk limits based on query quality and results.
"""

from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ConfidenceLevel(Enum):
    """Confidence levels for search results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AdaptiveLimits:
    """Dynamic limits for RAG operations."""
    max_iterations: int
    max_chunks_per_iteration: int
    min_confidence_threshold: float
    similarity_threshold: float
    early_stop_threshold: float


@dataclass
class SearchQuality:
    """Quality metrics for search results."""
    avg_similarity: float
    max_similarity: float
    chunk_count: int
    coverage_score: float
    confidence_level: ConfidenceLevel


class AdaptiveLimitsManager:
    """Manages dynamic limits for efficient resource usage."""
    
    def __init__(self):
        # Base limits for different query complexities
        self.base_limits = {
            QueryComplexity.SIMPLE: AdaptiveLimits(
                max_iterations=1,
                max_chunks_per_iteration=3,
                min_confidence_threshold=0.7,
                similarity_threshold=0.6,
                early_stop_threshold=0.8
            ),
            QueryComplexity.MODERATE: AdaptiveLimits(
                max_iterations=2,
                max_chunks_per_iteration=4,
                min_confidence_threshold=0.6,
                similarity_threshold=0.5,
                early_stop_threshold=0.75
            ),
            QueryComplexity.COMPLEX: AdaptiveLimits(
                max_iterations=3,
                max_chunks_per_iteration=5,
                min_confidence_threshold=0.5,
                similarity_threshold=0.4,
                early_stop_threshold=0.7
            ),
            QueryComplexity.VERY_COMPLEX: AdaptiveLimits(
                max_iterations=4,
                max_chunks_per_iteration=6,
                min_confidence_threshold=0.4,
                similarity_threshold=0.3,
                early_stop_threshold=0.65
            )
        }
    
    def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query to determine complexity level."""
        
        query_lower = query.lower()
        
        # Complexity indicators
        complex_indicators = [
            # Question types
            'how', 'why', 'explain', 'analyze', 'compare', 'evaluate',
            'what are the implications', 'what happens if', 'describe the relationship',
            
            # Legal complexity
            'compliance', 'liability', 'consequences', 'obligations', 'restrictions',
            'interpretation', 'enforcement', 'jurisdiction', 'applicable law',
            
            # Multi-part questions
            'and', 'or', 'also', 'additionally', 'furthermore', 'moreover',
            
            # Conceptual terms
            'overview', 'summary', 'comprehensive', 'detailed', 'thorough'
        ]
        
        simple_indicators = [
            # Direct questions
            'define', 'what is', 'who is', 'when', 'where',
            
            # Specific lookups
            'section', 'clause', 'article', 'paragraph',
            'date', 'deadline', 'term', 'definition'
        ]
        
        # Count indicators
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        
        # Word count factor
        word_count = len(query.split())
        
        # Calculate complexity score
        complexity_score = complex_count * 2 - simple_count
        
        if word_count > 20:
            complexity_score += 1
        elif word_count > 15:
            complexity_score += 0.5
        
        # Determine complexity level
        if complexity_score >= 3:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1 or word_count > 10:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def get_initial_limits(self, query: str) -> AdaptiveLimits:
        """Get initial limits based on query complexity."""
        
        complexity = self.analyze_query_complexity(query)
        limits = self.base_limits[complexity]
        
        logger.info(f"Query complexity: {complexity.value}, Initial limits: "
                   f"iterations={limits.max_iterations}, chunks={limits.max_chunks_per_iteration}")
        
        return limits
    
    def evaluate_search_quality(self, chunks: List[Dict], query: str) -> SearchQuality:
        """Evaluate the quality of search results."""
        
        if not chunks:
            return SearchQuality(
                avg_similarity=0.0,
                max_similarity=0.0,
                chunk_count=0,
                coverage_score=0.0,
                confidence_level=ConfidenceLevel.LOW
            )
        
        # Calculate similarity metrics
        similarities = [chunk.get('similarity_score', 0.0) for chunk in chunks]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        
        # Calculate coverage score (diversity of sources and types)
        unique_docs = set(chunk.get('document', {}).get('id') for chunk in chunks)
        unique_types = set(chunk.get('chunk', {}).get('chunk_type') for chunk in chunks)
        
        coverage_score = min(1.0, (len(unique_docs) * 0.3 + len(unique_types) * 0.2 + avg_similarity))
        
        # Determine confidence level
        if avg_similarity >= 0.8 and max_similarity >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif avg_similarity >= 0.6 and max_similarity >= 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif avg_similarity >= 0.4 and max_similarity >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return SearchQuality(
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            chunk_count=len(chunks),
            coverage_score=coverage_score,
            confidence_level=confidence_level
        )
    
    def should_continue_search(
        self, 
        current_iteration: int, 
        search_quality: SearchQuality,
        limits: AdaptiveLimits,
        query: str
    ) -> Tuple[bool, str]:
        """Determine if search should continue based on quality metrics."""
        
        # Check iteration limit
        if current_iteration >= limits.max_iterations:
            return False, f"Reached max iterations ({limits.max_iterations})"
        
        # Early stopping if results are very good
        if (search_quality.confidence_level == ConfidenceLevel.VERY_HIGH and 
            search_quality.avg_similarity >= limits.early_stop_threshold):
            return False, f"High quality results found (avg_sim: {search_quality.avg_similarity:.3f})"
        
        # Continue if results are poor
        if search_quality.confidence_level == ConfidenceLevel.LOW:
            return True, f"Low quality results, continuing search"
        
        # Continue if coverage is low but similarity is decent
        if (search_quality.avg_similarity >= limits.similarity_threshold and 
            search_quality.coverage_score < 0.5):
            return True, f"Good similarity but low coverage, continuing"
        
        # Stop if we have sufficient quality
        if (search_quality.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] and
            search_quality.chunk_count >= 3):
            return False, f"Sufficient quality achieved ({search_quality.confidence_level.value})"
        
        # Default: continue if we haven't reached minimum threshold
        if search_quality.avg_similarity < limits.min_confidence_threshold:
            return True, f"Below confidence threshold ({search_quality.avg_similarity:.3f} < {limits.min_confidence_threshold})"
        
        return False, f"Acceptable results found"
    
    def adjust_chunk_limit(
        self, 
        iteration: int, 
        search_quality: SearchQuality, 
        base_limit: int
    ) -> int:
        """Dynamically adjust chunk limit based on search quality."""
        
        # Increase chunks if results are poor
        if search_quality.confidence_level == ConfidenceLevel.LOW:
            return min(base_limit + 2, 8)  # Cap at 8
        
        # Decrease chunks if results are very good early
        if (iteration == 1 and 
            search_quality.confidence_level == ConfidenceLevel.VERY_HIGH):
            return max(base_limit - 1, 2)  # Minimum 2
        
        # Normal progression
        return base_limit
    
    def get_resource_usage_summary(
        self, 
        iterations_used: int, 
        total_chunks: int, 
        final_quality: SearchQuality
    ) -> Dict[str, Any]:
        """Generate resource usage summary."""
        
        efficiency_score = final_quality.avg_similarity / max(iterations_used, 1)
        
        return {
            "iterations_used": iterations_used,
            "total_chunks_retrieved": total_chunks,
            "final_avg_similarity": final_quality.avg_similarity,
            "final_confidence_level": final_quality.confidence_level.value,
            "efficiency_score": efficiency_score,
            "resource_usage": {
                "api_calls_estimated": iterations_used * 2,  # Query rewriting + response generation
                "search_operations": iterations_used,
                "efficiency_rating": "high" if efficiency_score > 0.6 else "medium" if efficiency_score > 0.3 else "low"
            }
        }


# Utility functions for integration
def create_adaptive_limits_manager() -> AdaptiveLimitsManager:
    """Factory function to create adaptive limits manager."""
    return AdaptiveLimitsManager()


def estimate_resource_cost(limits: AdaptiveLimits, iterations: int) -> Dict[str, float]:
    """Estimate resource cost for given limits and iterations."""
    
    estimated_chunks = limits.max_chunks_per_iteration * iterations
    estimated_api_calls = iterations * 3  # Query rewrite + search + response
    
    # Rough cost estimates (adjust based on your providers)
    cost_per_chunk = 0.001  # $0.001 per chunk processing
    cost_per_api_call = 0.01  # $0.01 per LLM API call
    
    return {
        "estimated_chunks": estimated_chunks,
        "estimated_api_calls": estimated_api_calls,
        "estimated_cost_usd": estimated_chunks * cost_per_chunk + estimated_api_calls * cost_per_api_call,
        "time_estimate_seconds": iterations * 2.5  # Rough estimate
    } 