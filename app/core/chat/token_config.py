"""
Token configuration and management for different query types.

This module provides intelligent token allocation based on query analysis,
ensuring adequate response length for different types of requests.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that require different token allocations."""
    SUMMARIZATION = "summarization"
    LEGAL_ANALYSIS = "legal_analysis"
    DETAILED_EXPLANATION = "detailed_explanation"
    SIMPLE_QUESTION = "simple_question"
    COMPARISON = "comparison"
    EXTRACTION = "extraction"
    GENERAL = "general"


@dataclass
class TokenConfig:
    """Token configuration for different query types."""
    max_tokens: int
    min_expected_length: int
    description: str
    keywords: List[str]


class TokenConfigManager:
    """Manages token allocation based on query analysis."""
    
    def __init__(self):
        self.configs = {
            QueryType.SUMMARIZATION: TokenConfig(
                max_tokens=4000,
                min_expected_length=200,
                description="Document summarization and overview tasks",
                keywords=[
                    "summarize", "summary", "overview", "key points", "main points"
                ]
            ),
            QueryType.LEGAL_ANALYSIS: TokenConfig(
                max_tokens=5000,
                min_expected_length=300,
                description="Legal analysis and interpretation tasks",
                keywords=[
                    "analyze", "analysis", "legal implications", "obligations",
                    "requirements", "compliance", "violation", "contract", "agreement"
                ]
            ),
            QueryType.DETAILED_EXPLANATION: TokenConfig(
                max_tokens=3500,
                min_expected_length=250,
                description="Detailed explanations and how-to questions",
                keywords=[
                    "explain", "how to", "process", "procedure", "steps",
                    "detailed", "comprehensive", "thorough"
                ]
            ),
            QueryType.COMPARISON: TokenConfig(
                max_tokens=3000,
                min_expected_length=200,
                description="Comparison and contrast tasks",
                keywords=[
                    "compare", "comparison", "difference", "similar",
                    "contrast", "versus", "vs"
                ]
            ),
            QueryType.EXTRACTION: TokenConfig(
                max_tokens=2000,
                min_expected_length=100,
                description="Information extraction tasks",
                keywords=[
                    "extract", "find", "list", "identify", "locate", "search"
                ]
            ),
            QueryType.SIMPLE_QUESTION: TokenConfig(
                max_tokens=1500,
                min_expected_length=50,
                description="Simple factual questions",
                keywords=[
                    "what is", "who is", "when", "where", "which"
                ]
            ),
            QueryType.GENERAL: TokenConfig(
                max_tokens=2500,
                min_expected_length=100,
                description="General queries and default case",
                keywords=[]
            )
        }
        
        # Compile regex patterns for better performance
        self.keyword_patterns = {}
        for query_type, config in self.configs.items():
            if config.keywords:
                pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in config.keywords) + r')\b'
                self.keyword_patterns[query_type] = re.compile(pattern, re.IGNORECASE)
    
    def analyze_query_type(self, query: str) -> QueryType:
        """
        Analyze query to determine its type.
        
        Args:
            query: The user query to analyze
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower().strip()
        
        # Score each query type based on keyword matches
        scores = {}
        
        for query_type, pattern in self.keyword_patterns.items():
            matches = pattern.findall(query_lower)
            scores[query_type] = len(matches)
        
        # Find the query type with the highest score
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                logger.info(f"Detected query type: {best_type.value} (score: {scores[best_type]})")
                return best_type
        
        # Additional heuristics for query length and complexity
        word_count = len(query.split())
        
        if word_count <= 5:
            return QueryType.SIMPLE_QUESTION
        elif word_count >= 20:
            return QueryType.DETAILED_EXPLANATION
        
        logger.info(f"Using default query type: {QueryType.GENERAL.value}")
        return QueryType.GENERAL
    
    def get_token_config(self, query: str) -> TokenConfig:
        """
        Get token configuration for a query.
        
        Args:
            query: The user query
            
        Returns:
            TokenConfig object with appropriate settings
        """
        query_type = self.analyze_query_type(query)
        config = self.configs[query_type]
        
        logger.info(f"Using token config for {query_type.value}: "
                   f"max_tokens={config.max_tokens}, "
                   f"min_expected={config.min_expected_length}")
        
        return config
    
    def get_max_tokens(self, query: str, context_length: Optional[int] = None) -> int:
        """
        Get appropriate max_tokens for a query.
        
        Args:
            query: The user query
            context_length: Optional context length to consider
            
        Returns:
            Recommended max_tokens value
        """
        config = self.get_token_config(query)
        base_tokens = config.max_tokens
        
        # Adjust based on context length if provided
        if context_length:
            # If context is very long, we might need more tokens for comprehensive response
            if context_length > 10000:  # Very long context
                base_tokens = int(base_tokens * 1.3)
            elif context_length > 5000:  # Long context
                base_tokens = int(base_tokens * 1.1)
        
        # Ensure we don't exceed reasonable limits
        max_allowed = 8000  # Reasonable upper limit
        return min(base_tokens, max_allowed)
    
    def should_use_streaming(self, query: str) -> bool:
        """
        Determine if streaming should be used for this query type.
        
        Args:
            query: The user query
            
        Returns:
            True if streaming is recommended
        """
        query_type = self.analyze_query_type(query)
        
        # Use streaming for longer responses
        streaming_types = {
            QueryType.SUMMARIZATION,
            QueryType.LEGAL_ANALYSIS,
            QueryType.DETAILED_EXPLANATION
        }
        
        return query_type in streaming_types
    
    def get_retry_config(self, query: str, previous_length: int) -> Dict[str, Any]:
        """
        Get retry configuration if previous response was truncated.
        
        Args:
            query: The user query
            previous_length: Length of the previous truncated response
            
        Returns:
            Dictionary with retry configuration
        """
        config = self.get_token_config(query)
        
        # If previous response was much shorter than expected, increase tokens significantly
        if previous_length < config.min_expected_length * 0.5:
            new_max_tokens = int(config.max_tokens * 1.5)
        elif previous_length < config.min_expected_length:
            new_max_tokens = int(config.max_tokens * 1.2)
        else:
            new_max_tokens = int(config.max_tokens * 1.1)
        
        return {
            "max_tokens": min(new_max_tokens, 8000),
            "temperature": 0.2,  # Lower temperature for more focused retry
            "retry_reason": "previous_response_truncated"
        }


# Global token config manager instance
token_config_manager = TokenConfigManager()
