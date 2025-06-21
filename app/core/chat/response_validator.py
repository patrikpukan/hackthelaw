"""
Response validation utilities for LLM responses.

This module provides utilities to validate and assess the quality of LLM responses,
particularly for document summarization and legal analysis tasks.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from app.core.chat.llm_client import LLMResponse

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    """Response quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """Result of response validation."""
    quality: ResponseQuality
    is_valid: bool
    is_truncated: bool
    is_empty: bool
    content_length: int
    finish_reason: Optional[str]
    issues: List[str]
    suggestions: List[str]


class ResponseValidator:
    """Validates and assesses LLM response quality."""
    
    def __init__(self):
        self.min_content_length = 20
        self.min_summary_length = 100
        self.truncation_indicators = [
            "length", "max_tokens", "stop_sequence"
        ]
    
    def validate_response(
        self, 
        response: LLMResponse, 
        query: str = "", 
        expected_type: str = "general"
    ) -> ValidationResult:
        """
        Validate an LLM response and assess its quality.
        
        Args:
            response: The LLM response to validate
            query: The original query (for context)
            expected_type: Type of response expected (general, summary, analysis)
            
        Returns:
            ValidationResult with quality assessment and suggestions
        """
        issues = []
        suggestions = []
        
        # Basic validation
        is_empty = not response.content or len(response.content.strip()) == 0
        content_length = len(response.content) if response.content else 0
        is_truncated = response.finish_reason in self.truncation_indicators
        
        # Check for empty response
        if is_empty:
            issues.append("Response is empty or contains only whitespace")
            suggestions.append("Check LLM configuration and input prompt")
            return ValidationResult(
                quality=ResponseQuality.FAILED,
                is_valid=False,
                is_truncated=is_truncated,
                is_empty=is_empty,
                content_length=content_length,
                finish_reason=response.finish_reason,
                issues=issues,
                suggestions=suggestions
            )
        
        # Check minimum length requirements
        min_length = self._get_min_length_for_type(expected_type)
        if content_length < min_length:
            issues.append(f"Response too short ({content_length} chars, expected at least {min_length})")
            suggestions.append(f"Increase max_tokens or simplify the query")
        
        # Check for truncation
        if is_truncated:
            issues.append(f"Response was truncated (finish_reason: {response.finish_reason})")
            suggestions.append("Increase max_tokens parameter or break down the query")
            
            # Severe truncation check
            if content_length < min_length * 0.5:
                issues.append("Response appears severely truncated")
                suggestions.append("Significantly increase max_tokens or use a different approach")
        
        # Check for incomplete sentences (common in truncated responses)
        if self._has_incomplete_sentences(response.content):
            issues.append("Response appears to end mid-sentence")
            suggestions.append("Increase max_tokens to allow complete responses")
        
        # Check for summarization-specific issues
        if expected_type == "summary":
            summary_issues, summary_suggestions = self._validate_summary(response.content, query)
            issues.extend(summary_issues)
            suggestions.extend(summary_suggestions)
        
        # Determine overall quality
        quality = self._assess_quality(
            content_length, is_truncated, len(issues), expected_type
        )
        
        is_valid = quality not in [ResponseQuality.FAILED, ResponseQuality.POOR]
        
        return ValidationResult(
            quality=quality,
            is_valid=is_valid,
            is_truncated=is_truncated,
            is_empty=is_empty,
            content_length=content_length,
            finish_reason=response.finish_reason,
            issues=issues,
            suggestions=suggestions
        )
    
    def _get_min_length_for_type(self, expected_type: str) -> int:
        """Get minimum expected length for different response types."""
        type_minimums = {
            "summary": self.min_summary_length,
            "analysis": 150,
            "general": self.min_content_length
        }
        return type_minimums.get(expected_type, self.min_content_length)
    
    def _has_incomplete_sentences(self, content: str) -> bool:
        """Check if content appears to end mid-sentence."""
        if not content:
            return False
        
        content = content.strip()
        if not content:
            return False
        
        # Check if ends with sentence terminators
        sentence_endings = ['.', '!', '?', ':', ';']
        if content[-1] not in sentence_endings:
            # Check if it's just a word fragment
            last_words = content.split()[-3:] if len(content.split()) >= 3 else content.split()
            if len(last_words) > 0 and len(last_words[-1]) > 2:
                return True
        
        return False
    
    def _validate_summary(self, content: str, query: str) -> tuple[List[str], List[str]]:
        """Validate summary-specific requirements."""
        issues = []
        suggestions = []
        
        # Check for summary indicators
        summary_indicators = [
            "summary", "summarize", "key points", "main points", 
            "overview"
        ]
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        is_summary_request = any(indicator in query_lower for indicator in summary_indicators)
        
        if is_summary_request:
            # Check if response actually provides a summary structure
            if not any(indicator in content_lower for indicator in ["summary", "key", "main", "overview"]):
                issues.append("Response doesn't appear to be structured as a summary")
                suggestions.append("Ensure the prompt explicitly requests summary format")
            
            # Check for bullet points or structured format
            if "•" not in content and "-" not in content and "1." not in content:
                suggestions.append("Consider requesting structured format with bullet points")
        
        return issues, suggestions
    
    def _assess_quality(
        self, 
        content_length: int, 
        is_truncated: bool, 
        issue_count: int, 
        expected_type: str
    ) -> ResponseQuality:
        """Assess overall response quality."""
        
        min_length = self._get_min_length_for_type(expected_type)
        
        # Failed responses
        if content_length == 0:
            return ResponseQuality.FAILED
        
        # Poor responses
        if content_length < min_length * 0.3 or issue_count >= 3:
            return ResponseQuality.POOR
        
        # Acceptable responses
        if is_truncated or content_length < min_length or issue_count >= 2:
            return ResponseQuality.ACCEPTABLE
        
        # Good responses
        if issue_count == 1 or content_length < min_length * 2:
            return ResponseQuality.GOOD
        
        # Excellent responses
        return ResponseQuality.EXCELLENT
    
    def log_validation_result(self, result: ValidationResult, query: str = ""):
        """Log validation results for debugging."""
        
        log_level = logging.WARNING if not result.is_valid else logging.INFO
        
        logger.log(
            log_level,
            f"Response validation: quality={result.quality.value}, "
            f"valid={result.is_valid}, truncated={result.is_truncated}, "
            f"length={result.content_length}, finish_reason={result.finish_reason}"
        )
        
        if result.issues:
            logger.warning(f"Response issues: {', '.join(result.issues)}")
        
        if result.suggestions:
            logger.info(f"Suggestions: {', '.join(result.suggestions)}")


# Global validator instance
response_validator = ResponseValidator()
