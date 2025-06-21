"""
Vertex AI specific helper utilities.

This module provides utilities to handle Vertex AI specific issues,
particularly around empty responses and safety filtering.
"""

import logging
from typing import List, Optional, Dict, Any
import re

logger = logging.getLogger(__name__)


class VertexAIHelper:
    """Helper class for Vertex AI specific operations."""
    
    @staticmethod
    def simplify_prompt_for_vertex_ai(prompt: str) -> str:
        """
        Simplify prompts for better Vertex AI compatibility.
        
        Vertex AI sometimes has issues with complex prompts or specific formatting.
        This function simplifies prompts to improve success rate.
        """
        # Remove complex JSON formatting instructions that might confuse Vertex AI
        simplified = prompt
        
        # Replace complex JSON instructions with simpler ones
        json_patterns = [
            r'```json\s*\{[^}]*\}\s*```',
            r'Respond with JSON:.*?```',
            r'```json.*?```',
        ]
        
        for pattern in json_patterns:
            simplified = re.sub(pattern, 'Respond with a simple list.', simplified, flags=re.DOTALL)
        
        # Simplify complex instructions
        if 'JSON' in simplified and len(simplified) > 1000:
            # For very long prompts with JSON, use a much simpler approach
            if 'search queries' in simplified.lower():
                simplified = f"Convert this question into 2-3 search terms: {simplified.split('User Question:')[-1].split('Respond')[0].strip()}"
            elif 'sufficient' in simplified.lower():
                simplified = f"Is this information sufficient to answer the question? Answer yes or no: {simplified.split('User Question:')[-1].split('Respond')[0].strip()}"
        
        # Remove excessive formatting
        simplified = re.sub(r'\n{3,}', '\n\n', simplified)
        simplified = re.sub(r'[*]{2,}', '', simplified)
        
        return simplified.strip()
    
    @staticmethod
    def extract_queries_from_simple_response(response_content: str, original_query: str) -> List[str]:
        """
        Extract search queries from a simplified Vertex AI response.
        
        When JSON parsing fails, try to extract meaningful queries from the response.
        """
        if not response_content:
            return [original_query]
        
        # Try to find quoted strings first
        quoted_queries = re.findall(r'"([^"]+)"', response_content)
        if quoted_queries:
            return [q.strip() for q in quoted_queries if q.strip()]
        
        # Try to find numbered lists
        numbered_queries = re.findall(r'\d+\.\s*([^\n]+)', response_content)
        if numbered_queries:
            return [q.strip() for q in numbered_queries if q.strip()]
        
        # Try to find bullet points
        bullet_queries = re.findall(r'[-*•]\s*([^\n]+)', response_content)
        if bullet_queries:
            return [q.strip() for q in bullet_queries if q.strip()]
        
        # Split by lines and take meaningful ones
        lines = [line.strip() for line in response_content.split('\n') if line.strip()]
        meaningful_lines = []
        
        for line in lines:
            # Skip very short or very long lines
            if 5 <= len(line) <= 100:
                # Skip lines that look like instructions
                if not any(word in line.lower() for word in ['here', 'these', 'following', 'above', 'below']):
                    meaningful_lines.append(line)
        
        if meaningful_lines:
            return meaningful_lines[:3]  # Take up to 3 queries
        
        # Fallback: return original query
        return [original_query]
    
    @staticmethod
    def is_vertex_ai_safety_block(response) -> bool:
        """
        Check if a response was blocked by Vertex AI safety filters.
        """
        if not response:
            return False
        
        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return True
        
        candidate = response.candidates[0]
        
        # Check finish reason
        if hasattr(candidate, 'finish_reason'):
            finish_reason = candidate.finish_reason
            if hasattr(finish_reason, 'name'):
                reason_name = finish_reason.name.lower()
                if 'safety' in reason_name or 'blocked' in reason_name:
                    return True
        
        # Check safety ratings
        if hasattr(candidate, 'safety_ratings'):
            for rating in candidate.safety_ratings:
                if hasattr(rating, 'blocked') and rating.blocked:
                    return True
                if hasattr(rating, 'probability'):
                    prob = rating.probability
                    if hasattr(prob, 'name') and 'high' in prob.name.lower():
                        return True
        
        return False
    
    @staticmethod
    def create_fallback_queries(original_query: str) -> List[str]:
        """
        Create fallback search queries when Vertex AI fails completely.
        """
        # Extract key terms from the original query
        query_lower = original_query.lower()
        
        # Common legal and document terms
        legal_terms = ['contract', 'agreement', 'obligation', 'requirement', 'law', 'legal', 'clause', 'section']
        action_terms = ['analyze', 'summarize', 'explain', 'describe', 'find', 'identify']
        
        # Build fallback queries
        fallback_queries = []
        
        # Add the original query as-is
        fallback_queries.append(original_query)
        
        # Extract key nouns and terms
        words = re.findall(r'\b\w{4,}\b', original_query)  # Words with 4+ characters
        if words:
            # Create a query with key terms
            key_terms = [word for word in words if word.lower() not in ['this', 'that', 'what', 'how', 'when', 'where']]
            if key_terms:
                fallback_queries.append(' '.join(key_terms[:3]))
        
        # Add specific legal query if it seems legal-related
        if any(term in query_lower for term in legal_terms):
            fallback_queries.append('legal obligations requirements')
        
        # Add summarization query if it seems like a summary request
        if any(term in query_lower for term in ['summary', 'summarize', 'overview']):
            fallback_queries.append('key points main obligations')
        
        return fallback_queries[:3]  # Return up to 3 fallback queries
    
    @staticmethod
    def log_vertex_ai_issue(response, prompt_length: int, query: str):
        """
        Log detailed information about Vertex AI issues for debugging.
        """
        logger.error(f"Vertex AI Issue Debug Info:")
        logger.error(f"  Query: {query[:100]}...")
        logger.error(f"  Prompt length: {prompt_length}")
        
        if response:
            logger.error(f"  Response object exists: True")
            
            if hasattr(response, 'candidates'):
                logger.error(f"  Candidates count: {len(response.candidates) if response.candidates else 0}")
                
                if response.candidates:
                    candidate = response.candidates[0]
                    logger.error(f"  Finish reason: {getattr(candidate, 'finish_reason', 'Unknown')}")
                    
                    if hasattr(candidate, 'safety_ratings'):
                        logger.error(f"  Safety ratings: {candidate.safety_ratings}")
                    
                    if hasattr(candidate, 'content'):
                        content_info = "None"
                        if candidate.content:
                            if hasattr(candidate.content, 'parts'):
                                content_info = f"Parts count: {len(candidate.content.parts) if candidate.content.parts else 0}"
                            elif hasattr(candidate.content, 'text'):
                                content_info = f"Text length: {len(candidate.content.text) if candidate.content.text else 0}"
                        logger.error(f"  Content: {content_info}")
            
            if hasattr(response, 'usage_metadata'):
                logger.error(f"  Usage metadata: {response.usage_metadata}")
        else:
            logger.error(f"  Response object exists: False")


# Global helper instance
vertex_ai_helper = VertexAIHelper()
