"""
Token Management System for Multi-Stage Legal Analysis

This module ensures that the final input to Gemini 2.5 Pro stays within the 300k token limit
while intelligently prioritizing the most important content.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken

from app.core.analysis.intelligent_compressor import CompressionResult

logger = logging.getLogger(__name__)


class TokenPriority(Enum):
    """Priority levels for different types of content."""
    CRITICAL = 1      # Must be included (legal entities, obligations)
    HIGH = 2          # Should be included (key concepts, relationships)
    MEDIUM = 3        # Nice to have (supporting context)
    LOW = 4           # Can be omitted (background information)


@dataclass
class TokenBudget:
    """Token budget allocation for different content types."""
    total_limit: int = 300000
    system_prompt: int = 2000
    query_context: int = 1000
    critical_elements: int = 50000
    legal_analysis: int = 100000
    document_content: int = 140000
    response_buffer: int = 7000  # Reserve for response generation


@dataclass
class ContentBlock:
    """A block of content with priority and token information."""
    content: str
    priority: TokenPriority
    content_type: str
    token_count: int
    metadata: Dict[str, Any]


@dataclass
class TokenAllocation:
    """Result of token allocation process."""
    allocated_content: str
    total_tokens: int
    content_blocks: List[ContentBlock]
    allocation_metadata: Dict[str, Any]
    truncated_content: List[ContentBlock]


class TokenManager:
    """
    Manages token allocation to ensure final input stays within limits.
    
    The manager:
    1. Accurately counts tokens using tiktoken
    2. Prioritizes content by importance
    3. Applies intelligent truncation strategies
    4. Ensures critical legal information is preserved
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize with specific model for accurate token counting."""
        self.model_name = model_name

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.use_tiktoken = True
        except KeyError:
            # Fallback to a common encoding for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
            logger.info(f"Model {model_name} not found in tiktoken, using cl100k_base encoding for estimation")

        self.budget = TokenBudget()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with model-specific adjustments."""
        if not text:
            return 0

        try:
            if self.use_tiktoken:
                base_count = len(self.encoding.encode(text))

                # Apply model-specific adjustments for better estimation
                if "gemini" in self.model_name.lower():
                    # Gemini models tend to have slightly different tokenization
                    # Apply a small adjustment factor based on empirical observations
                    return int(base_count * 0.95)  # Gemini tends to use slightly fewer tokens
                else:
                    return base_count
            else:
                # Fallback estimation method
                return self._estimate_tokens_fallback(text)

        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return self._estimate_tokens_fallback(text)

    def _estimate_tokens_fallback(self, text: str) -> int:
        """Fallback token estimation when tiktoken fails."""
        # More sophisticated fallback than simple division
        # Based on average token-to-character ratios for different content types

        # Count words and characters
        word_count = len(text.split())
        char_count = len(text)

        # Different estimation strategies based on content characteristics
        if char_count == 0:
            return 0

        # For very short text, use character-based estimation
        if char_count < 50:
            return max(1, char_count // 3)

        # For normal text, use a hybrid approach
        # English text typically has 3-4 characters per token
        char_based = char_count // 4

        # Word-based estimation (English averages ~1.3 tokens per word)
        word_based = int(word_count * 1.3)

        # Use the average of both methods for better accuracy
        estimated = (char_based + word_based) // 2

        return max(1, estimated)
    
    async def allocate_tokens(
        self, 
        compression_result: CompressionResult,
        query: str,
        system_prompt: str,
        additional_context: Optional[str] = None
    ) -> TokenAllocation:
        """
        Allocate tokens intelligently across different content types.
        
        Args:
            compression_result: Result from intelligent compression
            query: User query
            system_prompt: System prompt for the final model
            additional_context: Any additional context to include
            
        Returns:
            TokenAllocation with optimized content distribution
        """
        logger.info("Starting token allocation process")
        
        # Step 1: Count fixed tokens
        fixed_tokens = (
            self.count_tokens(system_prompt) +
            self.count_tokens(query) +
            self.budget.response_buffer
        )
        
        if additional_context:
            fixed_tokens += self.count_tokens(additional_context)
        
        available_tokens = self.budget.total_limit - fixed_tokens
        
        if available_tokens <= 0:
            raise ValueError("System prompt and query exceed token limit")
        
        logger.info(f"Available tokens for content: {available_tokens}")
        
        # Step 2: Create content blocks with priorities
        content_blocks = await self._create_content_blocks(compression_result)
        
        # Step 3: Allocate tokens by priority
        allocated_blocks, truncated_blocks = await self._allocate_by_priority(
            content_blocks, available_tokens
        )
        
        # Step 4: Build final content
        final_content = await self._build_final_content(
            allocated_blocks, system_prompt, query, additional_context
        )
        
        final_token_count = self.count_tokens(final_content)
        
        allocation_metadata = {
            'total_content_blocks': len(content_blocks),
            'allocated_blocks': len(allocated_blocks),
            'truncated_blocks': len(truncated_blocks),
            'fixed_tokens': fixed_tokens,
            'available_tokens': available_tokens,
            'final_token_count': final_token_count,
            'token_efficiency': final_token_count / self.budget.total_limit
        }
        
        return TokenAllocation(
            allocated_content=final_content,
            total_tokens=final_token_count,
            content_blocks=allocated_blocks,
            allocation_metadata=allocation_metadata,
            truncated_content=truncated_blocks
        )
    
    async def _create_content_blocks(
        self, 
        compression_result: CompressionResult
    ) -> List[ContentBlock]:
        """Create prioritized content blocks from compression result."""
        
        blocks = []
        
        # Critical elements (highest priority)
        if compression_result.preserved_elements:
            critical_content = self._format_critical_elements(compression_result)
            blocks.append(ContentBlock(
                content=critical_content,
                priority=TokenPriority.CRITICAL,
                content_type="critical_elements",
                token_count=self.count_tokens(critical_content),
                metadata={"source": "preserved_elements"}
            ))
        
        # Key legal points (critical priority)
        if compression_result.key_legal_points:
            legal_points_content = "\n".join([
                f"• {point}" for point in compression_result.key_legal_points
            ])
            blocks.append(ContentBlock(
                content=f"KEY LEGAL POINTS:\n{legal_points_content}",
                priority=TokenPriority.CRITICAL,
                content_type="legal_points",
                token_count=self.count_tokens(legal_points_content),
                metadata={"count": len(compression_result.key_legal_points)}
            ))
        
        # Relationships (high priority)
        if compression_result.critical_relationships:
            relationships_content = self._format_relationships(
                compression_result.critical_relationships
            )
            blocks.append(ContentBlock(
                content=relationships_content,
                priority=TokenPriority.HIGH,
                content_type="relationships",
                token_count=self.count_tokens(relationships_content),
                metadata={"count": len(compression_result.critical_relationships)}
            ))
        
        # Main compressed content (split by priority)
        content_lines = compression_result.compressed_content.split('\n')
        current_section = []
        current_priority = TokenPriority.MEDIUM
        
        for line in content_lines:
            if line.startswith('=== CRITICAL'):
                current_priority = TokenPriority.CRITICAL
            elif line.startswith('=== CROSS-DOCUMENT'):
                current_priority = TokenPriority.HIGH
            elif line.startswith('=== DOCUMENT SECTIONS'):
                current_priority = TokenPriority.MEDIUM
            elif line.startswith('---') and current_section:
                # End of section, create block
                section_content = '\n'.join(current_section)
                if section_content.strip():
                    blocks.append(ContentBlock(
                        content=section_content,
                        priority=current_priority,
                        content_type="document_section",
                        token_count=self.count_tokens(section_content),
                        metadata={"section_type": "document_content"}
                    ))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Handle last section
        if current_section:
            section_content = '\n'.join(current_section)
            if section_content.strip():
                blocks.append(ContentBlock(
                    content=section_content,
                    priority=current_priority,
                    content_type="document_section",
                    token_count=self.count_tokens(section_content),
                    metadata={"section_type": "document_content"}
                ))
        
        return blocks
    
    async def _allocate_by_priority(
        self, 
        content_blocks: List[ContentBlock],
        available_tokens: int
    ) -> Tuple[List[ContentBlock], List[ContentBlock]]:
        """Allocate tokens by priority, ensuring critical content is included."""
        
        # Sort blocks by priority
        sorted_blocks = sorted(content_blocks, key=lambda x: x.priority.value)
        
        allocated_blocks = []
        truncated_blocks = []
        used_tokens = 0
        
        # First pass: allocate critical and high priority content
        for block in sorted_blocks:
            if block.priority in [TokenPriority.CRITICAL, TokenPriority.HIGH]:
                if used_tokens + block.token_count <= available_tokens:
                    allocated_blocks.append(block)
                    used_tokens += block.token_count
                else:
                    # Try to truncate critical content rather than drop it
                    if block.priority == TokenPriority.CRITICAL:
                        remaining_tokens = available_tokens - used_tokens
                        if remaining_tokens > 100:  # Minimum viable size
                            truncated_block = await self._truncate_block(
                                block, remaining_tokens
                            )
                            allocated_blocks.append(truncated_block)
                            used_tokens += truncated_block.token_count
                        else:
                            truncated_blocks.append(block)
                    else:
                        truncated_blocks.append(block)
        
        # Second pass: allocate medium and low priority content
        for block in sorted_blocks:
            if block.priority in [TokenPriority.MEDIUM, TokenPriority.LOW]:
                if used_tokens + block.token_count <= available_tokens:
                    allocated_blocks.append(block)
                    used_tokens += block.token_count
                else:
                    # Try partial allocation for medium priority
                    if block.priority == TokenPriority.MEDIUM:
                        remaining_tokens = available_tokens - used_tokens
                        if remaining_tokens > 200:  # Minimum viable size
                            truncated_block = await self._truncate_block(
                                block, remaining_tokens
                            )
                            allocated_blocks.append(truncated_block)
                            used_tokens += truncated_block.token_count
                            break  # No more room
                        else:
                            truncated_blocks.append(block)
                            break
                    else:
                        truncated_blocks.append(block)
                        break
        
        logger.info(f"Allocated {len(allocated_blocks)} blocks, truncated {len(truncated_blocks)}")
        logger.info(f"Used {used_tokens}/{available_tokens} tokens ({used_tokens/available_tokens*100:.1f}%)")

        return allocated_blocks, truncated_blocks

    async def _truncate_block(self, block: ContentBlock, max_tokens: int) -> ContentBlock:
        """Truncate a content block to fit within token limit."""

        if block.token_count <= max_tokens:
            return block

        # Calculate truncation ratio
        ratio = max_tokens / block.token_count
        target_chars = int(len(block.content) * ratio * 0.9)  # 90% to be safe

        # Truncate content intelligently
        if block.content_type == "document_section":
            # Try to truncate at sentence boundaries
            sentences = block.content.split('. ')
            truncated_content = ""

            for sentence in sentences:
                test_content = truncated_content + sentence + ". "
                if self.count_tokens(test_content) > max_tokens:
                    break
                truncated_content = test_content

            if not truncated_content:
                # Fallback to character truncation
                truncated_content = block.content[:target_chars] + "..."
        else:
            # Simple character truncation for other types
            truncated_content = block.content[:target_chars] + "..."

        return ContentBlock(
            content=truncated_content,
            priority=block.priority,
            content_type=block.content_type,
            token_count=self.count_tokens(truncated_content),
            metadata={**block.metadata, "truncated": True, "original_tokens": block.token_count}
        )

    def _format_critical_elements(self, compression_result: CompressionResult) -> str:
        """Format critical elements for inclusion."""

        parts = ["=== CRITICAL LEGAL ELEMENTS ==="]

        preserved = compression_result.preserved_elements

        if preserved.get('legal_entities', 0) > 0:
            parts.append(f"Legal Entities: {preserved['legal_entities']} identified")

        if preserved.get('temporal_markers', 0) > 0:
            parts.append(f"Temporal Markers: {preserved['temporal_markers']} dates/deadlines")

        if preserved.get('cross_references', 0) > 0:
            parts.append(f"Cross-References: {preserved['cross_references']} document links")

        if preserved.get('key_concepts', 0) > 0:
            parts.append(f"Key Concepts: {preserved['key_concepts']} legal concepts")

        parts.append(f"Compression Ratio: {compression_result.compression_ratio:.2f}")

        return "\n".join(parts)

    def _format_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Format relationships for inclusion."""

        parts = ["=== CROSS-DOCUMENT RELATIONSHIPS ==="]

        for rel in relationships[:10]:  # Limit to top 10
            rel_type = rel.get('relationship_type', 'related')
            section1 = rel.get('section1_id', 'Unknown')
            section2 = rel.get('section2_id', 'Unknown')
            strength = rel.get('strength', 0)

            parts.append(f"• {rel_type.title()}: {section1} ↔ {section2} (strength: {strength:.2f})")

        return "\n".join(parts)

    async def _build_final_content(
        self,
        allocated_blocks: List[ContentBlock],
        system_prompt: str,
        query: str,
        additional_context: Optional[str] = None
    ) -> str:
        """Build the final content string for the model."""

        parts = [system_prompt]

        if additional_context:
            parts.append(f"\nADDITIONAL CONTEXT:\n{additional_context}")

        # Add allocated content blocks in priority order
        sorted_blocks = sorted(allocated_blocks, key=lambda x: x.priority.value)

        for block in sorted_blocks:
            parts.append(f"\n{block.content}")

        parts.append(f"\nUSER QUERY: {query}")

        return "\n".join(parts)

    def get_token_usage_summary(self, allocation: TokenAllocation) -> Dict[str, Any]:
        """Get a summary of token usage."""

        usage_by_type = {}
        for block in allocation.content_blocks:
            content_type = block.content_type
            if content_type not in usage_by_type:
                usage_by_type[content_type] = {'tokens': 0, 'blocks': 0}
            usage_by_type[content_type]['tokens'] += block.token_count
            usage_by_type[content_type]['blocks'] += 1

        return {
            'total_tokens': allocation.total_tokens,
            'token_limit': self.budget.total_limit,
            'utilization': allocation.total_tokens / self.budget.total_limit,
            'usage_by_type': usage_by_type,
            'allocation_metadata': allocation.allocation_metadata
        }
