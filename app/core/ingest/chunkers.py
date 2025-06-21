from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re
from dataclasses import dataclass

from app.utils.config import settings


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_char: int
    end_char: int
    chunk_type: str = "paragraph"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into chunks."""
        pass


class LegalClauseChunker(BaseChunker):
    """Chunker that identifies legal clauses using pattern matching."""
    
    def __init__(self):
        # Legal clause patterns
        self.clause_patterns = [
            r"Section\s+\d+\.?\s*",
            r"Article\s+\d+\.?\s*",
            r"Clause\s+\d+\.?\s*",
            r"\d+\.\s+[A-Z]",
            r"WHEREAS,?\s*",
            r"NOW,?\s+THEREFORE,?\s*",
            r"IN\s+WITNESS\s+WHEREOF,?\s*",
            r"DEFINITIONS\.?\s*",
            r"TERM\.?\s*",
            r"PAYMENT\.?\s*",
            r"LIABILITY\.?\s*",
            r"TERMINATION\.?\s*",
            r"CONFIDENTIALITY\.?\s*",
            r"GOVERNING\s+LAW\.?\s*"
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.clause_patterns]
    
    def identify_clause_boundaries(self, text: str) -> List[int]:
        """Identify positions where legal clauses begin."""
        boundaries = [0]  # Start of document
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                position = match.start()
                if position not in boundaries:
                    boundaries.append(position)
        
        boundaries.append(len(text))  # End of document
        return sorted(boundaries)
    
    def classify_clause_type(self, text: str) -> str:
        """Classify the type of legal clause."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["payment", "pay", "compensation", "fee"]):
            return "payment"
        elif any(keyword in text_lower for keyword in ["termination", "terminate", "end", "expiry"]):
            return "termination"
        elif any(keyword in text_lower for keyword in ["liability", "liable", "damages", "indemnity"]):
            return "liability"
        elif any(keyword in text_lower for keyword in ["confidential", "non-disclosure", "proprietary"]):
            return "confidentiality"
        elif any(keyword in text_lower for keyword in ["governing", "jurisdiction", "applicable law"]):
            return "governing_law"
        elif any(keyword in text_lower for keyword in ["whereas", "background", "recital"]):
            return "recital"
        elif any(keyword in text_lower for keyword in ["definition", "defined", "meaning"]):
            return "definition"
        else:
            return "general"
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into legal clause chunks."""
        boundaries = self.identify_clause_boundaries(text)
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 50:  # Minimum chunk size
                clause_type = self.classify_clause_type(chunk_text)
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    chunk_type=clause_type,
                    metadata={
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)
        
        return chunks


class SentenceChunker(BaseChunker):
    """Chunker that splits text by sentences with overlap."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into overlapping sentence-based chunks."""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, create chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_type="paragraph",
                    metadata={
                        "sentence_count": len(current_chunk),
                        "word_count": current_size
                    }
                )
                chunks.append(chunk)
                
                # Handle overlap
                overlap_sentences = current_chunk[-self.overlap:] if self.overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
                start_pos = end_pos - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            
            chunk = TextChunk(
                text=chunk_text,
                start_char=start_pos,
                end_char=end_pos,
                chunk_type="paragraph",
                metadata={
                    "sentence_count": len(current_chunk),
                    "word_count": current_size
                }
            )
            chunks.append(chunk)
        
        return chunks


class ParagraphChunker(BaseChunker):
    """Chunker that splits text by paragraphs."""
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into paragraph chunks."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 50:  # Minimum paragraph size
                chunk = TextChunk(
                    text=paragraph,
                    start_char=current_pos,
                    end_char=current_pos + len(paragraph),
                    chunk_type="paragraph",
                    metadata={
                        "word_count": len(paragraph.split()),
                        "line_count": len(paragraph.split('\n'))
                    }
                )
                chunks.append(chunk)
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return chunks


class HybridChunker(BaseChunker):
    """Hybrid chunker that combines legal clause detection with sentence chunking."""
    
    def __init__(self):
        self.clause_chunker = LegalClauseChunker()
        self.sentence_chunker = SentenceChunker()
        self.max_chunk_size = settings.chunk_size * 2  # Larger chunks for legal clauses
    
    def chunk(self, text: str) -> List[TextChunk]:
        """Use hybrid approach: legal clauses first, then sentence chunking for large clauses."""
        # First, try to identify legal clauses
        legal_chunks = self.clause_chunker.chunk(text)
        
        final_chunks = []
        
        for legal_chunk in legal_chunks:
            # If legal chunk is too large, split it further using sentence chunking
            if len(legal_chunk.text.split()) > self.max_chunk_size:
                sentence_chunks = self.sentence_chunker.chunk(legal_chunk.text)
                
                # Adjust positions relative to original text
                for sentence_chunk in sentence_chunks:
                    adjusted_chunk = TextChunk(
                        text=sentence_chunk.text,
                        start_char=legal_chunk.start_char + sentence_chunk.start_char,
                        end_char=legal_chunk.start_char + sentence_chunk.end_char,
                        chunk_type=legal_chunk.chunk_type,  # Keep legal clause type
                        metadata={
                            **sentence_chunk.metadata,
                            "parent_clause_type": legal_chunk.chunk_type,
                            "is_subchunk": True
                        }
                    )
                    final_chunks.append(adjusted_chunk)
            else:
                final_chunks.append(legal_chunk)
        
        return final_chunks


class DocumentChunkerFactory:
    """Factory for creating appropriate document chunkers."""
    
    @staticmethod
    def get_chunker(strategy: str = "hybrid") -> BaseChunker:
        """Get chunker based on strategy."""
        if strategy == "legal":
            return LegalClauseChunker()
        elif strategy == "sentence":
            return SentenceChunker()
        elif strategy == "paragraph":
            return ParagraphChunker()
        elif strategy == "hybrid":
            return HybridChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    @staticmethod
    def chunk_document(text: str, strategy: str = "hybrid") -> List[TextChunk]:
        """Chunk document using specified strategy."""
        chunker = DocumentChunkerFactory.get_chunker(strategy)
        return chunker.chunk(text) 