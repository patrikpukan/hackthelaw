"""
Enhanced citation system for precise legal document references.
Extracts and formats citations with article numbers, paragraphs, dates, and sections.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Structured citation with precise legal references."""
    document_id: str
    document_title: str
    section_number: Optional[str] = None
    article_number: Optional[str] = None  
    paragraph_number: Optional[str] = None
    subsection: Optional[str] = None
    clause: Optional[str] = None
    page_number: Optional[int] = None
    line_numbers: Optional[Tuple[int, int]] = None
    date_mentioned: Optional[str] = None
    effective_date: Optional[str] = None
    chunk_text: str = ""
    similarity_score: float = 0.0
    context_before: str = ""
    context_after: str = ""


@dataclass 
class ExtractedReferences:
    """Container for all extracted references from text."""
    articles: List[str]
    sections: List[str] 
    paragraphs: List[str]
    clauses: List[str]
    dates: List[str]
    page_references: List[int]
    definitions: List[str]


class EnhancedCitationExtractor:
    """Extracts precise legal citations from document chunks."""
    
    def __init__(self):
        # Regex patterns for legal references
        self.patterns = {
            # Article patterns
            'articles': [
                r'(?:Article|Art\.?|§)\s*(\d+(?:\.\d+)*)'
            ],
            
            # Section patterns
            'sections': [
                r'(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)'
            ],
            
            # Paragraph patterns
            'paragraphs': [
                r'(?:Paragraph|Para\.?|¶)\s*(\d+(?:\.\d+)*)'
            ],
            
            # Clause patterns
            'clauses': [
                r'(?:Clause|Cl\.?)\s*(\d+(?:\.\d+)*)'
            ],
            
            # Date patterns
            'dates': [
                r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b'
            ],
            
            # Page references
            'pages': [
                r'(?:page|pg\.?|p\.)\s*(\d+)'
            ],
        }
        
        # Legal definition patterns
        self.definition_patterns = [
            r'"([^"]+)"\s+means',
            r'"([^"]+)"\s+shall mean'
        ]
    
    def extract_references(self, text: str) -> ExtractedReferences:
        """Extract all legal references from text."""
        
        refs = ExtractedReferences(
            articles=[], sections=[], paragraphs=[], 
            clauses=[], dates=[], page_references=[], definitions=[]
        )
        
        # Extract each type of reference
        for pattern_list in self.patterns['articles']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.articles.extend(matches)
        
        for pattern_list in self.patterns['sections']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.sections.extend(matches)
        
        for pattern_list in self.patterns['paragraphs']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.paragraphs.extend(matches)
        
        for pattern_list in self.patterns['clauses']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.clauses.extend(matches)
        
        for pattern_list in self.patterns['dates']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.dates.extend(matches)
        
        for pattern_list in self.patterns['pages']:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            refs.page_references.extend([int(m) for m in matches])
        
        # Extract definitions
        for pattern in self.definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.definitions.extend(matches)
        
        # Remove duplicates
        refs.articles = list(set(refs.articles))
        refs.sections = list(set(refs.sections))
        refs.paragraphs = list(set(refs.paragraphs))
        refs.clauses = list(set(refs.clauses))
        refs.dates = list(set(refs.dates))
        refs.page_references = list(set(refs.page_references))
        refs.definitions = list(set(refs.definitions))
        
        return refs
    
    def create_citation(self, chunk: Dict[str, Any]) -> Citation:
        """Create enhanced citation from document chunk."""
        
        document = chunk.get('document', {})
        chunk_data = chunk.get('chunk', {})
        
        chunk_text = chunk_data.get('content', '')
        
        # Extract references from chunk text
        refs = self.extract_references(chunk_text)
        
        # Get context (previous and next sentences)
        context_before, context_after = self._extract_context(chunk_text)
        
        # Extract the most relevant reference numbers
        primary_article = refs.articles[0] if refs.articles else None
        primary_section = refs.sections[0] if refs.sections else None  
        primary_paragraph = refs.paragraphs[0] if refs.paragraphs else None
        primary_clause = refs.clauses[0] if refs.clauses else None
        
        # Extract dates
        primary_date = refs.dates[0] if refs.dates else None
        effective_date = self._extract_effective_date(chunk_text)
        
        # Extract page number if available
        page_num = refs.page_references[0] if refs.page_references else None
        
        return Citation(
            document_id=document.get('id', ''),
            document_title=document.get('title') or document.get('filename') or document.get('name') or 'Unknown Document',
            section_number=primary_section,
            article_number=primary_article,
            paragraph_number=primary_paragraph,
            subsection=None,  # Could be enhanced further
            clause=primary_clause,
            page_number=page_num,
            line_numbers=None,  # Could be calculated from chunk position
            date_mentioned=primary_date,
            effective_date=effective_date,
            chunk_text=chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text,
            similarity_score=chunk.get('similarity_score', 0.0),
            context_before=context_before,
            context_after=context_after
        )
    
    def _extract_context(self, text: str, context_sentences: int = 1) -> Tuple[str, str]:
        """Extract context sentences before and after the main content."""
        
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 3:
            return "", ""
        
        mid_point = len(sentences) // 2
        
        before = '. '.join(sentences[max(0, mid_point-context_sentences):mid_point]).strip()
        after = '. '.join(sentences[mid_point+1:mid_point+1+context_sentences]).strip()
        
        return before, after
    
    def _extract_effective_date(self, text: str) -> Optional[str]:
        """Extract effective/valid date from legal text."""
        
        effective_patterns = [
            r'effective\s+(?:as\s+of\s+)?(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'valid\s+from\s+(\d{1,2}[./]\d{1,2}[./]\d{2,4})'
        ]
        
        for pattern in effective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


class CitationFormatter:
    """Formats citations for display and reference."""
    
    def __init__(self):
        self.citation_styles = {
            'legal': self._format_legal_style,
            'academic': self._format_academic_style,
            'brief': self._format_brief_style,
            'detailed': self._format_detailed_style
        }
    
    def format_citation(self, citation: Citation, style: str = 'legal') -> str:
        """Format citation according to specified style."""
        
        formatter = self.citation_styles.get(style, self._format_legal_style)
        return formatter(citation)
    
    def _format_legal_style(self, citation: Citation) -> str:
        """Format in legal citation style."""
        
        parts = [citation.document_title]
        
        # Add structural references
        if citation.article_number:
            parts.append(f"Art. {citation.article_number}")
        if citation.section_number:
            parts.append(f"§ {citation.section_number}")
        if citation.paragraph_number:
            parts.append(f"¶ {citation.paragraph_number}")
        if citation.clause:
            parts.append(f"cl. {citation.clause}")
        
        # Add page reference
        if citation.page_number:
            parts.append(f"p. {citation.page_number}")
        
        # Add date if available
        if citation.effective_date:
            parts.append(f"(eff. {citation.effective_date})")
        elif citation.date_mentioned:
            parts.append(f"({citation.date_mentioned})")
        
        # Add similarity score for debugging
        parts.append(f"[sim: {citation.similarity_score:.3f}]")
        
        return ', '.join(parts)
    
    def _format_academic_style(self, citation: Citation) -> str:
        """Format in academic citation style."""
        
        base = f'"{citation.document_title}"'
        
        references = []
        if citation.article_number:
            references.append(f"Article {citation.article_number}")
        if citation.section_number:
            references.append(f"Section {citation.section_number}")
        if citation.paragraph_number:
            references.append(f"Paragraph {citation.paragraph_number}")
        
        if references:
            base += f", {', '.join(references)}"
        
        if citation.date_mentioned:
            base += f" ({citation.date_mentioned})"
        
        return base
    
    def _format_brief_style(self, citation: Citation) -> str:
        """Format in brief style for quick reference."""
        
        ref_parts = []
        if citation.article_number:
            ref_parts.append(f"Art.{citation.article_number}")
        if citation.paragraph_number:
            ref_parts.append(f"¶{citation.paragraph_number}")
        
        if ref_parts:
            return f"{citation.document_title} ({', '.join(ref_parts)})"
        else:
            return citation.document_title
    
    def _format_detailed_style(self, citation: Citation) -> str:
        """Format with full details for comprehensive reference."""
        
        details = {
            "Document": citation.document_title,
            "Article": citation.article_number,
            "Section": citation.section_number,
            "Paragraph": citation.paragraph_number,
            "Clause": citation.clause,
            "Page": citation.page_number,
            "Date": citation.date_mentioned,
            "Effective": citation.effective_date,
            "Similarity": f"{citation.similarity_score:.3f}"
        }
        
        # Filter out None values
        filtered_details = {k: v for k, v in details.items() if v is not None}
        
        return " | ".join([f"{k}: {v}" for k, v in filtered_details.items()])


def create_enhanced_citations(chunks: List[Dict[str, Any]]) -> List[Citation]:
    """Create enhanced citations from document chunks."""
    
    extractor = EnhancedCitationExtractor()
    citations = []
    
    for chunk in chunks:
        try:
            citation = extractor.create_citation(chunk)
            citations.append(citation)
        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            # Create basic citation as fallback
            document = chunk.get('document', {})
            citations.append(Citation(
                document_id=document.get('id', ''),
                document_title=document.get('title') or document.get('filename') or document.get('name') or 'Unknown Document',
                chunk_text=chunk.get('chunk', {}).get('content', ''),
                similarity_score=chunk.get('similarity_score', 0.0)
            ))
    
    return citations


def format_citations_for_response(citations: List[Citation], style: str = 'legal') -> List[str]:
    """Format citations for inclusion in response."""
    
    formatter = CitationFormatter()
    return [formatter.format_citation(citation, style) for citation in citations]


def generate_bibliography(citations: List[Citation]) -> str:
    """Generate a bibliography section from citations."""
    
    formatter = CitationFormatter()
    
    # Group by document
    doc_citations = {}
    for citation in citations:
        if citation.document_title not in doc_citations:
            doc_citations[citation.document_title] = []
        doc_citations[citation.document_title].append(citation)
    
    bibliography = ["## References\n"]
    
    for doc_title, doc_citations_list in doc_citations.items():
        # Sort by article/section number
        sorted_citations = sorted(doc_citations_list, key=lambda c: (
            c.article_number or "0", 
            c.section_number or "0", 
            c.paragraph_number or "0"
        ))
        
        ref_parts = []
        for citation in sorted_citations:
            parts = []
            if citation.article_number:
                parts.append(f"Art. {citation.article_number}")
            if citation.paragraph_number:
                parts.append(f"¶ {citation.paragraph_number}")
            if parts:
                ref_parts.append(", ".join(parts))
        
        if ref_parts:
            bibliography.append(f"- **{doc_title}**: {'; '.join(ref_parts)}")
        else:
            bibliography.append(f"- **{doc_title}**")
    
    return "\n".join(bibliography) 