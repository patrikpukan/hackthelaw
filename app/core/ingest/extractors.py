from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
from pathlib import Path
import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import chardet
except ImportError:
    chardet = None

logger = logging.getLogger(__name__)


class DocumentExtractor(ABC):
    """Abstract base class for document text extraction."""
    
    @abstractmethod
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from document."""
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if extractor supports given file extension."""
        pass


class PDFExtractor(DocumentExtractor):
    """PDF text extraction using pdfplumber."""
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.pdf'
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file."""
        if pdfplumber is None:
            raise ImportError("pdfplumber is required for PDF extraction")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text_content = []
                metadata = {
                    'total_pages': len(pdf.pages),
                    'page_texts': []
                }
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        metadata['page_texts'].append({
                            'page': page_num + 1,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                
                full_text = '\n\n'.join(text_content)
                
                return {
                    'text': full_text,
                    'char_count': len(full_text),
                    'word_count': len(full_text.split()),
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            raise


class DOCXExtractor(DocumentExtractor):
    """DOCX text extraction using python-docx."""
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.docx'
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX extraction")
        
        try:
            doc = DocxDocument(file_path)
            
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            full_text = '\n'.join(paragraphs)
            
            # Extract metadata
            metadata = {
                'paragraph_count': len(paragraphs),
                'core_properties': {}
            }
            
            # Try to extract document properties
            try:
                core_props = doc.core_properties
                metadata['core_properties'] = {
                    'title': core_props.title,
                    'author': core_props.author,
                    'created': str(core_props.created) if core_props.created else None,
                    'modified': str(core_props.modified) if core_props.modified else None
                }
            except Exception as e:
                logger.warning(f"Could not extract core properties: {e}")
            
            return {
                'text': full_text,
                'char_count': len(full_text),
                'word_count': len(full_text.split()),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
            raise


class TXTExtractor(DocumentExtractor):
    """Plain text file extraction."""
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.txt', '.text']
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from plain text file."""
        try:
            # Try to detect encoding
            encoding = 'utf-8'
            if chardet:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    if result['encoding']:
                        encoding = result['encoding']
            
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            return {
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split()),
                'metadata': {
                    'encoding': encoding,
                    'line_count': len(text.splitlines())
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting TXT {file_path}: {e}")
            raise


class RTFExtractor(DocumentExtractor):
    """RTF text extraction."""
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.rtf'
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from RTF file."""
        # For now, treat as plain text
        # TODO: Implement proper RTF parsing with python-rtf or striprtf
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            
            # Simple RTF text extraction (very basic)
            # This is a placeholder - in production, use proper RTF library
            text = rtf_content
            
            return {
                'text': text,
                'char_count': len(text),
                'word_count': len(text.split()),
                'metadata': {
                    'format': 'rtf',
                    'note': 'Basic RTF extraction - upgrade to proper parser needed'
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting RTF {file_path}: {e}")
            raise


class DocumentExtractorFactory:
    """Factory for creating appropriate document extractors."""
    
    def __init__(self):
        self.extractors = [
            PDFExtractor(),
            DOCXExtractor(),
            TXTExtractor(),
            RTFExtractor()
        ]
    
    def get_extractor(self, file_path: str) -> Optional[DocumentExtractor]:
        """Get appropriate extractor for file."""
        file_extension = Path(file_path).suffix.lower()
        
        for extractor in self.extractors:
            if extractor.supports(file_extension):
                return extractor
        
        return None
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text using appropriate extractor."""
        extractor = self.get_extractor(file_path)
        
        if extractor is None:
            raise ValueError(f"No extractor available for file: {file_path}")
        
        return extractor.extract(file_path) 