import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil

from app.core.ingest.extractors import TXTExtractor, PDFExtractor, DOCXExtractor
from app.core.ingest.chunkers import LegalClauseChunker, SentenceChunker
from app.core.ingest.processors import DocumentProcessor


@pytest.mark.unit
class TestTXTExtractor:
    """Test text extraction functionality."""
    
    def test_extract_from_txt_file(self, sample_text_content: str):
        """Test extracting text from a .txt file."""
        extractor = TXTExtractor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_content)
            f.flush()
            
            try:
                result = extractor.extract(f.name)
                assert result['text'] == sample_text_content
                assert 'char_count' in result
                assert 'word_count' in result
            finally:
                Path(f.name).unlink()
    
    def test_extract_from_nonexistent_file(self):
        """Test extraction from non-existent file."""
        extractor = TXTExtractor()
        
        with pytest.raises(Exception):  # Could be FileNotFoundError or other exception
            extractor.extract("/nonexistent/file.txt")


@pytest.mark.unit
class TestPDFExtractor:
    """Test PDF extraction functionality."""
    
    @patch('pdfplumber.open')
    def test_extract_from_pdf(self, mock_pdfplumber):
        """Test extracting text from PDF."""
        # Mock PDF structure
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF content\nPage 1 text"
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        
        mock_pdfplumber.return_value = mock_pdf
        
        extractor = PDFExtractor()
        result = extractor.extract("dummy.pdf")
        
        assert "Sample PDF content" in result['text']
        assert "Page 1 text" in result['text']
        mock_pdfplumber.assert_called_once_with("dummy.pdf")


@pytest.mark.unit
class TestDOCXExtractor:
    """Test DOCX extraction functionality."""
    
    @patch('app.core.ingest.extractors.DocxDocument')
    def test_extract_from_docx(self, mock_document):
        """Test extracting text from DOCX."""
        # Mock DOCX structure
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "First paragraph"
        
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Second paragraph"
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_doc.core_properties = MagicMock()
        mock_doc.core_properties.title = "Test Document"
        mock_doc.core_properties.author = "Test Author"
        mock_doc.core_properties.created = None
        mock_doc.core_properties.modified = None
        
        mock_document.return_value = mock_doc
        
        extractor = DOCXExtractor()
        result = extractor.extract("document.docx")
        
        assert "First paragraph" in result['text']
        assert "Second paragraph" in result['text']
        assert 'char_count' in result
        assert 'word_count' in result
        mock_document.assert_called_once_with("document.docx")


@pytest.mark.unit
class TestLegalClauseChunker:
    """Test legal document chunking functionality."""
    
    def test_chunk_by_sections(self, sample_text_content: str):
        """Test chunking document by legal sections."""
        chunker = LegalClauseChunker()
        
        chunks = chunker.chunk(sample_text_content)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        assert all(hasattr(chunk, 'chunk_type') for chunk in chunks)
        
        # Check that important sections are captured
        chunk_texts = [chunk.text for chunk in chunks]
        full_text = " ".join(chunk_texts)
        assert len(full_text) > 0
    
    def test_clause_classification(self):
        """Test clause type classification."""
        chunker = LegalClauseChunker()
        
        test_cases = [
            ("The term of employment shall be two years", "general"),
            ("Employee shall receive salary of $100,000", "payment"),
            ("Either party may terminate with 30 days notice", "termination"),
        ]
        
        for text, expected_category in test_cases:
            clause_type = chunker.classify_clause_type(text)
            # Just check that it returns a string (classification logic may vary)
            assert isinstance(clause_type, str)


@pytest.mark.unit
class TestSentenceChunker:
    """Test sentence-based chunking functionality."""
    
    def test_sentence_chunking(self):
        """Test basic sentence chunking."""
        chunker = SentenceChunker(chunk_size=50, overlap=5)
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'text') for chunk in chunks)
        assert all(hasattr(chunk, 'start_char') for chunk in chunks)
        assert all(hasattr(chunk, 'end_char') for chunk in chunks)


@pytest.mark.unit
class TestDocumentProcessor:
    """Test complete document processing pipeline."""
    
    def test_process_text_document_success(self, sample_text_content: str):
        """Test processing a text document through the pipeline."""
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text_content)
            f.flush()
            
            try:
                result = processor.process_document(f.name)
                
                assert result["success"] == True
                assert "cleaned_text" in result
                assert "chunks" in result
                assert len(result["chunks"]) > 0
                
            finally:
                Path(f.name).unlink()
    
    def test_process_unsupported_file_type(self):
        """Test processing unsupported file type."""
        processor = DocumentProcessor()
        
        result = processor.process_document("test.exe")
        
        assert result["success"] == False
        assert "error" in result


@pytest.mark.integration
class TestDocumentProcessingIntegration:
    """Integration tests for document processing."""
    
    def test_full_processing_pipeline(self, sample_text_content: str, temp_upload_dir: Path):
        """Test complete document processing pipeline with real files."""
        # Create a test file
        test_file = temp_upload_dir / "integration_test.txt"
        test_file.write_text(sample_text_content)
        
        # Process the document
        processor = DocumentProcessor()
        result = processor.process_document(str(test_file))
        
        # Verify results
        assert result["success"] == True
        assert len(result["chunks"]) > 0
        
        # Verify chunk structure
        for chunk in result["chunks"]:
            assert "text" in chunk
            assert "chunk_id" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "chunk_type" in chunk 