from typing import Dict, Any, List
import logging
from pathlib import Path

from app.core.ingest.extractors import DocumentExtractorFactory
from app.core.ingest.chunkers import DocumentChunkerFactory, TextChunk
from app.utils.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processor that coordinates text extraction and chunking."""
    
    def __init__(self):
        self.extractor_factory = DocumentExtractorFactory()
        self.chunker_factory = DocumentChunkerFactory()
    
    def process_document(self, file_path: str, chunking_strategy: str = "hybrid") -> Dict[str, Any]:
        """Process a document: extract text and create chunks."""
        
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Extract text from document
            extraction_result = self.extractor_factory.extract_text(file_path)
            
            if not extraction_result.get('text'):
                raise ValueError("No text extracted from document")
            
            # Step 2: Clean and preprocess text
            cleaned_text = self._clean_text(extraction_result['text'])
            
            # Step 3: Create chunks
            chunks = self.chunker_factory.chunk_document(cleaned_text, chunking_strategy)
            
            # Step 4: Post-process chunks
            processed_chunks = self._post_process_chunks(chunks)
            
            # Step 5: Generate document summary
            document_summary = self._generate_document_summary(
                cleaned_text, 
                processed_chunks, 
                extraction_result.get('metadata', {})
            )
            
            logger.info(f"Successfully processed document: {len(processed_chunks)} chunks created")
            
            return {
                'success': True,
                'original_text': extraction_result['text'],
                'cleaned_text': cleaned_text,
                'chunks': processed_chunks,
                'extraction_metadata': extraction_result.get('metadata', {}),
                'document_summary': document_summary,
                'processing_stats': {
                    'total_chunks': len(processed_chunks),
                    'original_char_count': extraction_result.get('char_count', 0),
                    'cleaned_char_count': len(cleaned_text),
                    'original_word_count': extraction_result.get('word_count', 0),
                    'chunking_strategy': chunking_strategy
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        
        # Remove excessive whitespace
        import re
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common OCR issues
        text = text.replace('`', "'")  # Fix backticks
        text = text.replace('"', '"').replace('"', '"')  # Normalize quotes
        text = text.replace(''', "'").replace(''', "'")  # Normalize apostrophes
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """Post-process chunks and convert to serializable format."""
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Additional cleaning for chunk text
            cleaned_chunk_text = self._clean_text(chunk.text)
            
            # Skip very short chunks
            if len(cleaned_chunk_text.split()) < 10:
                continue
            
            # Calculate additional metrics
            word_count = len(cleaned_chunk_text.split())
            sentence_count = len([s for s in cleaned_chunk_text.split('.') if s.strip()])
            
            processed_chunk = {
                'chunk_id': f"chunk_{i:04d}",
                'text': cleaned_chunk_text,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'chunk_type': chunk.chunk_type,
                'word_count': word_count,
                'char_count': len(cleaned_chunk_text),
                'sentence_count': sentence_count,
                'metadata': {
                    **chunk.metadata,
                    'chunk_index': i,
                    'density_score': word_count / len(cleaned_chunk_text) if cleaned_chunk_text else 0
                }
            }
            
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def _generate_document_summary(self, text: str, chunks: List[Dict], metadata: Dict) -> Dict[str, Any]:
        """Generate comprehensive legal document summary with semantic analysis."""

        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        # Chunk type distribution
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        # Enhanced legal analysis
        legal_analysis = self._analyze_legal_content(text, chunks)

        # Generate semantic summary
        semantic_summary = self._generate_semantic_summary(text, chunks)

        # Legal clause detection
        legal_indicators = self._detect_legal_indicators(text)

        # Key entities extraction
        key_entities = self._extract_key_entities(text)

        # Risk assessment
        risk_assessment = self._assess_legal_risks(text, chunks)
        
        # Legal clause detection
        legal_indicators = self._detect_legal_indicators(text)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'paragraph_count': paragraph_count,
            'chunk_count': len(chunks),
            'chunk_type_distribution': chunk_types,
            'legal_analysis': legal_analysis,
            'semantic_summary': semantic_summary,
            'legal_indicators': legal_indicators,
            'key_entities': key_entities,
            'risk_assessment': risk_assessment,
            'estimated_reading_time_minutes': word_count / 250,  # Avg 250 words per minute
            'complexity_score': self._calculate_complexity_score(text),
            'metadata': metadata
        }
    
    def _detect_legal_indicators(self, text: str) -> Dict[str, Any]:
        """Detect legal document indicators."""
        
        text_lower = text.lower()
        
        legal_keywords = {
            'contract_terms': ['whereas', 'therefore', 'party', 'agreement', 'contract'],
            'legal_entities': ['corporation', 'llc', 'inc', 'ltd', 'company'],
            'legal_actions': ['hereby', 'covenant', 'warrant', 'represent', 'indemnify'],
            'time_terms': ['effective date', 'term', 'expiration', 'renewal'],
            'financial_terms': ['payment', 'compensation', 'fee', 'penalty', 'damages'],
            'governance': ['governing law', 'jurisdiction', 'dispute resolution', 'arbitration']
        }
        
        detected_indicators = {}
        
        for category, keywords in legal_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            detected_indicators[category] = {
                'count': count,
                'keywords_found': [kw for kw in keywords if kw in text_lower]
            }
        
        # Overall legal document confidence score
        total_indicators = sum(ind['count'] for ind in detected_indicators.values())
        confidence_score = min(total_indicators / 20, 1.0)  # Cap at 1.0
        
        return {
            'categories': detected_indicators,
            'total_indicators': total_indicators,
            'legal_confidence_score': confidence_score,
            'likely_legal_document': confidence_score > 0.3
        }
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate document complexity score based on various factors."""
        
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Average characters per word
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Unique word ratio
        unique_words = set(word.lower() for word in words)
        unique_ratio = len(unique_words) / len(words)
        
        # Combine factors into complexity score (0-1)
        complexity = (
            min(avg_words_per_sentence / 25, 1.0) * 0.4 +  # Sentence length factor
            min(avg_chars_per_word / 8, 1.0) * 0.3 +       # Word length factor
            unique_ratio * 0.3                              # Vocabulary diversity factor
        )
        
        return round(complexity, 3)

    def _analyze_legal_content(self, text: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze legal content structure and extract key legal concepts."""
        import re

        # Detect document type
        doc_type = self._detect_document_type(text)

        # Extract legal sections
        sections = self._extract_legal_sections(text)

        # Identify key legal clauses
        clauses = self._identify_key_clauses(text)

        # Extract obligations and rights
        obligations = self._extract_obligations(text)

        return {
            'document_type': doc_type,
            'sections': sections,
            'key_clauses': clauses,
            'obligations': obligations,
            'total_sections': len(sections),
            'total_clauses': len(clauses)
        }

    def _generate_semantic_summary(self, text: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate semantic summary of document content."""

        # Extract key themes from chunks
        themes = self._extract_themes_from_chunks(chunks)

        # Generate executive summary points
        summary_points = self._generate_summary_points(text)

        # Identify critical information
        critical_info = self._identify_critical_information(text)

        return {
            'themes': themes,
            'summary_points': summary_points,
            'critical_information': critical_info,
            'summary_length': len(summary_points)
        }

    def _extract_key_entities(self, text: str) -> Dict[str, Any]:
        """Extract key legal entities from document."""
        import re

        # Extract dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)

        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*dollars?\b'
        monetary_amounts = re.findall(money_pattern, text, re.IGNORECASE)

        # Extract company names (simplified)
        company_pattern = r'\b[A-Z][a-zA-Z\s]+(?:Inc\.|LLC|Corp\.|Corporation|Company)\b'
        companies = re.findall(company_pattern, text)

        # Extract percentages
        percentage_pattern = r'\b\d+(?:\.\d+)?%|\b\d+\s*percent\b'
        percentages = re.findall(percentage_pattern, text, re.IGNORECASE)

        return {
            'dates': list(set(dates)),
            'monetary_amounts': list(set(monetary_amounts)),
            'companies': list(set(companies)),
            'percentages': list(set(percentages)),
            'total_entities': len(set(dates + monetary_amounts + companies + percentages))
        }

    def _assess_legal_risks(self, text: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Assess potential legal risks in the document."""

        risk_indicators = {
            'termination_risks': ['terminate', 'breach', 'default', 'violation'],
            'financial_risks': ['penalty', 'damages', 'liability', 'indemnify'],
            'compliance_risks': ['comply', 'regulation', 'law', 'requirement'],
            'confidentiality_risks': ['confidential', 'proprietary', 'trade secret', 'non-disclosure']
        }

        text_lower = text.lower()
        risk_assessment = {}

        for risk_type, indicators in risk_indicators.items():
            found_indicators = [ind for ind in indicators if ind in text_lower]
            risk_level = len(found_indicators) / len(indicators)

            risk_assessment[risk_type] = {
                'risk_level': risk_level,
                'found_indicators': found_indicators,
                'severity': 'high' if risk_level > 0.6 else 'medium' if risk_level > 0.3 else 'low'
            }

        # Overall risk score
        overall_risk = sum(r['risk_level'] for r in risk_assessment.values()) / len(risk_assessment)

        return {
            'risk_categories': risk_assessment,
            'overall_risk_score': round(overall_risk, 3),
            'overall_risk_level': 'high' if overall_risk > 0.6 else 'medium' if overall_risk > 0.3 else 'low'
        }

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document."""
        text_lower = text.lower()

        document_types = {
            'employment_agreement': ['employment', 'employee', 'employer', 'position', 'duties'],
            'service_agreement': ['service', 'provider', 'client', 'deliverables'],
            'lease_agreement': ['lease', 'tenant', 'landlord', 'premises', 'rent'],
            'purchase_agreement': ['purchase', 'buyer', 'seller', 'goods', 'price'],
            'nda': ['non-disclosure', 'confidential', 'proprietary', 'trade secret'],
            'license_agreement': ['license', 'licensor', 'licensee', 'intellectual property']
        }

        scores = {}
        for doc_type, keywords in document_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score

        # Return the document type with highest score, or 'unknown' if no clear match
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'unknown'

    def _extract_legal_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal sections from document."""
        import re

        # Pattern to match section headers
        section_pattern = r'(?:SECTION|Section|Article)\s+(\d+(?:\.\d+)*)\s*[.:]?\s*([^\n]+)'
        sections = []

        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        for match in matches:
            section_num = match.group(1)
            section_title = match.group(2).strip()
            start_pos = match.start()

            sections.append({
                'number': section_num,
                'title': section_title,
                'start_position': start_pos,
                'type': 'section'
            })

        return sections

    def _identify_key_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Identify key legal clauses in the document."""

        clause_patterns = {
            'termination': r'(?:termination|terminate|end|expire).*?(?:\.|;|\n)',
            'payment': r'(?:payment|pay|compensation|salary|fee).*?(?:\.|;|\n)',
            'confidentiality': r'(?:confidential|proprietary|trade secret|non-disclosure).*?(?:\.|;|\n)',
            'liability': r'(?:liability|liable|responsible|damages).*?(?:\.|;|\n)',
            'governing_law': r'(?:governing law|jurisdiction|applicable law).*?(?:\.|;|\n)'
        }

        clauses = []
        import re

        for clause_type, pattern in clause_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clauses.append({
                    'type': clause_type,
                    'text': match.group(0).strip(),
                    'start_position': match.start(),
                    'end_position': match.end()
                })

        return clauses

    def _extract_obligations(self, text: str) -> List[Dict[str, Any]]:
        """Extract obligations and responsibilities from the document."""
        import re

        obligation_patterns = [
            r'(?:shall|must|will|agrees? to|required to|obligated to)\s+([^.;]+)',
            r'(?:Employee|Company|Party)\s+(?:shall|must|will)\s+([^.;]+)',
            r'(?:is responsible for|has the duty to|undertakes to)\s+([^.;]+)'
        ]

        obligations = []

        for pattern in obligation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                obligation_text = match.group(0).strip()
                obligations.append({
                    'text': obligation_text,
                    'extracted_obligation': match.group(1).strip(),
                    'start_position': match.start(),
                    'type': 'obligation'
                })

        return obligations

    def _extract_themes_from_chunks(self, chunks: List[Dict]) -> List[str]:
        """Extract main themes from document chunks."""

        theme_keywords = {
            'employment': ['employee', 'employer', 'work', 'job', 'position'],
            'financial': ['payment', 'salary', 'compensation', 'money', 'fee'],
            'legal_compliance': ['law', 'regulation', 'compliance', 'legal'],
            'confidentiality': ['confidential', 'secret', 'proprietary', 'disclosure'],
            'termination': ['terminate', 'end', 'expire', 'breach'],
            'intellectual_property': ['patent', 'copyright', 'trademark', 'ip']
        }

        themes = []
        all_text = ' '.join([chunk.get('text', '') for chunk in chunks]).lower()

        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                themes.append(theme)

        return themes

    def _generate_summary_points(self, text: str) -> List[str]:
        """Generate key summary points from the document."""

        # This is a simplified implementation
        # In a production system, you'd use more sophisticated NLP

        summary_points = []

        # Look for key sentences that contain important information
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        important_keywords = [
            'agreement', 'contract', 'party', 'shall', 'payment', 'termination',
            'confidential', 'liability', 'governing law', 'effective date'
        ]

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                    summary_points.append(sentence.strip())

        # Return top 5 most relevant points
        return summary_points[:5]

    def _identify_critical_information(self, text: str) -> Dict[str, Any]:
        """Identify critical information that requires attention."""
        import re

        critical_info = {
            'deadlines': [],
            'monetary_obligations': [],
            'termination_conditions': [],
            'compliance_requirements': []
        }

        # Extract deadlines and dates
        date_patterns = [
            r'(?:by|before|within|no later than)\s+([^.;]+(?:day|week|month|year|date))',
            r'(?:deadline|due date|expiration).*?([^.;]+)'
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                critical_info['deadlines'].append(match.group(0).strip())

        # Extract monetary obligations
        money_obligations = re.finditer(r'(?:pay|payment|fee|penalty).*?\$[\d,]+(?:\.\d{2})?', text, re.IGNORECASE)
        for match in money_obligations:
            critical_info['monetary_obligations'].append(match.group(0).strip())

        return critical_info