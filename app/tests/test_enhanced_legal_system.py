"""
Comprehensive Testing Framework for Enhanced Legal Document Analysis System

This module provides extensive testing for the improved legal analysis capabilities,
including multi-document reasoning, summarization, memory system, and search enhancements.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os

# Import the enhanced components
from app.core.ingest.processors import DocumentProcessor
from app.core.analysis.multi_document_analyzer import MultiDocumentAnalyzer, ContradictionSeverity
from app.core.memory.legal_memory_system import LegalMemorySystem, MemoryQuery, QueryType
from app.core.search.enhanced_legal_search import EnhancedLegalSearchEngine, SearchQuery


class TestEnhancedSummarization:
    """Test the enhanced summarization capabilities."""
    
    @pytest.fixture
    def sample_legal_document(self):
        """Sample legal document for testing."""
        return """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into on January 15, 2024, 
        between TechCorp Industries, Inc. and John Smith.
        
        SECTION 1. POSITION AND DUTIES
        Employee shall serve as Senior Software Engineer.
        
        SECTION 2. COMPENSATION
        Company shall pay Employee an annual base salary of $120,000.
        Employee may be eligible for an annual performance bonus of up to 20%.
        
        SECTION 3. TERMINATION
        Either party may terminate this Agreement with thirty (30) days written notice.
        If Company terminates Employee without cause, Employee shall receive 
        severance pay equal to three (3) months of base salary.
        
        SECTION 4. CONFIDENTIALITY
        Employee agrees to maintain the confidentiality of proprietary information.
        """
    
    @pytest.fixture
    def document_processor(self):
        """Document processor instance."""
        return DocumentProcessor()
    
    def test_enhanced_document_summary(self, document_processor, sample_legal_document):
        """Test enhanced document summarization with legal analysis."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_legal_document)
            temp_path = f.name
        
        try:
            # Process document
            result = document_processor.process_document(temp_path)
            
            # Verify enhanced summarization
            assert result['success'] is True
            summary = result['document_summary']
            
            # Check basic statistics
            assert 'word_count' in summary
            assert 'char_count' in summary
            assert summary['word_count'] > 0
            
            # Check enhanced legal analysis
            assert 'legal_analysis' in summary
            legal_analysis = summary['legal_analysis']
            assert 'document_type' in legal_analysis
            assert 'key_clauses' in legal_analysis
            assert 'obligations' in legal_analysis
            
            # Check semantic summary
            assert 'semantic_summary' in summary
            semantic_summary = summary['semantic_summary']
            assert 'themes' in semantic_summary
            assert 'summary_points' in semantic_summary
            
            # Check legal indicators
            assert 'legal_indicators' in summary
            legal_indicators = summary['legal_indicators']
            assert 'legal_confidence_score' in legal_indicators
            assert legal_indicators['legal_confidence_score'] > 0.5  # Should detect as legal doc
            
            # Check key entities
            assert 'key_entities' in summary
            key_entities = summary['key_entities']
            assert 'monetary_amounts' in key_entities
            assert '$120,000' in str(key_entities['monetary_amounts'])
            
            # Check risk assessment
            assert 'risk_assessment' in summary
            risk_assessment = summary['risk_assessment']
            assert 'overall_risk_score' in risk_assessment
            
        finally:
            os.unlink(temp_path)
    
    def test_legal_clause_detection(self, document_processor, sample_legal_document):
        """Test detection of specific legal clauses."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_legal_document)
            temp_path = f.name
        
        try:
            result = document_processor.process_document(temp_path)
            legal_analysis = result['document_summary']['legal_analysis']
            
            # Check clause detection
            key_clauses = legal_analysis['key_clauses']
            clause_types = [clause['type'] for clause in key_clauses]
            
            assert 'payment' in clause_types
            assert 'termination' in clause_types
            assert 'confidentiality' in clause_types
            
        finally:
            os.unlink(temp_path)
    
    def test_risk_assessment_accuracy(self, document_processor, sample_legal_document):
        """Test accuracy of legal risk assessment."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_legal_document)
            temp_path = f.name
        
        try:
            result = document_processor.process_document(temp_path)
            risk_assessment = result['document_summary']['risk_assessment']
            
            # Check risk categories
            assert 'risk_categories' in risk_assessment
            risk_categories = risk_assessment['risk_categories']
            
            # Should detect termination risks (due to termination clause)
            assert 'termination_risks' in risk_categories
            
            # Should detect financial risks (due to payment terms)
            assert 'financial_risks' in risk_categories
            
            # Overall risk should be reasonable
            overall_risk = risk_assessment['overall_risk_score']
            assert 0.0 <= overall_risk <= 1.0
            
        finally:
            os.unlink(temp_path)


class TestMultiDocumentAnalysis:
    """Test multi-document reasoning and analysis capabilities."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for multi-document testing."""
        return [
            {
                'id': 'doc1',
                'content': """
                EMPLOYMENT AGREEMENT - Version 1
                Termination: Either party may terminate with 60 days notice.
                Salary: $100,000 annually.
                Confidentiality: Employee must maintain confidentiality for 2 years.
                """,
                'metadata': {
                    'type': 'employment_agreement',
                    'date': '2023-01-01',
                    'parties': ['TechCorp', 'John Smith']
                }
            },
            {
                'id': 'doc2',
                'content': """
                EMPLOYMENT AGREEMENT - Version 2
                Termination: Either party may terminate with 30 days notice.
                Salary: $120,000 annually.
                Confidentiality: Employee must maintain confidentiality for 3 years.
                """,
                'metadata': {
                    'type': 'employment_agreement',
                    'date': '2024-01-01',
                    'parties': ['TechCorp', 'John Smith']
                }
            },
            {
                'id': 'doc3',
                'content': """
                SERVICE AGREEMENT
                Payment: $5,000 monthly fee.
                Termination: 30 days notice required.
                Confidentiality: Standard NDA applies.
                """,
                'metadata': {
                    'type': 'service_agreement',
                    'date': '2024-02-01',
                    'parties': ['TechCorp', 'ServiceCorp']
                }
            }
        ]
    
    @pytest.fixture
    def multi_doc_analyzer(self):
        """Multi-document analyzer instance."""
        return MultiDocumentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_document_relationship_analysis(self, multi_doc_analyzer, sample_documents):
        """Test document relationship detection."""
        
        analysis_options = {
            'detect_contradictions': False,
            'analyze_relationships': True,
            'track_clause_evolution': False,
            'semantic_clustering': False
        }
        
        result = await multi_doc_analyzer.analyze_document_corpus(
            sample_documents, analysis_options
        )
        
        # Check relationships were detected
        assert 'document_relationships' in result
        relationships = result['document_relationships']
        
        # Should find relationship between doc1 and doc2 (same type, parties)
        relationship_pairs = [(r.doc1_id, r.doc2_id) for r in relationships]
        assert ('doc1', 'doc2') in relationship_pairs or ('doc2', 'doc1') in relationship_pairs
    
    @pytest.mark.asyncio
    async def test_contradiction_detection(self, multi_doc_analyzer, sample_documents):
        """Test contradiction detection between documents."""
        
        analysis_options = {
            'detect_contradictions': True,
            'analyze_relationships': False,
            'track_clause_evolution': False,
            'semantic_clustering': False
        }
        
        result = await multi_doc_analyzer.analyze_document_corpus(
            sample_documents, analysis_options
        )
        
        # Check contradictions were detected
        assert 'contradictions' in result
        contradictions = result['contradictions']
        
        # Should detect contradictions between doc1 and doc2
        # (different notice periods, salaries, confidentiality terms)
        assert len(contradictions) > 0
        
        # Check contradiction details
        for contradiction in contradictions:
            assert hasattr(contradiction, 'severity')
            assert hasattr(contradiction, 'contradiction_type')
            assert hasattr(contradiction, 'confidence_score')
    
    @pytest.mark.asyncio
    async def test_clause_evolution_tracking(self, multi_doc_analyzer, sample_documents):
        """Test clause evolution tracking over time."""
        
        analysis_options = {
            'detect_contradictions': False,
            'analyze_relationships': False,
            'track_clause_evolution': True,
            'semantic_clustering': False
        }
        
        result = await multi_doc_analyzer.analyze_document_corpus(
            sample_documents, analysis_options
        )
        
        # Check clause evolution tracking
        assert 'clause_evolution' in result
        evolution = result['clause_evolution']
        
        # Should track evolution in employment agreements
        assert 'employment_agreement' in evolution
    
    @pytest.mark.asyncio
    async def test_semantic_clustering(self, multi_doc_analyzer, sample_documents):
        """Test semantic clustering of documents."""
        
        analysis_options = {
            'detect_contradictions': False,
            'analyze_relationships': False,
            'track_clause_evolution': False,
            'semantic_clustering': True
        }
        
        result = await multi_doc_analyzer.analyze_document_corpus(
            sample_documents, analysis_options
        )
        
        # Check semantic clustering
        assert 'semantic_clusters' in result
        clusters = result['semantic_clusters']
        
        assert 'clusters' in clusters
        assert 'cluster_analysis' in clusters
        assert 'total_clusters' in clusters


class TestLegalMemorySystem:
    """Test the legal memory system capabilities."""
    
    @pytest.fixture
    def memory_system(self):
        """Legal memory system instance."""
        return LegalMemorySystem()
    
    @pytest.mark.asyncio
    async def test_precedent_search(self, memory_system):
        """Test precedent search functionality."""
        
        query = MemoryQuery(
            query_text="Have we agreed to similar termination terms before?",
            query_type=QueryType.PRECEDENT_SEARCH,
            context={'document_type': 'employment'},
            filters={'clause_type': 'termination'}
        )
        
        response = await memory_system.query_memory(query)
        
        # Check response structure
        assert response.query == query
        assert isinstance(response.results, list)
        assert isinstance(response.confidence_score, float)
        assert isinstance(response.reasoning, str)
        assert isinstance(response.suggestions, list)
        assert response.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_similar_terms_detection(self, memory_system):
        """Test similar terms detection."""
        
        query = MemoryQuery(
            query_text="Find similar payment terms across all contracts",
            query_type=QueryType.SIMILAR_TERMS,
            context={},
            filters={'clause_type': 'payment'}
        )
        
        response = await memory_system.query_memory(query)
        
        # Should return similar terms results
        assert len(response.results) >= 0
        assert response.confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_complex_question_answering(self, memory_system):
        """Test complex legal question answering."""
        
        question = "Have we agreed to similar confidentiality terms in previous employment contracts?"
        context = {'document_type': 'employment', 'focus': 'confidentiality'}
        
        answer = await memory_system.answer_complex_question(question, context)
        
        # Check answer structure
        assert 'answer' in answer
        assert 'confidence' in answer
        assert 'supporting_evidence' in answer
        assert 'recommendations' in answer
        
        # Answer should be meaningful
        assert len(answer['answer']) > 10
        assert 0.0 <= answer['confidence'] <= 1.0


class TestEnhancedSearch:
    """Test enhanced search capabilities."""
    
    @pytest.fixture
    def search_engine(self):
        """Enhanced search engine instance."""
        return EnhancedLegalSearchEngine()
    
    @pytest.mark.asyncio
    async def test_legal_concept_search(self, search_engine):
        """Test legal concept-based search."""
        
        query = SearchQuery(
            query_text="termination clauses with notice periods",
            legal_context="employment law",
            clause_types=["termination"],
            max_results=5
        )
        
        results = await search_engine.search(query)
        
        # Check results structure
        assert isinstance(results, list)
        assert len(results) <= query.max_results
        
        for result in results:
            assert hasattr(result, 'document_id')
            assert hasattr(result, 'content')
            assert hasattr(result, 'relevance_score')
            assert hasattr(result, 'legal_concepts')
    
    @pytest.mark.asyncio
    async def test_query_analysis(self, search_engine):
        """Test search query analysis."""
        
        query = SearchQuery(
            query_text="payment terms and salary information in employment contracts"
        )
        
        analysis = await search_engine._analyze_search_query(query)
        
        # Check analysis components
        assert 'detected_concepts' in analysis
        assert 'detected_clause_types' in analysis
        assert 'query_complexity' in analysis
        assert 'search_intent' in analysis
        
        # Should detect payment-related concepts
        detected_concepts = analysis['detected_concepts']
        assert any('payment' in concept for concept in detected_concepts)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_document_processing_performance(self):
        """Test document processing performance."""
        
        # Create a larger test document
        large_document = """
        COMPREHENSIVE LEGAL AGREEMENT
        """ + "\n".join([f"SECTION {i}. This is section {i} with legal content." for i in range(1, 51)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_document)
            temp_path = f.name
        
        try:
            processor = DocumentProcessor()
            start_time = datetime.now()
            
            result = processor.process_document(temp_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            assert result['success'] is True
            assert processing_time < 30.0  # Should process within 30 seconds
            
            # Check that all enhanced features were processed
            summary = result['document_summary']
            assert 'legal_analysis' in summary
            assert 'semantic_summary' in summary
            assert 'risk_assessment' in summary
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_multi_document_analysis_performance(self):
        """Test multi-document analysis performance."""
        
        # Create multiple test documents
        documents = []
        for i in range(5):
            documents.append({
                'id': f'perf_doc_{i}',
                'content': f"Legal document {i} with various clauses and terms.",
                'metadata': {
                    'type': 'test_document',
                    'date': f'2024-0{i+1}-01'
                }
            })
        
        analyzer = MultiDocumentAnalyzer()
        start_time = datetime.now()
        
        result = await analyzer.analyze_document_corpus(documents)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert processing_time < 60.0  # Should complete within 60 seconds
        assert 'analysis_summary' in result
        assert result['analysis_summary']['total_documents'] == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
