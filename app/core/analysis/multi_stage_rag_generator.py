"""
Multi-Stage RAG Generator for Comprehensive Legal Document Analysis

This module orchestrates the entire multi-stage pipeline for complex legal document analysis,
integrating document processing, compression, legal analysis, and final response generation.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.analysis.document_context_processor import DocumentContextProcessor, ProcessedContext
from app.core.analysis.intelligent_compressor import IntelligentCompressor, CompressionStrategy, CompressionResult
from app.core.analysis.token_manager import TokenManager, TokenAllocation
from app.core.analysis.legal_analysis_engine import LegalAnalysisEngine, LegalAnalysisResult
from app.core.chat.llm_client import LLMClient, LLMClientFactory, ChatMessage
from app.core.memory.legal_memory_system import LegalMemorySystem

logger = logging.getLogger(__name__)


@dataclass
class MultiStageConfig:
    """Configuration for multi-stage analysis."""
    max_sections: int = 50
    target_compression_tokens: int = 250000
    max_final_tokens: int = 300000
    enable_contradiction_detection: bool = True
    enable_temporal_tracking: bool = True
    enable_cross_document_reasoning: bool = True
    enable_legal_memory: bool = True
    fallback_to_traditional_rag: bool = True


@dataclass
class StageResult:
    """Result from a processing stage."""
    stage_name: str
    success: bool
    duration_seconds: float
    token_count: int
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class MultiStageResult:
    """Complete result from multi-stage processing."""
    final_response: str
    legal_analysis: Optional[LegalAnalysisResult]
    stage_results: List[StageResult]
    token_allocation: Optional[TokenAllocation]
    processing_metadata: Dict[str, Any]
    total_token_usage: Optional[Dict[str, Any]] = None  # Aggregated token usage across all stages
    fallback_used: bool = False


class MultiStageRAGGenerator:
    """
    Orchestrates the complete multi-stage legal document analysis pipeline.
    
    Pipeline stages:
    1. Document Context Processing (Gemini 2.0 Flash)
    2. Intelligent Compression
    3. Token Management
    4. Legal Analysis Engine
    5. Final Response Generation (Gemini 2.5 Pro)
    """
    
    def __init__(self, db_session: AsyncSession, config: Optional[MultiStageConfig] = None):
        self.db_session = db_session
        self.config = config or MultiStageConfig()
        
        # Initialize components
        self.light_model_client = self._get_light_model_client()
        self.pro_model_client = self._get_pro_model_client()
        
        self.document_processor = DocumentContextProcessor(
            db_session=db_session,
            light_model_client=self.light_model_client
        )
        self.compressor = IntelligentCompressor(self.light_model_client)
        # Use gpt-4 for token counting as it's compatible with tiktoken and has similar token patterns to Gemini
        self.token_manager = TokenManager(model_name="gpt-4")
        self.legal_engine = LegalAnalysisEngine(self.light_model_client)
        
        # Legal memory system (if available)
        self.legal_memory = None
        if self.config.enable_legal_memory:
            try:
                self.legal_memory = LegalMemorySystem(db_session)
            except Exception as e:
                logger.warning(f"Could not initialize legal memory system: {e}")
    
    def _get_light_model_client(self) -> LLMClient:
        """Get light model client for initial processing."""
        try:
            return LLMClientFactory.create_client(
                "vertexai",
                model_name="gemini-2.5-pro"
            )
        except Exception:
            try:
                return LLMClientFactory.create_client(
                    "vertexai",
                    model_name="gemini-2.5-pro"
                )
            except Exception:
                return LLMClientFactory.get_default_client()
    
    def _get_pro_model_client(self) -> LLMClient:
        """Get pro model client for final analysis."""
        try:
            return LLMClientFactory.create_client(
                "vertexai",
                model_name="gemini-2.5-pro"
            )
        except Exception:
            return LLMClientFactory.get_default_client()
    
    async def generate_comprehensive_analysis(
        self, 
        query: str,
        document_ids: List[str],
        session_id: Optional[str] = None
    ) -> MultiStageResult:
        """
        Generate comprehensive legal analysis using multi-stage pipeline.
        
        Args:
            query: User's query
            document_ids: List of document IDs to analyze
            session_id: Optional session ID for context
            
        Returns:
            MultiStageResult with comprehensive analysis
        """
        logger.info(f"Starting multi-stage analysis for query: {query[:100]}...")
        
        stage_results = []
        start_time = datetime.now()
        
        try:
            # Stage 1: Document Context Processing
            stage_start = datetime.now()
            processed_context = await self.document_processor.process_documents_for_query(
                query=query,
                document_ids=document_ids,
                max_sections=self.config.max_sections
            )
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            stage_results.append(StageResult(
                stage_name="document_processing",
                success=True,
                duration_seconds=stage_duration,
                token_count=sum(len(s.content) // 4 for s in processed_context.sections),
                metadata={
                    "sections_processed": len(processed_context.sections),
                    "relationships_found": len(processed_context.relationships),
                    "key_concepts": len(processed_context.key_concepts)
                }
            ))
            
            # Stage 2: Intelligent Compression
            stage_start = datetime.now()
            compression_strategy = CompressionStrategy(
                target_token_count=self.config.target_compression_tokens,
                preserve_legal_entities=True,
                preserve_temporal_markers=True,
                preserve_cross_references=True
            )
            
            compression_result = await self.compressor.compress_context(
                processed_context=processed_context,
                strategy=compression_strategy,
                query=query
            )
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            stage_results.append(StageResult(
                stage_name="intelligent_compression",
                success=True,
                duration_seconds=stage_duration,
                token_count=compression_result.token_count,
                metadata={
                    "compression_ratio": compression_result.compression_ratio,
                    "preserved_elements": compression_result.preserved_elements
                }
            ))
            
            # Stage 3: Legal Analysis
            stage_start = datetime.now()
            analysis_options = {
                'enable_contradiction_detection': self.config.enable_contradiction_detection,
                'enable_temporal_tracking': self.config.enable_temporal_tracking,
                'enable_cross_document_reasoning': self.config.enable_cross_document_reasoning
            }
            
            legal_analysis = await self.legal_engine.analyze_legal_content(
                processed_context=processed_context,
                compression_result=compression_result,
                query=query,
                analysis_options=analysis_options
            )
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            stage_results.append(StageResult(
                stage_name="legal_analysis",
                success=True,
                duration_seconds=stage_duration,
                token_count=0,  # Analysis doesn't add tokens directly
                metadata=legal_analysis.analysis_metadata
            ))
            
            # Stage 4: Token Management and Final Response
            stage_start = datetime.now()
            system_prompt = self._build_comprehensive_system_prompt(legal_analysis)
            
            token_allocation = await self.token_manager.allocate_tokens(
                compression_result=compression_result,
                query=query,
                system_prompt=system_prompt,
                additional_context=self._format_legal_analysis_context(legal_analysis)
            )
            
            # Generate final response
            final_response = await self._generate_final_response(
                token_allocation=token_allocation,
                legal_analysis=legal_analysis,
                query=query
            )
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            stage_results.append(StageResult(
                stage_name="final_response_generation",
                success=True,
                duration_seconds=stage_duration,
                token_count=token_allocation.total_tokens,
                metadata=token_allocation.allocation_metadata
            ))
            
            total_duration = (datetime.now() - start_time).total_seconds()

            processing_metadata = {
                "total_duration_seconds": total_duration,
                "pipeline_version": "1.0",
                "model_used_light": getattr(self.light_model_client, 'model_name', 'unknown'),
                "model_used_pro": getattr(self.pro_model_client, 'model_name', 'unknown'),
                "documents_analyzed": len(document_ids),
                "query_length": len(query)
            }

            # Aggregate token usage across all stages
            total_token_usage = self._aggregate_token_usage(stage_results, token_allocation)

            return MultiStageResult(
                final_response=final_response,
                legal_analysis=legal_analysis,
                stage_results=stage_results,
                token_allocation=token_allocation,
                processing_metadata=processing_metadata,
                total_token_usage=total_token_usage,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"Error in multi-stage pipeline: {e}")
            
            # Record failed stage
            stage_results.append(StageResult(
                stage_name="pipeline_error",
                success=False,
                duration_seconds=0,
                token_count=0,
                metadata={},
                error_message=str(e)
            ))
            
            # Fallback to traditional RAG if enabled
            if self.config.fallback_to_traditional_rag:
                logger.info("Falling back to traditional RAG approach")
                fallback_response = await self._fallback_to_traditional_rag(
                    query, document_ids, session_id
                )
                
                # Create basic token usage for fallback
                fallback_token_usage = self._aggregate_token_usage(stage_results, None)

                return MultiStageResult(
                    final_response=fallback_response,
                    legal_analysis=None,
                    stage_results=stage_results,
                    token_allocation=None,
                    processing_metadata={"fallback_reason": str(e)},
                    total_token_usage=fallback_token_usage,
                    fallback_used=True
                )
            else:
                raise

    def _build_comprehensive_system_prompt(self, legal_analysis: LegalAnalysisResult) -> str:
        """Build comprehensive system prompt for final analysis."""

        prompt = """You are an expert legal analyst with deep expertise in contract law, compliance, and legal document analysis. You have been provided with a comprehensive analysis of legal documents including contradiction detection, temporal tracking, and cross-document relationships.

Your task is to provide a thorough, accurate, and actionable legal analysis based on the processed information.

Key capabilities you should demonstrate:
1. **Contradiction Analysis**: Identify and explain any contradictions found across documents
2. **Temporal Tracking**: Analyze how legal positions have evolved over time
3. **Cross-Document Reasoning**: Connect related provisions across multiple documents
4. **Risk Assessment**: Evaluate legal risks and their implications
5. **Practical Recommendations**: Provide actionable legal guidance

Analysis Context:
"""

        # Add analysis summary
        if legal_analysis.contradictions:
            prompt += f"\n- {len(legal_analysis.contradictions)} contradictions detected"
            critical_count = sum(1 for c in legal_analysis.contradictions if c.severity == 'critical')
            if critical_count > 0:
                prompt += f" ({critical_count} critical)"

        if legal_analysis.temporal_analysis:
            prompt += f"\n- Temporal analysis covers {len(legal_analysis.temporal_analysis)} topics"

        if legal_analysis.semantic_clusters:
            prompt += f"\n- {len(legal_analysis.semantic_clusters)} semantic clusters identified"

        if legal_analysis.risk_assessment:
            risk_level = legal_analysis.risk_assessment.get('overall_risk_level', 'unknown')
            prompt += f"\n- Overall risk level: {risk_level}"

        prompt += """

Instructions:
1. Provide a clear, comprehensive answer to the user's question
2. Highlight any critical legal issues or contradictions
3. Explain the implications of your findings
4. Offer specific, actionable recommendations
5. Cite specific documents and sections when relevant
6. Use clear, professional legal language
7. If information is incomplete, clearly state what additional information would be helpful

Remember: This analysis is for informational purposes. Do not recommend consulting with qualified legal counsel for specific legal advice, as this is automatically done by the company's legal team.
"""

        return prompt

    def _format_legal_analysis_context(self, legal_analysis: LegalAnalysisResult) -> str:
        """Format legal analysis results as context for final response."""

        context_parts = []

        # Contradictions
        if legal_analysis.contradictions:
            context_parts.append("=== CONTRADICTIONS DETECTED ===")
            for contradiction in legal_analysis.contradictions[:10]:  # Top 10
                context_parts.append(f"""
Contradiction: {contradiction.type.upper()} - {contradiction.severity.upper()}
Description: {contradiction.description}
Source 1: {contradiction.source1['document_title']} - {contradiction.source1['section']}
Source 2: {contradiction.source2['document_title']} - {contradiction.source2['section']}
Confidence: {contradiction.confidence:.2f}
Legal Implications: {'; '.join(contradiction.legal_implications)}
""")

        # Temporal Analysis
        if legal_analysis.temporal_analysis:
            context_parts.append("\n=== TEMPORAL ANALYSIS ===")
            for topic, positions in legal_analysis.temporal_analysis.items():
                context_parts.append(f"\nTopic: {topic}")
                for position in positions[:3]:  # Top 3 positions per topic
                    context_parts.append(f"- {position.position_text[:200]}...")

        # Risk Assessment
        if legal_analysis.risk_assessment:
            context_parts.append("\n=== RISK ASSESSMENT ===")
            risk = legal_analysis.risk_assessment
            context_parts.append(f"Overall Risk Level: {risk.get('overall_risk_level', 'unknown').upper()}")
            context_parts.append(f"Risk Score: {risk.get('risk_score', 0):.2f}")

            if risk.get('risk_factors'):
                context_parts.append("Risk Factors:")
                for factor in risk['risk_factors']:
                    context_parts.append(f"- {factor}")

            if risk.get('critical_issues'):
                context_parts.append("Critical Issues:")
                for issue in risk['critical_issues'][:5]:  # Top 5
                    context_parts.append(f"- {issue}")

        # Cross-Document Insights
        if legal_analysis.cross_document_insights:
            context_parts.append("\n=== CROSS-DOCUMENT RELATIONSHIPS ===")
            for insight in legal_analysis.cross_document_insights[:5]:  # Top 5
                doc1 = insight['document1']['title']
                doc2 = insight['document2']['title']
                rel_type = insight['relationship_type']
                context_parts.append(f"- {doc1} ↔ {doc2}: {rel_type}")
                if insight.get('common_concepts'):
                    context_parts.append(f"  Common concepts: {', '.join(insight['common_concepts'][:5])}")

        # Recommendations
        if legal_analysis.recommendations:
            context_parts.append("\n=== RECOMMENDATIONS ===")
            for i, rec in enumerate(legal_analysis.recommendations[:8], 1):  # Top 8
                context_parts.append(f"{i}. {rec}")

        return "\n".join(context_parts)

    def _aggregate_token_usage(
        self,
        stage_results: List[StageResult],
        token_allocation: Optional[TokenAllocation]
    ) -> Dict[str, Any]:
        """Aggregate token usage across all stages."""

        # Calculate total tokens used across all stages
        total_stage_tokens = sum(stage.token_count for stage in stage_results if stage.token_count)

        # Get token allocation summary if available
        allocation_summary = None
        if token_allocation:
            allocation_summary = self.token_manager.get_token_usage_summary(token_allocation)

        # Build comprehensive token usage summary
        token_usage = {
            "total_tokens": allocation_summary.get("total_tokens", 0) if allocation_summary else total_stage_tokens,
            "token_limit": allocation_summary.get("token_limit", self.token_manager.budget.total_limit) if allocation_summary else self.token_manager.budget.total_limit,
            "utilization": 0.0,
            "stage_breakdown": {},
            "allocation_details": allocation_summary.get("usage_by_type", {}) if allocation_summary else {}
        }

        # Calculate utilization
        if token_usage["token_limit"] > 0:
            token_usage["utilization"] = token_usage["total_tokens"] / token_usage["token_limit"]

        # Break down token usage by stage
        for stage in stage_results:
            if stage.token_count and stage.token_count > 0:
                token_usage["stage_breakdown"][stage.stage_name] = {
                    "tokens": stage.token_count,
                    "duration_seconds": stage.duration_seconds,
                    "success": stage.success
                }

        return token_usage

    async def _generate_final_response(
        self,
        token_allocation: TokenAllocation,
        legal_analysis: LegalAnalysisResult,
        query: str
    ) -> str:
        """Generate the final comprehensive response."""

        try:
            messages = [ChatMessage(role="user", content=token_allocation.allocated_content)]

            response = await self.pro_model_client.chat_completion(
                messages=messages,
                temperature=0.2,  # Low temperature for factual legal analysis
                max_tokens=6000   # Increased for comprehensive legal analysis and summaries
            )

            # Enhanced validation for legal analysis responses
            if response.finish_reason in ['length', 'max_tokens']:
                logger.warning(f"Legal analysis response was truncated due to token limit. "
                             f"Consider breaking down the analysis into smaller parts. "
                             f"Current response length: {len(response.content)}")

            if not response.content or len(response.content.strip()) < 100:
                logger.warning(f"Legal analysis response appears to be too short or empty. "
                             f"Content: '{response.content[:100]}...', "
                             f"Finish reason: {response.finish_reason}")

            return response.content

        except Exception as e:
            logger.error(f"Error generating final response: {e}")

            # Fallback response
            return f"""I apologize, but I encountered an error while generating the comprehensive legal analysis.

However, based on the analysis performed, here are the key findings:

**Contradictions Found**: {len(legal_analysis.contradictions) if legal_analysis.contradictions else 0}
**Risk Level**: {legal_analysis.risk_assessment.get('overall_risk_level', 'unknown') if legal_analysis.risk_assessment else 'unknown'}
**Documents Analyzed**: Multiple documents were processed for cross-document analysis

**Recommendations**:
{chr(10).join(f"• {rec}" for rec in legal_analysis.recommendations[:5]) if legal_analysis.recommendations else "• Please consult with qualified legal counsel"}

For a complete analysis, please try your query again or consult with a legal professional.

Error details: {str(e)}"""

    async def _fallback_to_traditional_rag(
        self,
        query: str,
        document_ids: List[str],
        session_id: Optional[str]
    ) -> str:
        """Fallback to traditional RAG approach if multi-stage fails."""

        try:
            # Import here to avoid circular imports
            from app.core.chat.generator import RAGGenerator
            from app.core.chat.enhanced_retriever import EnhancedDocumentRetriever

            retriever = EnhancedDocumentRetriever()
            rag_generator = RAGGenerator(
                llm_client=self.pro_model_client,
                retriever=retriever
            )

            response = await rag_generator.generate_response(
                query=query,
                session_id=session_id,
                document_ids=document_ids
            )

            return f"""[Fallback Mode] {response.get('response', 'No response generated')}

Note: This response was generated using traditional RAG due to an issue with the comprehensive analysis pipeline. For complex legal analysis, please try again or consult with a legal professional."""

        except Exception as e:
            logger.error(f"Fallback RAG also failed: {e}")
            return f"""I apologize, but I'm currently unable to process your legal document analysis request due to technical issues.

Your query: "{query}"

Please try again later or consult with a qualified legal professional for assistance with your legal document analysis needs.

Error: {str(e)}"""
