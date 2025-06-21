from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.core.chat.llm_client import LLMClient, LLMClientFactory, ChatMessage, LLMResponse
from app.core.chat.retriever import DocumentRetriever
from app.core.chat.token_config import token_config_manager
from app.core.chat.response_validator import response_validator
from app.db.connection import async_session
from app.db.models import ChatSession, ChatMessage as DBChatMessage

logger = logging.getLogger(__name__)


class RAGGenerator:
    """RAG (Retrieval-Augmented Generation) system for legal document Q&A."""
    
    def __init__(self, llm_client: LLMClient = None, retriever: DocumentRetriever = None):
        self.llm_client = llm_client or LLMClientFactory.get_default_client()
        self.retriever = retriever or DocumentRetriever()
        
        # Legal RAG system prompt
        self.system_prompt = """You are a specialized legal AI assistant in a legal company with access to a corpus of legal documents. Your role is to provide accurate, helpful responses about legal clauses, contracts, and document analysis.

Key Instructions:
1. Base your answers on the provided document context
2. If information is not in the context, clearly state this limitation
3. For legal interpretations, do not recommend consulting with qualified legal counsel, as this is done by the company's legal team.
4. Cite specific documents when referencing information
5. Be precise about clause types, dates, and legal terms
6. Identify conflicts or inconsistencies when they exist

Context Format:
- Each document chunk includes the source document name and clause type
- Pay attention to document dates for chronological analysis
- Consider clause hierarchies and relationships

Always provide clear, professional responses that help legal professionals understand their documents. The profesiononals always double-check your answers, you do not have to remind them of this. If you are asked to generate a document from already existing documents, only change what the user specified, the rest of the document should be the same as the original document."""
    
    async def generate_response(
        self, 
        query: str, 
        session_id: str = None,
        conversation_history: List[Dict] = None,
        max_context_chunks: int = 5,
        document_ids: List[str] = None
    ) -> Dict[str, Any]:
        """Generate RAG response for a legal query."""
        
        try:
            # Step 1: Retrieve relevant document chunks
            logger.info(f"Retrieving relevant documents for query: {query[:100]}...")
            retrieval_results = await self.retriever.retrieve_relevant_chunks(
                query, 
                limit=max_context_chunks,
                document_ids=document_ids
            )
            
            # Step 2: Build context from retrieved documents
            context = self._build_context(retrieval_results)
            
            # Step 3: Construct conversation with system prompt and context
            messages = self._build_conversation(
                query, 
                context, 
                conversation_history or []
            )
            
            # Step 4: Generate LLM response
            logger.info("Generating LLM response...")

            # Use intelligent token allocation based on query type and context
            max_tokens = token_config_manager.get_max_tokens(query, len(context))
            logger.info(f"Using {max_tokens} max_tokens for query type")

            llm_response = await self.llm_client.chat_completion(
                messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=max_tokens
            )

            # Enhanced validation using response validator
            query_type = token_config_manager.analyze_query_type(query)
            expected_type = "summary" if query_type.value == "summarization" else "general"

            validation_result = response_validator.validate_response(llm_response, query, expected_type)
            response_validator.log_validation_result(validation_result, query)
            
            # Step 5: Post-process and format response
            formatted_response = self._format_response(
                llm_response, 
                retrieval_results, 
                query
            )
            
            # Step 6: Save to database if session provided
            if session_id:
                await self._save_conversation(session_id, query, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "content": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "error": True,
                "query": query
            }
    
    def _build_context(self, retrieval_results: List[Dict]) -> str:
        """Build context string from retrieved document chunks."""
        
        if not retrieval_results:
            return "No relevant documents found in the corpus."
        
        context_parts = ["RELEVANT DOCUMENT CONTEXT:\n"]
        
        for i, result in enumerate(retrieval_results, 1):
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            similarity = result.get('similarity_score', 0.0)
            
            context_part = f"""
Document {i}:
- Source: {document.get('filename', 'Unknown')}
- Clause Type: {chunk.get('chunk_type', 'general')}
- Relevance Score: {similarity:.3f}
- Content: {chunk.get('text', '')}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_conversation(
        self, 
        query: str, 
        context: str, 
        conversation_history: List[Dict]
    ) -> List[ChatMessage]:
        """Build conversation messages for LLM."""
        
        messages = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        
        # Add conversation history (last 5 exchanges)
        for exchange in conversation_history[-5:]:
            if exchange.get('user_message'):
                messages.append(ChatMessage(role="user", content=exchange['user_message']))
            if exchange.get('assistant_message'):
                messages.append(ChatMessage(role="assistant", content=exchange['assistant_message']))
        
        # Add current query with context
        user_prompt = f"""Based on the following legal document context, please answer this question:

{context}

QUESTION: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please state what additional information would be needed."""
        
        messages.append(ChatMessage(role="user", content=user_prompt))
        
        return messages
    
    def _format_response(
        self, 
        llm_response: LLMResponse, 
        retrieval_results: List[Dict],
        original_query: str
    ) -> Dict[str, Any]:
        """Format the final response with sources and metadata."""
        
        # Extract source information
        sources = []
        for result in retrieval_results:
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            
            source = {
                "document_name": document.get('filename', 'Unknown'),
                "document_id": document.get('id'),
                "chunk_type": chunk.get('chunk_type', 'general'),
                "similarity_score": result.get('similarity_score', 0.0),
                "chunk_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            }
            sources.append(source)
        
        return {
            "content": llm_response.content,
            "sources": sources,
            "query": original_query,
            "model_used": llm_response.model,
            "usage": llm_response.usage,
            "timestamp": datetime.utcnow().isoformat(),
            "error": False
        }
    
    async def _save_conversation(
        self, 
        session_id: str, 
        user_query: str, 
        response: Dict[str, Any]
    ):
        """Save conversation to database."""
        
        try:
            async with async_session() as session:
                # Save user message
                user_message = DBChatMessage(
                    session_id=session_id,
                    message_type="user",
                    content=user_query,
                    message_metadata=None
                )
                session.add(user_message)
                
                # Save assistant message
                assistant_message = DBChatMessage(
                    session_id=session_id,
                    message_type="assistant",
                    content=response["content"],
                    message_metadata=str({
                        "sources": response.get("sources", []),
                        "model_used": response.get("model_used"),
                        "usage": response.get("usage")
                    })
                )
                session.add(assistant_message)
                
                await session.commit()
                logger.info(f"Saved conversation to session {session_id}")
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


class LegalPromptTemplates:
    """Legal-specific prompt templates for different types of queries."""
    
    @staticmethod
    def contract_analysis_prompt(context: str, query: str) -> str:
        """Prompt template for contract analysis."""
        return f"""You are analyzing legal contracts. Based on the provided contract context, please answer the following question with focus on:

1. Specific clause identification and location
2. Legal implications and obligations
3. Potential risks or conflicts
4. Relevant dates and timelines

CONTEXT:
{context}

QUESTION: {query}

Please provide a detailed analysis with specific references to the contract terms."""
    
    @staticmethod
    def clause_comparison_prompt(context: str, query: str) -> str:
        """Prompt template for comparing clauses across documents."""
        return f"""You are comparing legal clauses across multiple documents. Please focus on:

1. Identifying similarities and differences
2. Highlighting conflicts or inconsistencies
3. Noting evolution of terms over time
4. Assessing legal consistency

CONTEXT:
{context}

QUESTION: {query}

Please provide a comparative analysis highlighting key differences and potential issues."""
    
    @staticmethod
    def compliance_check_prompt(context: str, query: str) -> str:
        """Prompt template for compliance checking."""
        return f"""You are performing a compliance check on legal documents. Please evaluate:

1. Adherence to legal requirements
2. Missing mandatory clauses
3. Potential compliance risks
4. Recommendations for improvement

CONTEXT:
{context}

QUESTION: {query}

Please provide a compliance assessment with specific recommendations.""" 