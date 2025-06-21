from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import asyncio

from app.core.chat.llm_client import LLMClient, LLMClientFactory, ChatMessage, LLMResponse
from app.core.chat.enhanced_retriever import EnhancedDocumentRetriever
from app.core.chat.adaptive_limits import AdaptiveLimitsManager, create_adaptive_limits_manager
from app.core.chat.enhanced_citations import create_enhanced_citations, format_citations_for_response, generate_bibliography
from app.core.chat.search_cache import get_search_cache, cached_search
from app.core.chat.response_validator import response_validator
from app.core.chat.token_config import token_config_manager
from app.core.chat.vertex_ai_helper import vertex_ai_helper
from app.db.connection import async_session
from app.db.models import ChatSession, ChatMessage as DBChatMessage

logger = logging.getLogger(__name__)


class IterativeRAGGenerator:
    """
    Advanced RAG system with iterative retrieval and query rewriting.
    
    Pipeline:
    1. Query Analysis & Rewriting - Convert user question to search queries
    2. Initial Retrieval - Find relevant documents
    3. Self-Assessment - Evaluate if enough information found
    4. Iterative Search - Additional searches if needed
    5. Response Generation - Generate final answer
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, retriever: Optional[EnhancedDocumentRetriever] = None, use_resilient_client: bool = True):
        if use_resilient_client and llm_client is None:
            # Use resilient client for better fallback handling
            self.llm_client = LLMClientFactory.create_resilient_client()
        else:
            self.llm_client = llm_client or LLMClientFactory.get_default_client()
        self.retriever = retriever or EnhancedDocumentRetriever()
        self.limits_manager = create_adaptive_limits_manager()
        self.search_cache = get_search_cache()
        
        # System prompts for different tasks
        self.query_rewriter_prompt = """You are an expert at converting user questions into effective search queries for legal and technical documents.

Your task: Convert the user's question into 1-3 specific search queries that will help find relevant information.

Guidelines:
1. Extract key concepts, entities, legal terms, and technical terminology
2. For numbered items (stages, steps, phases), include both the number and descriptive terms
3. Consider synonyms, related terms, and alternative phrasings
4. Think about what document sections would contain this information
5. Generate queries of different specificity levels
6. For technical frameworks, include both formal names and common abbreviations

Special handling for numbered concepts:
- If asking about "5th stage", also search for "stage 5", "step 5", "phase 5"
- Include descriptive terms that might appear instead of numbers
- Consider that documents might use different numbering systems

User Question: {question}

Respond with a JSON array of search queries, ordered by priority:
```json
[
  "specific search query 1",
  "broader search query 2",
  "alternative angle query 3"
]
```"""

        self.assessment_prompt = """You are evaluating whether retrieved document chunks contain sufficient information to answer a user's question.

User Question: {question}

Retrieved Information:
{context}

Evaluate:
1. Is there enough information to provide a complete answer?
2. What specific information is missing (if any)?
3. What additional search terms might find missing information?

Respond with JSON:
```json
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_info": "description of missing information",
  "additional_queries": ["query1", "query2"]
}}
```"""

        self.response_prompt = """You are a specialized legal AI assistant. Answer the user's question based on the provided document context.

Key Instructions:
1. Provide a comprehensive answer based on the document context
2. If information is incomplete, clearly state what's missing
3. Cite specific documents and sections when possible
4. For legal matters, do not recommend consulting qualified legal counsel, as this is done by the company's legal team.
5. Be precise about legal terms, dates, and requirements

User Question: {question}

Document Context:
{context}

Provide a clear, detailed answer based on the available information."""

    async def generate_response(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_chunks_per_iteration: Optional[int] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate response using adaptive iterative RAG approach."""
        
        try:
            logger.info(f"Starting adaptive iterative RAG for query: {query[:100]}...")
            
            # Step 1: Analyze query complexity and set adaptive limits
            adaptive_limits = self.limits_manager.get_initial_limits(query)
            
            # Override with user-provided limits if specified
            effective_max_iterations = max_iterations or adaptive_limits.max_iterations
            effective_max_chunks = max_chunks_per_iteration or adaptive_limits.max_chunks_per_iteration
            
            logger.info(f"Adaptive limits: max_iterations={effective_max_iterations}, "
                       f"max_chunks={effective_max_chunks}")
            
            # Step 2: Query Analysis & Rewriting (with caching)
            search_queries = await self._rewrite_query_cached(query)
            logger.info(f"Generated {len(search_queries)} search queries")
            
            # Step 3: Iterative Retrieval with adaptive limits
            all_chunks = []
            iteration = 0
            
            while iteration < effective_max_iterations:
                iteration += 1
                
                # Evaluate current search quality
                search_quality = self.limits_manager.evaluate_search_quality(all_chunks, query)
                
                # Check if we should continue
                should_continue, reason = self.limits_manager.should_continue_search(
                    iteration - 1, search_quality, adaptive_limits, query
                )
                
                if not should_continue and iteration > 1:
                    logger.info(f"Early stopping: {reason}")
                    break
                
                # Adjust chunk limit dynamically
                dynamic_chunk_limit = self.limits_manager.adjust_chunk_limit(
                    iteration, search_quality, effective_max_chunks
                )
                
                logger.info(f"Iteration {iteration}: Searching with {len(search_queries)} queries, "
                           f"limit={dynamic_chunk_limit}")
                
                # Retrieve chunks for current queries (with caching)
                new_chunks = await self._retrieve_chunks_cached(
                    search_queries, 
                    dynamic_chunk_limit,
                    document_ids,
                    exclude_chunk_ids=[chunk.get('chunk', {}).get('id') for chunk in all_chunks]
                )
                
                all_chunks.extend(new_chunks)
                
                # Step 4: Self-Assessment for additional queries
                if iteration < effective_max_iterations:
                    assessment = await self._assess_information_sufficiency(query, all_chunks)
                    
                    logger.info(f"Assessment: sufficient={assessment['sufficient']}, "
                               f"confidence={assessment['confidence']}")
                    
                    # Update search queries for next iteration
                    if assessment.get('additional_queries'):
                        search_queries = assessment['additional_queries']
                    else:
                        break  # No more queries to try
            
            # Step 5: Create enhanced citations
            enhanced_citations = create_enhanced_citations(all_chunks)
            
            # Step 6: Generate Final Response with enhanced context
            context = self._build_context_with_citations(all_chunks, enhanced_citations)
            final_response = await self._generate_final_response(query, context)
            
            # Step 7: Format response with all enhancements
            final_search_quality = self.limits_manager.evaluate_search_quality(all_chunks, query)
            resource_summary = self.limits_manager.get_resource_usage_summary(
                iteration, len(all_chunks), final_search_quality
            )
            
            formatted_response = self._format_enhanced_response(
                final_response, 
                all_chunks,
                enhanced_citations,
                query,
                iteration,
                resource_summary
            )
            
            # Step 8: Save conversation
            if session_id:
                await self._save_conversation(session_id, query, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in adaptive iterative RAG: {e}")
            return {
                "content": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "error": True,
                "query": query,
                "resource_usage": {"error": str(e)}
            }
    
    async def _rewrite_query(self, query: str) -> List[str]:
        """Convert user question into effective search queries."""

        try:
            prompt = self.query_rewriter_prompt.format(question=query)

            response = await self.llm_client.chat_completion(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=500
            )

            # Special handling for Vertex AI empty responses
            if (not response or not response.content) and hasattr(self.llm_client, 'primary_client'):
                logger.warning("Primary LLM returned empty response for query rewriting, trying simpler prompt")

                # Try with a much simpler prompt
                simple_prompt = f"Convert this question into 2-3 search queries: {query}"
                response = await self.llm_client.chat_completion(
                    [ChatMessage(role="user", content=simple_prompt)],
                    temperature=0.5,
                    max_tokens=300
                )

            # Validate response content
            if not response or not response.content:
                logger.warning(f"Empty response from LLM for query rewriting. "
                             f"Finish reason: {response.finish_reason if response else 'No response'}")

                # Use Vertex AI helper to create fallback queries
                fallback_queries = vertex_ai_helper.create_fallback_queries(query)
                logger.info(f"Using fallback queries: {fallback_queries}")
                return fallback_queries

            # Parse JSON response with improved error handling
            content = response.content.strip()

            # Handle empty content
            if not content:
                logger.warning("Empty content in LLM response for query rewriting")
                return [query]

            # Extract JSON from markdown code blocks if present
            if content.startswith("```json"):
                json_parts = content.split("```json")
                if len(json_parts) > 1:
                    json_content = json_parts[1].split("```")[0].strip()
                else:
                    logger.warning("Malformed JSON markdown in LLM response")
                    return [query]
            elif content.startswith("```"):
                # Handle generic code blocks
                json_parts = content.split("```")
                if len(json_parts) >= 3:
                    json_content = json_parts[1].strip()
                else:
                    logger.warning("Malformed code block in LLM response")
                    return [query]
            else:
                json_content = content

            # Validate JSON content before parsing
            if not json_content or json_content.isspace():
                logger.warning("Empty JSON content after extraction")
                return [query]

            # Attempt to parse JSON
            try:
                search_queries = json.loads(json_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error: {json_err}. Content: '{json_content[:200]}...'")

                # Use Vertex AI helper to extract queries from malformed response
                extracted_queries = vertex_ai_helper.extract_queries_from_simple_response(json_content, query)
                if extracted_queries and extracted_queries != [query]:
                    logger.info(f"Extracted {len(extracted_queries)} queries using Vertex AI helper")
                    return extracted_queries
                else:
                    # Final fallback
                    fallback_queries = vertex_ai_helper.create_fallback_queries(query)
                    logger.info(f"Using fallback queries: {fallback_queries}")
                    return fallback_queries

            # Ensure we have a list of strings
            if isinstance(search_queries, list):
                valid_queries = [str(q).strip() for q in search_queries if str(q).strip()]
                if valid_queries:
                    return valid_queries
                else:
                    logger.warning("No valid queries found in parsed response")
                    return [query]
            else:
                logger.warning(f"Expected list but got {type(search_queries)}: {search_queries}")
                return [query]  # Fallback to original query

        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return [query]  # Fallback to original query

    async def _rewrite_query_cached(self, query: str) -> List[str]:
        """Convert user question into effective search queries with caching."""
        
        cache_key = f"rewrite:{query}"
        
        # Try to get from cache
        cached_result = self.search_cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached query rewrite for: {query[:50]}...")
            return [r.get('query', '') for r in cached_result if r.get('query')]
        
        # Generate new queries
        search_queries = await self._rewrite_query(query)
        
        # Cache the result
        cache_results = [{'query': q} for q in search_queries]
        self.search_cache.put(cache_key, cache_results, ttl_seconds=1800)  # 30 minutes
        
        return search_queries
    
    async def _retrieve_chunks(
        self, 
        queries: List[str], 
        max_chunks: int,
        document_ids: Optional[List[str]] = None,
        exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve chunks for multiple queries."""
        
        all_chunks = []
        chunks_per_query = max(1, max_chunks // len(queries))
        
        # Execute searches in parallel
        tasks = []
        for query in queries:
            task = self.retriever.retrieve_relevant_chunks(
                query,
                limit=chunks_per_query,
                document_ids=document_ids
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and deduplicate
        seen_chunk_ids = set(exclude_chunk_ids or [])
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in retrieval: {result}")
                continue
            
            # Ensure result is iterable (should be a list of chunks)
            if not isinstance(result, (list, tuple)):
                logger.error(f"Unexpected result type: {type(result)}")
                continue
                
            for chunk in result:
                chunk_id = chunk.get('chunk', {}).get('id')
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
        
        # Sort by relevance and limit
        all_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return all_chunks[:max_chunks]

    async def _retrieve_chunks_cached(
        self, 
        queries: List[str], 
        max_chunks: int,
        document_ids: Optional[List[str]] = None,
        exclude_chunk_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Retrieve chunks for multiple queries with intelligent caching."""
        
        all_chunks = []
        seen_chunk_ids = set(exclude_chunk_ids or [])
        chunks_per_query = max(1, max_chunks // len(queries))
        
        for query in queries:
            cache_key = f"search:{query}:{chunks_per_query}:{str(document_ids)}"
            
            # Try to get from cache
            cached_chunks = self.search_cache.get(cache_key)
            if cached_chunks:
                logger.info(f"Using cached search results for: {query[:50]}...")
                
                # Filter out already seen chunks
                for chunk in cached_chunks:
                    chunk_id = chunk.get('chunk', {}).get('id')
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                        
                        if len(all_chunks) >= max_chunks:
                            break
            else:
                # Perform new search
                try:
                    results = await self.retriever.retrieve_relevant_chunks(
                        query,
                        limit=chunks_per_query,
                        document_ids=document_ids
                    )
                    
                    # Cache the results
                    if results:
                        self.search_cache.put(cache_key, results, ttl_seconds=3600)  # 1 hour
                        logger.info(f"Cached search results for: {query[:50]}...")
                    
                    # Add to results
                    for chunk in results:
                        chunk_id = chunk.get('chunk', {}).get('id')
                        if chunk_id and chunk_id not in seen_chunk_ids:
                            all_chunks.append(chunk)
                            seen_chunk_ids.add(chunk_id)
                            
                            if len(all_chunks) >= max_chunks:
                                break
                                
                except Exception as e:
                    logger.error(f"Error retrieving chunks for query '{query}': {e}")
                    continue
            
            # Stop when we have enough chunks overall
            if len(all_chunks) >= max_chunks:
                break
        
        # Sort by relevance and limit
        all_chunks.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        logger.info(f"Retrieved {len(all_chunks)} unique chunks (with caching)")
        return all_chunks[:max_chunks]
    
    async def _assess_information_sufficiency(
        self,
        query: str,
        chunks: List[Dict]
    ) -> Dict[str, Any]:
        """Assess if retrieved information is sufficient to answer the question."""

        try:
            context = self._build_context(chunks)
            prompt = self.assessment_prompt.format(question=query, context=context)

            response = await self.llm_client.chat_completion(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=300
            )

            # Validate response content
            if not response or not response.content:
                logger.warning(f"Empty response from LLM for assessment. "
                             f"Finish reason: {response.finish_reason if response else 'No response'}")

                # Try with a simpler assessment prompt if using resilient client
                if hasattr(self.llm_client, 'primary_client'):
                    logger.info("Trying simpler assessment prompt")
                    simple_prompt = f"Is this information sufficient to answer '{query}'? Answer yes or no and explain briefly."

                    try:
                        simple_response = await self.llm_client.chat_completion(
                            [ChatMessage(role="user", content=simple_prompt)],
                            temperature=0.1,
                            max_tokens=200
                        )

                        if simple_response and simple_response.content:
                            # Parse simple response
                            content = simple_response.content.lower()
                            sufficient = "yes" in content or "sufficient" in content
                            return {
                                "sufficient": sufficient,
                                "confidence": 0.7 if sufficient else 0.3,
                                "missing_info": "Assessment based on simplified prompt",
                                "additional_queries": []
                            }
                    except Exception as e:
                        logger.warning(f"Simple assessment also failed: {e}")

                return self._get_fallback_assessment(chunks)

            # Parse JSON response with improved error handling
            content = response.content.strip()

            # Handle empty content
            if not content:
                logger.warning("Empty content in LLM response for assessment")
                return self._get_fallback_assessment(chunks)

            # Extract JSON from markdown code blocks if present
            if content.startswith("```json"):
                json_parts = content.split("```json")
                if len(json_parts) > 1:
                    json_content = json_parts[1].split("```")[0].strip()
                else:
                    logger.warning("Malformed JSON markdown in assessment response")
                    return self._get_fallback_assessment(chunks)
            elif content.startswith("```"):
                # Handle generic code blocks
                json_parts = content.split("```")
                if len(json_parts) >= 3:
                    json_content = json_parts[1].strip()
                else:
                    logger.warning("Malformed code block in assessment response")
                    return self._get_fallback_assessment(chunks)
            else:
                json_content = content

            # Validate JSON content before parsing
            if not json_content or json_content.isspace():
                logger.warning("Empty JSON content after extraction in assessment")
                return self._get_fallback_assessment(chunks)

            # Attempt to parse JSON
            try:
                assessment = json.loads(json_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error in assessment: {json_err}. Content: '{json_content[:200]}...'")
                return self._get_fallback_assessment(chunks)

            # Validate response structure and provide defaults
            return {
                "sufficient": bool(assessment.get("sufficient", False)),
                "confidence": max(0.0, min(1.0, float(assessment.get("confidence", 0.0)))),
                "missing_info": str(assessment.get("missing_info", "")),
                "additional_queries": [
                    str(q).strip() for q in assessment.get("additional_queries", [])
                    if str(q).strip()
                ]
            }

        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            return self._get_fallback_assessment(chunks)

    def _get_fallback_assessment(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Get fallback assessment when LLM assessment fails."""
        # Conservative fallback based on chunk count and content
        has_chunks = len(chunks) > 0
        confidence = 0.6 if has_chunks else 0.0

        # If we have chunks, assume we might have some useful information
        sufficient = has_chunks and len(chunks) >= 2

        return {
            "sufficient": sufficient,
            "confidence": confidence,
            "missing_info": "Could not assess information sufficiency due to LLM error",
            "additional_queries": []
        }
    
    async def _generate_final_response(self, query: str, context: str) -> LLMResponse:
        """Generate the final response based on all retrieved information."""

        prompt = self.response_prompt.format(question=query, context=context)

        # Use intelligent token allocation based on query type
        max_tokens = token_config_manager.get_max_tokens(query, len(context))
        logger.info(f"Using {max_tokens} max_tokens for query type")

        response = await self.llm_client.chat_completion(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=max_tokens
        )

        # Enhanced validation using response validator
        query_lower = query.lower()
        expected_type = "summary" if any(keyword in query_lower for keyword in ['summarize', 'summary']) else "general"

        validation_result = response_validator.validate_response(response, query, expected_type)
        response_validator.log_validation_result(validation_result, query)

        # Additional logging for debugging
        if not validation_result.is_valid:
            logger.warning(f"Response validation failed: {validation_result.issues}")
            if validation_result.suggestions:
                logger.info(f"Suggestions for improvement: {validation_result.suggestions}")

        return response
    
    def _build_context(self, retrieval_results: List[Dict]) -> str:
        """Build context string from retrieved document chunks."""
        
        if not retrieval_results:
            return "No relevant documents found in the corpus."
        
        context_parts = ["RELEVANT DOCUMENT INFORMATION:\n"]
        
        for i, result in enumerate(retrieval_results, 1):
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            similarity = result.get('similarity_score', 0.0)
            
            context_part = f"""
Document {i}:
- Source: {document.get('filename') or document.get('title') or document.get('name') or 'Unknown Document'}
- Section: {chunk.get('chunk_type', 'general')}
- Relevance: {similarity:.3f}
- Content: {chunk.get('text', '')}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def _build_context_with_citations(self, retrieval_results: List[Dict], citations: List) -> str:
        """Build context string with enhanced citations."""
        
        if not retrieval_results:
            return "No relevant documents found in the corpus."
        
        context_parts = ["RELEVANT DOCUMENT INFORMATION WITH PRECISE CITATIONS:\n"]
        
        for i, (result, citation) in enumerate(zip(retrieval_results, citations), 1):
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            similarity = result.get('similarity_score', 0.0)
            
            # Build citation reference
            citation_ref = []
            if citation.article_number:
                citation_ref.append(f"Art. {citation.article_number}")
            if citation.section_number:
                citation_ref.append(f"§ {citation.section_number}")
            if citation.paragraph_number:
                citation_ref.append(f"¶ {citation.paragraph_number}")
            if citation.page_number:
                citation_ref.append(f"p. {citation.page_number}")
            
            citation_str = ", ".join(citation_ref) if citation_ref else "General"
            
            context_part = f"""
Document {i} [{citation_str}]:
- Source: {document.get('filename') or document.get('title') or document.get('name') or 'Unknown Document'}
- Reference: {citation_str}
- Dates: {citation.date_mentioned or citation.effective_date or 'N/A'}
- Relevance: {similarity:.3f}
- Content: {chunk.get('text', '')}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _format_response(
        self, 
        llm_response: LLMResponse, 
        retrieval_results: List[Dict],
        original_query: str,
        iterations_used: int = 1
    ) -> Dict[str, Any]:
        """Format the final response with sources and metadata."""
        
        # Extract source information
        sources = []
        for result in retrieval_results:
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            
            source = {
                "document_name": document.get('filename') or document.get('title') or document.get('name') or 'Unknown Document',
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
            "iterations_used": iterations_used,
            "total_chunks_found": len(retrieval_results),
            "timestamp": datetime.utcnow().isoformat(),
            "error": False
        }

    def _format_enhanced_response(
        self, 
        llm_response: LLMResponse, 
        retrieval_results: List[Dict],
        citations: List,
        original_query: str,
        iterations_used: int,
        resource_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the final response with enhanced citations and resource tracking."""
        
        # Format enhanced citations
        formatted_citations = format_citations_for_response(citations, style='legal')
        
        # Extract enhanced source information with precise citations
        sources = []
        for result, citation in zip(retrieval_results, citations):
            chunk = result.get('chunk', {})
            document = result.get('document', {})
            
            source = {
                "document_name": document.get('filename') or document.get('title') or document.get('name') or 'Unknown Document',
                "document_id": document.get('id'), 
                "chunk_type": chunk.get('chunk_type', 'general'),
                "similarity_score": result.get('similarity_score', 0.0),
                "chunk_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                # Enhanced citation information
                "article_number": citation.article_number,
                "section_number": citation.section_number,
                "paragraph_number": citation.paragraph_number,
                "page_number": citation.page_number,
                "date_mentioned": citation.date_mentioned,
                "effective_date": citation.effective_date,
                "formatted_citation": formatted_citations[sources.__len__()]  # Get corresponding citation
            }
            sources.append(source)
        
        # Generate bibliography
        bibliography = generate_bibliography(citations)
        
        return {
            "content": llm_response.content,
            "sources": sources,
            "citations": formatted_citations,
            "bibliography": bibliography,
            "query": original_query,
            "model_used": llm_response.model,
            "usage": llm_response.usage,
            "iterations_used": iterations_used,
            "total_chunks_found": len(retrieval_results),
            "resource_usage": resource_summary,
            "rag_approach": "iterative_adaptive",
            "cache_stats": self.search_cache.get_stats(),
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
                    timestamp=datetime.utcnow()
                )
                session.add(user_message)
                
                # Save assistant response
                assistant_message = DBChatMessage(
                    session_id=session_id,
                    message_type="assistant",
                    content=response["content"],
                    metadata={
                        "sources": response["sources"],
                        "iterations_used": response["iterations_used"],
                        "total_chunks": response["total_chunks_found"]
                    },
                    timestamp=datetime.utcnow()
                )
                session.add(assistant_message)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


class QueryExpansionStrategy:
    """Different strategies for expanding queries."""
    
    @staticmethod
    def legal_domain_expansion(query: str) -> List[str]:
        """Expand query with legal domain knowledge."""
        
        # Legal synonyms and related terms
        legal_expansions = {
            "definition": ["define", "meaning", "term", "definition", "what is", "what does", "refers to"],
            "rights": ["rights", "privileges", "entitlements", "permissions", "authority"],
            "obligations": ["obligations", "duties", "responsibilities", "requirements", "must"],
            "liability": ["liability", "responsibility", "damages", "compensation", "penalties"],
            "termination": ["termination", "end", "conclude", "expire", "cancel", "dissolve"],
            "agreement": ["agreement", "contract", "document", "terms", "conditions"],
            "parties": ["parties", "participant", "entity", "organization", "individual"]
        }
        
        expanded_queries = [query]
        query_lower = query.lower()
        
        for concept, terms in legal_expansions.items():
            if concept in query_lower:
                for term in terms:
                    if term not in query_lower:
                        expanded_queries.append(query.replace(concept, term))
        
        return expanded_queries[:3]  # Limit to top 3 expansions 