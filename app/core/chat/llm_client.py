from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import httpx
import json
import logging
import asyncio
from pydantic import BaseModel
import os

from app.utils.config import settings

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str  # 'system', 'user', 'assistant'
    content: str


class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, Any]] = None  # Changed from int to Any to handle floats
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[ChatMessage], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate chat completion."""
        pass


class GroqClient(LLMClient):
    """Groq API client for fast language model inference."""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key or getattr(settings, 'groq_api_key', None)
        self.base_url = base_url
        self.default_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        if not self.api_key:
            logger.warning("Groq API key not provided. Set GROQ_API_KEY environment variable.")
    
    async def chat_completion(
        self, 
        messages: List[ChatMessage], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate chat completion using Groq API."""
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        model = model or self.default_model
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'choices' not in data or not data['choices']:
                    raise ValueError("No response choices returned from Groq API")
                
                choice = data['choices'][0]
                message = choice.get('message', {})
                
                response_obj = LLMResponse(
                    content=message.get('content', ''),
                    usage=data.get('usage'),
                    model=data.get('model'),
                    finish_reason=choice.get('finish_reason')
                )

                # Enhanced logging for debugging truncation issues
                logger.info(f"Groq response: finish_reason={response_obj.finish_reason}, "
                           f"content_length={len(response_obj.content)}, "
                           f"usage={response_obj.usage}")

                # Warn if response was truncated
                if response_obj.finish_reason == 'length':
                    logger.warning(f"Groq response was truncated due to max_tokens limit. "
                                 f"Content length: {len(response_obj.content)}")

                return response_obj
                
        except httpx.RequestError as e:
            logger.error(f"Request error calling Groq API: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Groq API: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            raise


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        self.base_url = base_url
        self.default_model = "gpt-3.5-turbo"
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
    
    async def chat_completion(
        self, 
        messages: List[ChatMessage], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate chat completion using OpenAI API."""
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        model = model or self.default_model
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'choices' not in data or not data['choices']:
                    raise ValueError("No response choices returned from OpenAI API")
                
                choice = data['choices'][0]
                message = choice.get('message', {})
                
                response_obj = LLMResponse(
                    content=message.get('content', ''),
                    usage=data.get('usage'),
                    model=data.get('model'),
                    finish_reason=choice.get('finish_reason')
                )

                # Enhanced logging for debugging truncation issues
                logger.info(f"OpenAI response: finish_reason={response_obj.finish_reason}, "
                           f"content_length={len(response_obj.content)}, "
                           f"usage={response_obj.usage}")

                # Warn if response was truncated
                if response_obj.finish_reason == 'length':
                    logger.warning(f"OpenAI response was truncated due to max_tokens limit. "
                                 f"Content length: {len(response_obj.content)}")

                return response_obj
                
        except httpx.RequestError as e:
            logger.error(f"Request error calling OpenAI API: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from OpenAI API: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise


class VertexAIClient(LLMClient):
    """Google Vertex AI client for Gemini models."""

    def __init__(self, project_id: str = None, location: str = "us-central1", model_name: str = "gemini-2.5-pro"):
        self.project_id = project_id or getattr(settings, 'vertex_ai_project_id', None)
        self.location = location or getattr(settings, 'vertex_ai_location', 'us-central1')
        self.model_name = model_name or getattr(settings, 'vertex_ai_model', 'gemini-2.5-pro')
        self.default_model = self.model_name

        # Initialize Vertex AI
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            if not self.project_id:
                logger.warning("Vertex AI project ID not provided. Set VERTEX_AI_PROJECT_ID environment variable.")
                return

            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Vertex AI client with project: {self.project_id}, location: {self.location}, model: {self.model_name}")

        except ImportError:
            logger.error("vertexai package not installed. Install with: pip install vertexai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate chat completion using Vertex AI Gemini."""

        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Vertex AI client not properly initialized")

        try:
            from vertexai.generative_models import GenerationConfig

            # Convert messages to Vertex AI format
            # Vertex AI works better with simpler prompts
            conversation_parts = []
            system_content = ""

            for msg in messages:
                if msg.role == "system":
                    system_content = msg.content
                elif msg.role == "user":
                    conversation_parts.append(msg.content)
                elif msg.role == "assistant":
                    conversation_parts.append(f"Assistant: {msg.content}")

            # Build the final prompt - simpler format for Vertex AI
            if system_content and conversation_parts:
                prompt = f"{system_content}\n\n{conversation_parts[-1]}"  # Use only the last user message
            elif conversation_parts:
                prompt = conversation_parts[-1]  # Just the user message
            else:
                prompt = "Please provide a response."

            # Log the prompt for debugging
            logger.debug(f"Vertex AI prompt (first 200 chars): {prompt[:200]}...")

            # Configure generation parameters with safety settings
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,  # Add top_p for better generation
                top_k=40,    # Add top_k for more diverse responses
            )

            # Import safety settings to prevent blocking
            try:
                from vertexai.generative_models import HarmCategory, HarmBlockThreshold
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            except ImportError:
                logger.warning("Could not import Vertex AI safety settings")
                safety_settings = None

            # Generate response with safety settings
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings if safety_settings else None
                )
            )

            # Extract content from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]

                # More robust content extraction
                content = ""
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        content = candidate.content.parts[0].text if candidate.content.parts[0].text else ""
                    elif hasattr(candidate.content, 'text'):
                        content = candidate.content.text or ""

                # Extract usage information if available
                usage_metadata = getattr(response, 'usage_metadata', None)
                usage = None
                if usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(usage_metadata, 'total_token_count', 0)
                    }

                # Convert finish_reason enum to string if present
                finish_reason_raw = getattr(candidate, 'finish_reason', None)
                finish_reason = None
                if finish_reason_raw is not None:
                    # Handle Vertex AI FinishReason enum
                    if hasattr(finish_reason_raw, 'name'):
                        finish_reason = finish_reason_raw.name.lower()
                    else:
                        finish_reason = str(finish_reason_raw).lower()

                # Debug logging for empty responses
                if not content or len(content.strip()) == 0:
                    logger.warning(f"Vertex AI returned empty content. "
                                 f"Finish reason: {finish_reason}, "
                                 f"Candidate: {candidate}, "
                                 f"Prompt length: {len(prompt)}")

                    # Enhanced safety filter detection
                    safety_blocked = False
                    if hasattr(candidate, 'safety_ratings'):
                        logger.warning(f"Safety ratings: {candidate.safety_ratings}")

                        # Check if any safety rating blocked the response
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'blocked') and rating.blocked:
                                safety_blocked = True
                                logger.warning(f"Content blocked by safety filter: {rating.category} - {rating.probability}")
                            elif hasattr(rating, 'probability'):
                                # Log high probability safety concerns
                                prob_name = getattr(rating.probability, 'name', str(rating.probability))
                                if 'HIGH' in prob_name or 'MEDIUM' in prob_name:
                                    logger.warning(f"Safety concern detected: {rating.category} - {prob_name}")

                    # Override finish_reason if safety filters detected blocking
                    if safety_blocked:
                        finish_reason = 'safety'
                        logger.warning("Overriding finish_reason to 'safety' due to detected safety blocking")

                    # If finish_reason suggests truncation but content is empty,
                    # this might be a Vertex AI issue
                    if finish_reason in ['max_tokens', 'length']:
                        logger.error("Vertex AI reported truncation but returned empty content - possible API issue")

                response_obj = LLMResponse(
                    content=content,
                    usage=usage,
                    model=model or self.default_model,
                    finish_reason=finish_reason
                )

                # Enhanced logging for debugging truncation issues
                logger.info(f"Vertex AI response: finish_reason={response_obj.finish_reason}, "
                           f"content_length={len(response_obj.content)}, "
                           f"usage={response_obj.usage}")

                # Warn if response was truncated or incomplete
                if response_obj.finish_reason in ['max_tokens', 'length']:
                    logger.warning(f"Vertex AI response was truncated due to max_tokens limit. "
                                 f"Content length: {len(response_obj.content)}")
                elif response_obj.finish_reason == 'safety':
                    logger.warning(f"Vertex AI response was blocked due to safety filters.")
                elif not response_obj.content or len(response_obj.content.strip()) == 0:
                    logger.warning(f"Vertex AI returned empty content. Finish reason: {response_obj.finish_reason}")

                return response_obj
            else:
                raise ValueError("No response candidates returned from Vertex AI")

        except Exception as e:
            logger.error(f"Error calling Vertex AI: {e}")
            raise


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.default_model = "mock-model"
    
    async def chat_completion(
        self, 
        messages: List[ChatMessage], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate mock chat completion."""
        
        # Generate a simple mock response based on the last user message
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        mock_content = f"This is a mock response to your query: '{last_user_message[:100]}...'. In a real implementation, this would be generated by an AI model."
        
        return LLMResponse(
            content=mock_content,
            usage={"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
            model=model or self.default_model,
            finish_reason="stop"
        )


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    @staticmethod
    def create_client(provider: str = "groq", **kwargs) -> LLMClient:
        """Create LLM client based on provider."""

        if provider.lower() == "groq":
            return GroqClient(**kwargs)
        elif provider.lower() == "openai":
            return OpenAIClient(**kwargs)
        elif provider.lower() == "vertexai" or provider.lower() == "vertex_ai":
            return VertexAIClient(**kwargs)
        elif provider.lower() == "mock":
            return MockLLMClient()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @staticmethod
    def get_default_client() -> LLMClient:
        """Get default LLM client based on configuration with fallback strategy."""

        # Try to determine the best available client
        provider = getattr(settings, 'llm_provider', 'groq')

        # Try primary provider first
        try:
            if provider == 'groq' and getattr(settings, 'groq_api_key', None):
                return GroqClient()
            elif provider == 'openai' and getattr(settings, 'openai_api_key', None):
                return OpenAIClient()
            elif provider in ['vertexai', 'vertex_ai'] and getattr(settings, 'vertex_ai_project_id', None):
                return VertexAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize primary LLM provider '{provider}': {e}")

        # Fallback strategy: try other available providers
        fallback_providers = ['groq', 'openai', 'vertexai']
        if provider in fallback_providers:
            fallback_providers.remove(provider)

        for fallback_provider in fallback_providers:
            try:
                if fallback_provider == 'groq' and getattr(settings, 'groq_api_key', None):
                    logger.info(f"Falling back to Groq client")
                    return GroqClient()
                elif fallback_provider == 'openai' and getattr(settings, 'openai_api_key', None):
                    logger.info(f"Falling back to OpenAI client")
                    return OpenAIClient()
                elif fallback_provider == 'vertexai' and getattr(settings, 'vertex_ai_project_id', None):
                    logger.info(f"Falling back to Vertex AI client")
                    return VertexAIClient()
            except Exception as e:
                logger.warning(f"Failed to initialize fallback provider '{fallback_provider}': {e}")
                continue

        # Final fallback to mock client
        logger.warning("No LLM providers available. Using mock client.")
        return MockLLMClient()

    @staticmethod
    def create_resilient_client() -> 'ResilientLLMClient':
        """Create a resilient client with multiple fallbacks."""

        # Collect all available clients
        available_clients = []

        # Try to create each client type
        if getattr(settings, 'vertex_ai_project_id', None):
            try:
                available_clients.append(VertexAIClient())
                logger.info("Added Vertex AI client to resilient setup")
            except Exception as e:
                logger.warning(f"Could not add Vertex AI client: {e}")

        if getattr(settings, 'groq_api_key', None):
            try:
                available_clients.append(GroqClient())
                logger.info("Added Groq client to resilient setup")
            except Exception as e:
                logger.warning(f"Could not add Groq client: {e}")

        if getattr(settings, 'openai_api_key', None):
            try:
                available_clients.append(OpenAIClient())
                logger.info("Added OpenAI client to resilient setup")
            except Exception as e:
                logger.warning(f"Could not add OpenAI client: {e}")

        # Always add mock client as final fallback
        available_clients.append(MockLLMClient())

        if len(available_clients) == 1:
            # Only mock client available
            logger.warning("Only mock client available for resilient setup")
            return ResilientLLMClient(available_clients[0], [])

        # Use first client as primary, rest as fallbacks
        primary = available_clients[0]
        fallbacks = available_clients[1:]

        logger.info(f"Created resilient client with {len(fallbacks)} fallbacks")
        return ResilientLLMClient(primary, fallbacks)


class ResilientLLMClient(LLMClient):
    """Resilient LLM client wrapper with fallback mechanisms."""

    def __init__(self, primary_client: LLMClient, fallback_clients: List[LLMClient] = None):
        self.primary_client = primary_client
        self.fallback_clients = fallback_clients or []
        self.default_model = getattr(primary_client, 'default_model', 'unknown')

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate chat completion with fallback strategy."""

        # Try primary client first
        try:
            response = await self.primary_client.chat_completion(
                messages, model, temperature, max_tokens
            )

            # Enhanced response validation
            if response and response.content and response.content.strip():
                # Check if response was truncated and might need retry with higher token limit
                if response.finish_reason in ['length', 'max_tokens'] and len(response.content) < 100:
                    logger.warning(f"Primary client returned very short truncated response "
                                 f"(length: {len(response.content)}), trying fallbacks")
                else:
                    return response
            else:
                logger.warning("Primary client returned empty response, trying fallbacks")

        except Exception as e:
            logger.warning(f"Primary client failed: {e}, trying fallbacks")

        # Try fallback clients
        for i, fallback_client in enumerate(self.fallback_clients):
            try:
                logger.info(f"Trying fallback client {i+1}/{len(self.fallback_clients)}")
                response = await fallback_client.chat_completion(
                    messages, model, temperature, max_tokens
                )

                # Enhanced response validation for fallbacks
                if response and response.content and response.content.strip():
                    # Accept fallback response even if truncated, as it's better than nothing
                    if response.finish_reason in ['length', 'max_tokens']:
                        logger.warning(f"Fallback client {i+1} returned truncated response "
                                     f"(length: {len(response.content)}), but accepting it")
                    else:
                        logger.info(f"Fallback client {i+1} succeeded")
                    return response
                else:
                    logger.warning(f"Fallback client {i+1} returned empty response")

            except Exception as e:
                logger.warning(f"Fallback client {i+1} failed: {e}")
                continue

        # Final fallback: return a structured error response
        logger.error("All LLM clients failed, returning error response")
        return LLMResponse(
            content="I apologize, but I'm experiencing technical difficulties and cannot process your request at the moment. Please try again later.",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=model or self.default_model,
            finish_reason="error"
        )

