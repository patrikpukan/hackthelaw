import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.chat.llm_client import (
    GroqClient, OpenAIClient, MockLLMClient, LLMClientFactory,
    ChatMessage, LLMResponse
)


class TestChatMessage:
    """Test ChatMessage model."""
    
    def test_create_chat_message(self):
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"


class TestLLMResponse:
    """Test LLMResponse model."""
    
    def test_create_llm_response(self):
        response = LLMResponse(
            content="Hello there!",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="test-model"
        )
        assert response.content == "Hello there!"
        assert response.usage["prompt_tokens"] == 10
        assert response.model == "test-model"


class TestMockLLMClient:
    """Test MockLLMClient."""
    
    @pytest.mark.asyncio
    async def test_mock_client_response(self):
        client = MockLLMClient()
        messages = [ChatMessage(role="user", content="Test query")]
        
        response = await client.chat_completion(messages)
        
        assert isinstance(response, LLMResponse)
        assert "Test query" in response.content
        assert response.usage is not None
        assert response.model == "mock-model"


class TestGroqClient:
    """Test GroqClient."""
    
    def setup_method(self):
        self.api_key = "test_api_key"
        self.client = GroqClient(api_key=self.api_key)
    
    def test_groq_client_init(self):
        assert self.client.api_key == self.api_key
        assert self.client.base_url == "https://api.groq.com/openai/v1"
        assert self.client.default_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    
    @pytest.mark.asyncio
    async def test_groq_client_no_api_key(self):
        client = GroqClient(api_key=None)
        messages = [ChatMessage(role="user", content="Test")]
        
        with pytest.raises(ValueError, match="Groq API key is required"):
            await client.chat_completion(messages)
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_groq_client_successful_response(self, mock_client):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello! This is a test response."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "meta-llama/llama-4-scout-17b-16e-instruct"
        }
        mock_response.raise_for_status.return_value = None
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = await self.client.chat_completion(messages)
        
        assert response.content == "Hello! This is a test response."
        assert response.usage["total_tokens"] == 15
        assert response.model == "meta-llama/llama-4-scout-17b-16e-instruct"


class TestOpenAIClient:
    """Test OpenAIClient."""
    
    def setup_method(self):
        self.api_key = "test_openai_key"
        self.client = OpenAIClient(api_key=self.api_key)
    
    def test_openai_client_init(self):
        assert self.client.api_key == self.api_key
        assert self.client.base_url == "https://api.openai.com/v1"
        assert self.client.default_model == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_openai_client_no_api_key(self):
        client = OpenAIClient(api_key=None)
        messages = [ChatMessage(role="user", content="Test")]
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            await client.chat_completion(messages)


class TestLLMClientFactory:
    """Test LLMClientFactory."""
    
    def test_create_groq_client(self):
        client = LLMClientFactory.create_client("groq", api_key="test_key")
        assert isinstance(client, GroqClient)
        assert client.api_key == "test_key"
    
    def test_create_openai_client(self):
        client = LLMClientFactory.create_client("openai", api_key="test_key")
        assert isinstance(client, OpenAIClient)
        assert client.api_key == "test_key"
    
    def test_create_mock_client(self):
        client = LLMClientFactory.create_client("mock")
        assert isinstance(client, MockLLMClient)
    
    def test_create_unknown_client(self):
        with pytest.raises(ValueError, match="Unknown LLM provider: unknown"):
            LLMClientFactory.create_client("unknown")
    
    @patch('app.core.chat.llm_client.settings')
    def test_get_default_client_with_groq_key(self, mock_settings):
        mock_settings.llm_provider = 'groq'
        mock_settings.groq_api_key = 'test_groq_key'
        
        client = LLMClientFactory.get_default_client()
        assert isinstance(client, GroqClient)
    
    @patch('app.core.chat.llm_client.settings')
    def test_get_default_client_fallback_to_mock(self, mock_settings):
        mock_settings.llm_provider = 'groq'
        mock_settings.groq_api_key = None
        mock_settings.openai_api_key = None
        
        client = LLMClientFactory.get_default_client()
        assert isinstance(client, MockLLMClient)


# Integration test (requires actual API key)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_groq_client_real_api():
    """Integration test with real Groq API. Requires GROQ_API_KEY environment variable."""
    import os
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY environment variable not set")
    
    client = GroqClient(api_key=api_key)
    messages = [ChatMessage(role="user", content="Say hello in one sentence.")]
    
    try:
        response = await client.chat_completion(messages, max_tokens=50)
        
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.usage is not None
        assert response.model is not None
        
        print(f"Real API Response: {response.content}")
        
    except Exception as e:
        pytest.fail(f"Real API test failed: {e}")


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__]) 