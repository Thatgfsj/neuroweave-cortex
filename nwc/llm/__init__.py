from nwc.llm.base import BaseLLM
from nwc.llm.openai import OpenAIAdapter
from nwc.llm.deepseek import DeepSeekAdapter
from nwc.llm.anthropic import AnthropicAdapter
from nwc.llm.ollama import OllamaAdapter
from nwc.llm.gemini import GeminiAdapter

PROVIDER_MAP = {
    "openai": OpenAIAdapter,
    "deepseek": DeepSeekAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
    "gemini": GeminiAdapter,
}


def get_llm(provider: str, model: str, api_key: str = "", base_url: str = "") -> BaseLLM:
    """Factory: get the right LLM adapter for a provider."""
    cls = PROVIDER_MAP.get(provider)
    if cls is None:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: {list(PROVIDER_MAP)}")
    return cls(model=model, api_key=api_key, base_url=base_url)
