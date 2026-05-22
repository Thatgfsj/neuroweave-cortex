"""DeepSeek adapter — OpenAI-compatible with DeepSeek base URL."""

from nwc.llm.openai import OpenAIAdapter


class DeepSeekAdapter(OpenAIAdapter):
    def __init__(self, model: str = "deepseek-chat", api_key: str = "", base_url: str = ""):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com/v1",
        )
