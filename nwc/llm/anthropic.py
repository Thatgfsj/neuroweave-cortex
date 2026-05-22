"""Anthropic adapter — Messages API."""

import json
from nwc.llm.base import BaseLLM, ChatMessage, ChatResponse, EmbeddingResponse


class AnthropicAdapter(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = "", base_url: str = ""):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._base = base_url.rstrip("/") if base_url else "https://api.anthropic.com/v1"
        self._version = "2023-06-01"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self._version,
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        system = None
        converted = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                converted.append({"role": m.role, "content": m.content})
        return system, converted

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import aiohttp
        system, converted = self._convert_messages(messages)
        payload = {"model": self.model, "messages": converted, "max_tokens": kwargs.get("max_tokens", 4096)}
        if system:
            payload["system"] = system
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base}/messages", headers=self._headers(), json=payload
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", str(data)))
                return ChatResponse(
                    content=data["content"][0]["text"],
                    model=data.get("model", self.model),
                    usage=data.get("usage", {}),
                )

    async def embedding(self, text: str | list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Anthropic does not provide embedding models. Use a dedicated embedding provider.")

    def chat_sync(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import urllib.request
        system, converted = self._convert_messages(messages)
        payload = json.dumps({
            "model": self.model, "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
            **({"system": system} if system else {}),
        }).encode("utf-8")
        req = urllib.request.Request(f"{self._base}/messages", data=payload, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data)))
            return ChatResponse(
                content=data["content"][0]["text"],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
            )

    def embedding_sync(self, text: str | list[str]) -> EmbeddingResponse:
        raise NotImplementedError("Anthropic does not provide embedding models.")
