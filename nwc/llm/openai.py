"""OpenAI-compatible adapter (OpenAI, DeepSeek, any /v1 endpoint)."""

import json
from typing import Optional

from nwc.llm.base import BaseLLM, ChatMessage, ChatResponse, EmbeddingResponse


class OpenAIAdapter(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: str = "", base_url: str = ""):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._base = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import aiohttp

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base}/chat/completions", headers=self._headers(), json=payload
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", str(data)))
                choice = data["choices"][0]
                return ChatResponse(
                    content=choice["message"]["content"],
                    model=data.get("model", self.model),
                    usage=data.get("usage", {}),
                )

    async def embedding(self, text: str | list[str]) -> EmbeddingResponse:
        import aiohttp

        texts = [text] if isinstance(text, str) else text
        payload = {"model": "text-embedding-3-small", "input": texts}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base}/embeddings", headers=self._headers(), json=payload
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", str(data)))
                embs = [d["embedding"] for d in data["data"]]
                return EmbeddingResponse(
                    embeddings=embs,
                    model=data.get("model", ""),
                    dimensions=len(embs[0]) if embs else 0,
                )

    def chat_sync(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/chat/completions", data=payload, headers=self._headers()
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data)))
            choice = data["choices"][0]
            return ChatResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
            )

    def embedding_sync(self, text: str | list[str]) -> EmbeddingResponse:
        import urllib.request

        texts = [text] if isinstance(text, str) else text
        payload = json.dumps({"model": "text-embedding-3-small", "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/embeddings", data=payload, headers=self._headers()
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data)))
            embs = [d["embedding"] for d in data["data"]]
            return EmbeddingResponse(
                embeddings=embs, model=data.get("model", ""), dimensions=len(embs[0]) if embs else 0
            )
