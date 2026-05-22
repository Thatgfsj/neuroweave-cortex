"""Ollama adapter — local LLM via Ollama API."""

import json
from nwc.llm.base import BaseLLM, ChatMessage, ChatResponse, EmbeddingResponse


class OllamaAdapter(BaseLLM):
    def __init__(self, model: str = "llama3", api_key: str = "", base_url: str = ""):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._base = base_url.rstrip("/") if base_url else "http://localhost:11434/api"

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import aiohttp
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._base}/chat", json=payload) as resp:
                data = await resp.json()
                return ChatResponse(
                    content=data["message"]["content"],
                    model=data.get("model", self.model),
                )

    async def embedding(self, text: str | list[str]) -> EmbeddingResponse:
        import aiohttp
        texts = [text] if isinstance(text, str) else text
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for t in texts:
                async with session.post(
                    f"{self._base}/embeddings", json={"model": self.model, "prompt": t}
                ) as resp:
                    data = await resp.json()
                    embeddings.append(data["embedding"])
        return EmbeddingResponse(
            embeddings=embeddings, model=self.model, dimensions=len(embeddings[0]) if embeddings else 0
        )

    def chat_sync(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self._base}/chat", data=payload)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return ChatResponse(content=data["message"]["content"], model=data.get("model", self.model))

    def embedding_sync(self, text: str | list[str]) -> EmbeddingResponse:
        import urllib.request
        texts = [text] if isinstance(text, str) else text
        embeddings = []
        for t in texts:
            payload = json.dumps({"model": self.model, "prompt": t}).encode("utf-8")
            req = urllib.request.Request(f"{self._base}/embeddings", data=payload)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                embeddings.append(data["embedding"])
        return EmbeddingResponse(
            embeddings=embeddings, model=self.model, dimensions=len(embeddings[0]) if embeddings else 0
        )
