"""Gemini adapter — Google Generative AI API."""

import json
from nwc.llm.base import BaseLLM, ChatMessage, ChatResponse, EmbeddingResponse


class GeminiAdapter(BaseLLM):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = "", base_url: str = ""):
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self._base = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com/v1beta"

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": m.content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": m.content}]})
        return contents, system_instruction

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import aiohttp
        contents, sys_inst = self._convert_messages(messages)
        payload = {"contents": contents}
        if sys_inst:
            payload["systemInstruction"] = {"parts": [{"text": sys_inst}]}
        url = f"{self._base}/models/{self.model}:generateContent?key={self.api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", str(data)))
                return ChatResponse(
                    content=data["candidates"][0]["content"]["parts"][0]["text"],
                    model=self.model,
                )

    async def embedding(self, text: str | list[str]) -> EmbeddingResponse:
        import aiohttp
        texts = [text] if isinstance(text, str) else text
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for t in texts:
                url = f"{self._base}/models/text-embedding-004:embedContent?key={self.api_key}"
                async with session.post(url, json={"content": {"parts": [{"text": t}]}}) as resp:
                    data = await resp.json()
                    embeddings.append(data["embedding"]["values"])
        return EmbeddingResponse(
            embeddings=embeddings, model="text-embedding-004", dimensions=len(embeddings[0]) if embeddings else 0
        )

    def chat_sync(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        import urllib.request
        contents, sys_inst = self._convert_messages(messages)
        payload = json.dumps({"contents": contents, **({"systemInstruction": {"parts": [{"text": sys_inst}]}} if sys_inst else {})}).encode("utf-8")
        url = f"{self._base}/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", str(data)))
            return ChatResponse(
                content=data["candidates"][0]["content"]["parts"][0]["text"],
                model=self.model,
            )

    def embedding_sync(self, text: str | list[str]) -> EmbeddingResponse:
        import urllib.request
        texts = [text] if isinstance(text, str) else text
        embeddings = []
        for t in texts:
            url = f"{self._base}/models/text-embedding-004:embedContent?key={self.api_key}"
            payload = json.dumps({"content": {"parts": [{"text": t}]}}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                embeddings.append(data["embedding"]["values"])
        return EmbeddingResponse(
            embeddings=embeddings, model="text-embedding-004", dimensions=len(embeddings[0]) if embeddings else 0
        )
