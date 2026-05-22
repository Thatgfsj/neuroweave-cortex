"""LLM-agnostic adapter base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    role: str  # system, user, assistant
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str = ""
    dimensions: int = 0


class BaseLLM(ABC):
    """LLM-agnostic adapter. Implementations handle provider-specific APIs."""

    def __init__(self, model: str = "", api_key: str = "", base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def embedding(self, text: str | list[str]) -> EmbeddingResponse:
        """Get embeddings for text."""
        ...

    @abstractmethod
    def chat_sync(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Synchronous chat completion."""
        ...

    @abstractmethod
    def embedding_sync(self, text: str | list[str]) -> EmbeddingResponse:
        """Synchronous embedding."""
        ...
