"""Embedding providers — local and cloud-based."""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    def __init__(self, model: str = ""):
        self.model = model

    @abstractmethod
    def encode(self, text: str | list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def encode_query(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def encode_document(self, text: str) -> list[float]:
        ...


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Local BGE / sentence-transformers embeddings."""

    def __init__(self, model: str = "BAAI/bge-m3"):
        super().__init__(model=model)
        self._model_obj = None

    def _load(self):
        if self._model_obj is None:
            from sentence_transformers import SentenceTransformer
            self._model_obj = SentenceTransformer(self.model)
        return self._model_obj

    def encode(self, text: str | list[str]) -> list[list[float]]:
        model = self._load()
        texts = [text] if isinstance(text, str) else text
        result = model.encode(texts, normalize_embeddings=True)
        return result.tolist()

    def encode_query(self, text: str) -> list[float]:
        model = self._load()
        result = model.encode(text, normalize_embeddings=True, prompt_name="query")
        return result.tolist()

    def encode_document(self, text: str) -> list[float]:
        model = self._load()
        result = model.encode(text, normalize_embeddings=True)
        return result.tolist()


class TfidfFallbackProvider(BaseEmbeddingProvider):
    """Lightweight TF-IDF fallback — no model download needed."""

    def __init__(self, model: str = "tfidf"):
        super().__init__(model=model)
        self._vectorizer = None
        self._corpus: list[str] = []

    def _load(self):
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=768)
        return self._vectorizer

    def encode(self, text: str | list[str]) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        vectorizer = self._load()
        if not self._corpus:
            vectorizer.fit(texts)
            self._corpus = texts
        result = vectorizer.transform(texts).toarray()
        return result.tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.encode(text)[0]

    def encode_document(self, text: str) -> list[float]:
        return self.encode(text)[0]


PROVIDER_MAP = {
    "sentence-transformers": SentenceTransformersProvider,
    "bge": SentenceTransformersProvider,
    "tfidf": TfidfFallbackProvider,
}


def get_embedding_provider(provider: str, model: str = "") -> BaseEmbeddingProvider:
    cls = PROVIDER_MAP.get(provider)
    if cls is None:
        raise ValueError(f"Unknown embedding provider: {provider}. Supported: {list(PROVIDER_MAP)}")
    return cls(model=model)
