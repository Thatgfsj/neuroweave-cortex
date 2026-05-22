"""NWC configuration management — YAML + env vars, ENV > config.yaml > defaults."""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml

DEFAULT_CONFIG_YAML = """
llm:
  provider: openai
  model: gpt-4o
  api_key: ""
  base_url: ""

embedding:
  provider: sentence-transformers
  model: BAAI/bge-m3

memory:
  backend: cortexgraph
  max_depth: 5
  working_capacity: 20
  consolidation_interval: 3600

retrieval:
  top_k: 8
  rerank: true
  fusion: hybrid

storage:
  path: ~/.nwc/data

server:
  host: 127.0.0.1
  port: 8765
"""


@dataclass
class LlmConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model: str = "BAAI/bge-m3"


@dataclass
class MemoryConfig:
    backend: str = "cortexgraph"
    max_depth: int = 5
    working_capacity: int = 20
    consolidation_interval: int = 3600


@dataclass
class RetrievalConfig:
    top_k: int = 8
    rerank: bool = True
    fusion: str = "hybrid"


@dataclass
class StorageConfig:
    path: str = "~/.nwc/data"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class NwcConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "NwcConfig":
        return cls(
            llm=LlmConfig(**d.get("llm", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            memory=MemoryConfig(**d.get("memory", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
            storage=StorageConfig(**d.get("storage", {})),
            server=ServerConfig(**d.get("server", {})),
        )

    def to_dict(self) -> dict:
        return {
            "llm": asdict(self.llm),
            "embedding": asdict(self.embedding),
            "memory": asdict(self.memory),
            "retrieval": asdict(self.retrieval),
            "storage": asdict(self.storage),
            "server": asdict(self.server),
        }


def _config_path() -> Path:
    return Path(os.environ.get("NWC_CONFIG_PATH", Path.home() / ".nwc" / "config.yaml"))


def _apply_env_overrides(cfg: NwcConfig) -> NwcConfig:
    """Apply environment variable overrides. ENV > config.yaml > defaults."""
    if os.environ.get("NWC_API_KEY"):
        cfg.llm.api_key = os.environ["NWC_API_KEY"]
    if os.environ.get("NWC_MODEL"):
        cfg.llm.model = os.environ["NWC_MODEL"]
    if os.environ.get("NWC_PROVIDER"):
        cfg.llm.provider = os.environ["NWC_PROVIDER"]
    if os.environ.get("NWC_BASE_URL"):
        cfg.llm.base_url = os.environ["NWC_BASE_URL"]
    if os.environ.get("NWC_SERVER_HOST"):
        cfg.server.host = os.environ["NWC_SERVER_HOST"]
    if os.environ.get("NWC_SERVER_PORT"):
        cfg.server.port = int(os.environ["NWC_SERVER_PORT"])
    return cfg


def init_config(provider: str = "", model: str = "", api_key: str = "",
                base_url: str = "", embedding_provider: str = "",
                embedding_model: str = "") -> NwcConfig:
    """Initialize and persist config. Creates ~/.nwc/config.yaml."""
    cfg_path = _config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)

    if provider:
        raw.setdefault("llm", {})["provider"] = provider
    if model:
        raw.setdefault("llm", {})["model"] = model
    if api_key:
        raw.setdefault("llm", {})["api_key"] = api_key
    if base_url:
        raw.setdefault("llm", {})["base_url"] = base_url
    if embedding_provider:
        raw.setdefault("embedding", {})["provider"] = embedding_provider
    if embedding_model:
        raw.setdefault("embedding", {})["model"] = embedding_model

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True)

    cfg = NwcConfig.from_dict(raw)
    return _apply_env_overrides(cfg)


def get_config() -> NwcConfig:
    """Load config: ENV > config.yaml > defaults."""
    cfg_path = _config_path()
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)
    cfg = NwcConfig.from_dict(raw)
    return _apply_env_overrides(cfg)


def reset_config() -> None:
    """Remove config file, reverting to defaults."""
    cfg_path = _config_path()
    if cfg_path.exists():
        cfg_path.unlink()


# Module-level cache
_config: Optional[NwcConfig] = None
