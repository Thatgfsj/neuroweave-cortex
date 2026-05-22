"""NWC API Server — OpenAI-compatible REST API.

Endpoints:
    /v1/chat/completions   — OpenAI-compatible chat (with memory-augmented context)
    /health                — Health check
    /memory/write          — Store a memory
    /memory/query          — Retrieve memories
    /memory/context        — Get cognitive context for prompt injection
    /graph                 — Get memory graph overview
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nwc.core.cortex import Cortex

cortex: Optional[Cortex] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cortex
    cortex = Cortex()
    yield


app = FastAPI(
    title="NeuroWeave Cortex API",
    version="1.1.0",
    description="Cognitive Runtime API — memory, context, and graph for LLM agents.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nwc-cortex"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    inject_memory: bool = True


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = "nwc-chat"
    object: str = "chat.completion"
    created: int = 0
    model: str = "nwc-cortex"
    choices: list[ChatChoice]
    usage: ChatUsage = ChatUsage()


class MemoryWriteRequest(BaseModel):
    text: str
    tags: list[str] = []
    importance: float = 0.5
    emotional_valence: float = 0.0


class MemoryWriteResponse(BaseModel):
    id: str
    status: str = "stored"


class MemoryQueryRequest(BaseModel):
    query: str
    max_items: int = 8
    context: Optional[dict] = None


class MemoryItem(BaseModel):
    id: str
    content: str
    score: float = 0.0
    tags: list[str] = []


class MemoryQueryResponse(BaseModel):
    memory: list[MemoryItem] = []
    entities: list[str] = []
    relations: list[dict] = []
    summary: str = ""


class ContextRequest(BaseModel):
    prompt: str = ""


class ContextResponse(BaseModel):
    focus: str = ""
    active_goals: list[str] = []
    active_concepts: list[str] = []
    relevant_memories: list[dict] = []
    emotional_tone: str = "neutral"
    summary: str = ""
    system_prompt: str = ""


class GraphResponse(BaseModel):
    anchors: int = 0
    edges: int = 0
    ghosts: int = 0
    schemas: int = 0
    health_score: float = 0.0


# ── OpenAI-compatible chat ─────────────────────────────────

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat endpoint with optional memory injection.

    When inject_memory=True (default), the last user message is used to retrieve
    relevant cognitive context, which is prepended as a system message.
    """
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")

    messages = req.messages
    if req.inject_memory:
        user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if user_msg:
            frame = cortex.context(user_msg)
            system_content = frame.to_system_prompt()
            # Prepend cognitive context as system message
            messages = [ChatMessage(role="system", content=system_content)] + list(req.messages)

    # If a real LLM is configured, forward there
    try:
        from nwc.config.config import get_config
        from nwc.llm import get_llm
        cfg = get_config()
        if cfg.llm.api_key:
            llm = get_llm(cfg.llm.provider, cfg.llm.model, cfg.llm.api_key, cfg.llm.base_url)
            from nwc.llm.base import ChatMessage as BaseMsg
            llm_messages = [BaseMsg(role=m.role, content=m.content) for m in messages]
            resp = llm.chat_sync(llm_messages)
            return ChatResponse(
                choices=[ChatChoice(message=ChatMessage(role="assistant", content=resp.content))],
                model=cfg.llm.model,
            )
    except Exception:
        pass

    # Fallback: return memory-augmented context as response
    frame = cortex.context(req.messages[-1].content if req.messages else "")
    return ChatResponse(
        choices=[ChatChoice(message=ChatMessage(
            role="assistant",
            content=f"[NWC Cognitive Context]\n\n{frame.to_system_prompt()}\n\n"
                    f"[Note: Configure an LLM provider via 'nwc init' for full chat capabilities. "
                    f"This is the cognitive context that would be injected.]"
        ))],
    )


# ── Memory endpoints ───────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")
    s = cortex.stats()
    return {"status": "healthy", **s}


@app.post("/memory/write", response_model=MemoryWriteResponse)
async def memory_write(req: MemoryWriteRequest):
    """Store a memory."""
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")
    anchor_id = cortex.remember(req.text, tags=req.tags, importance=req.importance)
    return MemoryWriteResponse(id=anchor_id)


@app.post("/memory/query", response_model=MemoryQueryResponse)
async def memory_query(req: MemoryQueryRequest):
    """Retrieve memories by semantic query."""
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")
    result = cortex.recall(req.query, max_items=req.max_items, context=req.context)
    return MemoryQueryResponse(
        memory=[MemoryItem(**m) for m in result.memory],
        entities=result.entities,
        relations=result.relations,
        summary=result.summary,
    )


@app.post("/memory/context", response_model=ContextResponse)
async def memory_context(req: ContextRequest):
    """Get compressed cognitive context for LLM injection."""
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")
    frame = cortex.context(req.prompt)
    return ContextResponse(
        focus=frame.focus,
        active_goals=frame.active_goals,
        active_concepts=frame.active_concepts,
        relevant_memories=frame.relevant_memories,
        emotional_tone=frame.emotional_tone,
        summary=frame.summary,
        system_prompt=frame.to_system_prompt(),
    )


@app.get("/graph", response_model=GraphResponse)
async def graph():
    """Get memory graph overview."""
    if not cortex:
        raise HTTPException(503, "Cortex not initialized")
    s = cortex.stats()
    return GraphResponse(
        anchors=s.get("anchors", s.get("anchor_count", 0)),
        edges=s.get("edges", s.get("edge_count", 0)),
        ghosts=s.get("ghosts", s.get("ghost_count", 0)),
        schemas=s.get("schemas", s.get("schema_count", 0)),
        health_score=s.get("health_score", s.get("cognitive_health", 0)),
    )


def entry():
    import uvicorn
    from nwc.config.config import get_config
    cfg = get_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


if __name__ == "__main__":
    entry()
