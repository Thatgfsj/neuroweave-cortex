"""用户中心记忆链（User-Center Memory Chain）—— 四维接近度参考实现。

Issue #11 的可测试参考实现，独立、低耦合：不导入检索/运行时模块，不改变
``retrieval_core`` / ``cortex_api.runtime`` 的默认检索路径。规范见
``docs/spec/user-center-memory-chain.md``。

四维：``text`` 文本（embedding 余弦 / token Jaccard）、``tech`` 技术（tags +
cortex_path 段 + 关键词的加权 Jaccard）、``event`` 事件（时间衰减 + 同会话 +
事件标签）、``center`` 用户中心（与挂载记忆的最大加权文本/技术相似度）。

用法::

    from star_graph import Anchor, UserCenter, UserCenterConfig, build_memory_chain

    center = UserCenter(UserCenterConfig(capacity=16))
    center.mount(Anchor.create("用户常用 Python + pytest 工作流", tags=["python"]))
    chain = build_memory_chain(seed, candidates, center=center, query="改 pytest 用例")
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .math_utils import clamp, cosine_sim

# MemoryChain.stop_reason 的取值
STOP_CANDIDATES_EXHAUSTED = "candidates_exhausted"
STOP_SIMILARITY_BELOW_THRESHOLD = "similarity_below_threshold"
STOP_CENTER_BELOW_THRESHOLD = "center_below_threshold"
STOP_MAX_LENGTH = "max_chain_length"

#: 技术关键词表 —— 默认值与 defaults.yaml 的 user_center.tech_keywords 一致。
DEFAULT_TECH_KEYWORDS: tuple[str, ...] = (
    "python", "javascript", "typescript", "rust", "golang", "java", "kotlin",
    "c++", "c#", "sql", "bash", "docker", "kubernetes", "git", "linux",
    "numpy", "pandas", "pytorch", "tensorflow", "fastapi", "django", "flask",
    "react", "vue", "node", "pytest", "redis", "postgres", "mysql", "mongodb",
    "llm", "transformer", "embedding", "rag", "向量", "数据库", "前端", "后端",
)

_EVENT_TAG_PREFIXES: tuple[str, ...] = ("event:", "event/", "evt:", "事件:")
_TOKEN_RE = re.compile(r"[a-z0-9_+#]+|[\u4e00-\u9fff]")
_CORTEX_SPLIT_RE = re.compile(r"[./>:\s]+")

# (字段名, defaults.yaml 路径, 类型)
_CONFIG_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("capacity", "capacity", "int"),
    ("mount_weight_default", "mount_weight_default", "float"),
    ("w_text", "weights.text", "float"),
    ("w_tech", "weights.tech", "float"),
    ("w_event", "weights.event", "float"),
    ("w_center", "weights.center", "float"),
    ("task_center_weight", "center_split.task_weight", "float"),
    ("tech_center_weight", "center_split.tech_weight", "float"),
    ("chain_similarity_threshold", "thresholds.chain_similarity", "float"),
    ("stop_center_threshold", "thresholds.stop_center", "float"),
    ("max_chain_length", "max_chain_length", "int"),
    ("chain_gate", "chain_gate", "str"),
    ("eviction_half_life_hours", "eviction_half_life_hours", "float"),
    ("event_time_weight", "event.time_weight", "float"),
    ("event_session_weight", "event.session_weight", "float"),
    ("event_tag_weight", "event.tag_weight", "float"),
    ("event_half_life_days", "event.half_life_days", "float"),
    ("tech_tag_weight", "tech.tag_weight", "float"),
    ("tech_path_weight", "tech.path_weight", "float"),
    ("tech_keyword_weight", "tech.keyword_weight", "float"),
)


@dataclass
class UserCenterConfig:
    """用户中心与记忆链的全部可调参数（默认值与 defaults.yaml 同源）。

    用 :meth:`from_config` 读取 Config 覆盖值；构造后仍可逐字段修改（如运行时
    调整 ``capacity``）。越界值在 ``__post_init__`` 夹紧，非法 ``chain_gate``
    回退为 ``"combined"``。
    """

    capacity: int = 32
    mount_weight_default: float = 1.0
    w_text: float = 0.40
    w_tech: float = 0.25
    w_event: float = 0.15
    w_center: float = 0.20
    # 第 4 维内部分层（用户确认 2026-11-04）：本次任务中心为主，长期技术中心为辅。
    # 记忆每次任务只调用一次，因此"本次任务/回复中心"占 85%，"技术中心"占 15%。
    task_center_weight: float = 0.85
    tech_center_weight: float = 0.15
    chain_similarity_threshold: float = 0.70
    stop_center_threshold: float = 0.35
    max_chain_length: int = 12
    chain_gate: str = "combined"
    eviction_half_life_hours: float = 168.0
    event_time_weight: float = 0.60
    event_session_weight: float = 0.25
    event_tag_weight: float = 0.15
    event_half_life_days: float = 7.0
    tech_tag_weight: float = 1.0
    tech_path_weight: float = 0.80
    tech_keyword_weight: float = 0.60
    tech_keywords: tuple[str, ...] = DEFAULT_TECH_KEYWORDS

    def __post_init__(self) -> None:
        self.capacity = max(0, int(self.capacity))
        self.max_chain_length = max(0, int(self.max_chain_length))
        self.mount_weight_default = max(0.0, float(self.mount_weight_default))
        for name in ("w_text", "w_tech", "w_event", "w_center",
                     "task_center_weight", "tech_center_weight"):
            setattr(self, name, max(0.0, float(getattr(self, name))))
        self.chain_similarity_threshold = clamp(float(self.chain_similarity_threshold))
        self.stop_center_threshold = clamp(float(self.stop_center_threshold))
        self.chain_gate = self.chain_gate if self.chain_gate in ("combined", "text") else "combined"
        self.eviction_half_life_hours = max(1e-6, float(self.eviction_half_life_hours))
        for name in ("event_time_weight", "event_session_weight", "event_tag_weight"):
            setattr(self, name, max(0.0, float(getattr(self, name))))
        self.event_half_life_days = max(1e-6, float(self.event_half_life_days))
        for name in ("tech_tag_weight", "tech_path_weight", "tech_keyword_weight"):
            setattr(self, name, max(0.0, float(getattr(self, name))))
        if isinstance(self.tech_keywords, str):
            self.tech_keywords = (self.tech_keywords,)
        else:
            self.tech_keywords = tuple(str(k) for k in self.tech_keywords if str(k).strip())

    @classmethod
    def from_config(cls, source: object = None, **overrides: Any) -> "UserCenterConfig":
        """从 Config 的 ``user_center`` 段读取覆盖值，缺省回退 dataclass 默认值。

        ``source`` 省略时用全局 ``Config.get()``；也接受 ``Mapping`` 与 ``None``。
        """
        section = _config_section(source)
        kwargs: dict[str, Any] = {}
        for name, path, kind in _CONFIG_FIELDS:
            raw = _lookup(section, path)
            if raw is None or isinstance(raw, bool):
                continue
            if kind == "int" and isinstance(raw, int):
                kwargs[name] = int(raw)
            elif kind == "float" and isinstance(raw, (int, float)):
                kwargs[name] = float(raw)
            elif kind == "str" and isinstance(raw, str):
                kwargs[name] = raw
        keywords = _lookup(section, "tech_keywords")
        if isinstance(keywords, (list, tuple)):
            kwargs["tech_keywords"] = tuple(str(k) for k in keywords if str(k).strip())
        kwargs.update(overrides)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        """序列化为与 defaults.yaml 同构的嵌套字典。"""
        return {
            "capacity": self.capacity, "mount_weight_default": self.mount_weight_default,
            "weights": {"text": self.w_text, "tech": self.w_tech,
                        "event": self.w_event, "center": self.w_center},
            "center_split": {"task_weight": self.task_center_weight,
                             "tech_weight": self.tech_center_weight},
            "thresholds": {"chain_similarity": self.chain_similarity_threshold,
                           "stop_center": self.stop_center_threshold},
            "max_chain_length": self.max_chain_length,
            "chain_gate": self.chain_gate,
            "eviction_half_life_hours": self.eviction_half_life_hours,
            "event": {"time_weight": self.event_time_weight,
                      "session_weight": self.event_session_weight,
                      "tag_weight": self.event_tag_weight,
                      "half_life_days": self.event_half_life_days},
            "tech": {"tag_weight": self.tech_tag_weight,
                     "path_weight": self.tech_path_weight,
                     "keyword_weight": self.tech_keyword_weight},
            "tech_keywords": list(self.tech_keywords),
        }

@dataclass
class MountedMemory:
    """用户中心上的一条挂载记录（锚点 + 权重 + 时间戳）。"""

    anchor: Any
    anchor_id: str
    weight: float
    mounted_at: float
    last_used: float
    seq: int


class UserCenter:
    """用户中心：挂载最常用记忆（如"常用 Python"），容量可调。

    挂载打分（淘汰与 ``top`` 排序）= ``weight × 0.5 ** (闲置时长 / 半衰期)``；
    同分时先淘汰更久未使用者，再按 id 字典序，保证确定性。持久化不在本版范围。
    """

    def __init__(self, config: UserCenterConfig | None = None, *,
                 clock: Callable[[], float] | None = None) -> None:
        self.config = config if config is not None else UserCenterConfig.from_config()
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._mounts: dict[str, MountedMemory] = {}
        self._seq = 0

    @property
    def capacity(self) -> int:
        """挂载容量上限；赋值即生效，缩小容量会立即淘汰弱者。"""
        return self.config.capacity

    @capacity.setter
    def capacity(self, value: int) -> None:
        self.config.capacity = max(0, int(value))
        while len(self._mounts) > self.config.capacity:
            self._evict_weakest(self._clock())

    def __contains__(self, anchor: object) -> bool:
        return self.is_mounted(anchor)

    def mount(self, anchor: Any, weight: float | None = None) -> bool:
        """挂载一条记忆；已挂载则更新权重与使用时间。容量为 0 时返回 False。"""
        anchor_id = _anchor_id(anchor)
        if self.config.capacity <= 0:
            return False
        weight = self.config.mount_weight_default if weight is None else max(0.0, float(weight))
        now = self._clock()
        existing = self._mounts.get(anchor_id)
        if existing is not None:
            existing.anchor, existing.weight, existing.last_used = anchor, weight, now
            return True
        while len(self._mounts) >= self.config.capacity:
            self._evict_weakest(now)
        self._seq += 1
        self._mounts[anchor_id] = MountedMemory(
            anchor=anchor, anchor_id=anchor_id, weight=weight,
            mounted_at=now, last_used=now, seq=self._seq)
        return True

    def unmount(self, anchor: Any) -> bool:
        """卸载一条记忆（接受 id 或锚点对象）；不存在返回 False。"""
        try:
            return self._mounts.pop(_anchor_id(anchor), None) is not None
        except ValueError:
            return False

    def is_mounted(self, anchor: Any) -> bool:
        """判断某条记忆是否挂在用户中心上。"""
        try:
            return _anchor_id(anchor) in self._mounts
        except ValueError:
            return False

    @property
    def mounted_ids(self) -> tuple[str, ...]:
        """已挂载锚点 id（挂载顺序）。"""
        return tuple(self._mounts)

    def mounted_items(self) -> tuple[tuple[Any, float], ...]:
        """(锚点, 权重) 序列，供接近度函数使用。"""
        return tuple((m.anchor, m.weight) for m in self._mounts.values())

    def weight_of(self, anchor: Any) -> float | None:
        """某条挂载记忆的权重；未挂载返回 None。"""
        try:
            mounted = self._mounts.get(_anchor_id(anchor))
        except ValueError:
            return None
        return None if mounted is None else mounted.weight

    def top(self, n: int) -> list[Any]:
        """按挂载打分返回前 ``n`` 条记忆（锚点对象）。"""
        now = self._clock()
        ranked = sorted(self._mounts.values(),
                        key=lambda m: (-self._rank(m, now), m.seq, m.anchor_id))
        return [m.anchor for m in ranked[: max(0, int(n))]]

    def center_embedding(self, normalize: bool = True) -> list[float] | None:
        """挂载记忆 embedding 的加权平均；维度不一致者跳过。

        默认归一化到单位长度。无可用向量或加权平均为零向量时返回 None。
        """
        total, acc = 0.0, None
        for mounted in self._mounts.values():
            embedding = _embedding(mounted.anchor)
            if not embedding or mounted.weight <= 0:
                continue
            if acc is None:
                acc = [0.0] * len(embedding)
            elif len(embedding) != len(acc):
                continue
            for index, value in enumerate(embedding):
                acc[index] += mounted.weight * value
            total += mounted.weight
        if acc is None or total <= 0:
            return None
        vector = [value / total for value in acc]
        if normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm < 1e-12:
                return None
            vector = [value / norm for value in vector]
        return vector

    def _rank(self, mounted: MountedMemory, now: float) -> float:
        age = max(0.0, now - mounted.last_used)
        return mounted.weight * (0.5 ** (age / (self.config.eviction_half_life_hours * 3600.0)))

    def _evict_weakest(self, now: float) -> None:
        if self._mounts:
            victim = min(self._mounts.values(),
                         key=lambda m: (self._rank(m, now), m.last_used, m.anchor_id))
            self._mounts.pop(victim.anchor_id, None)



# ── 四维接近度 ──────────────────────────────────────────────────────────
def text_proximity(a: Any, b: Any, *, config: UserCenterConfig | None = None) -> float:
    """第 1 维：文本接近度 ∈ [0, 1]。

    同维 embedding 取余弦并截断到 [0, 1]；否则退化为文本 token Jaccard。
    空文本、维度不一致、零向量都不抛错。
    """
    _ = config  # 统一签名；文本维无额外旋钮
    embedding_a, embedding_b = _embedding(a), _embedding(b)
    if embedding_a and embedding_b and len(embedding_a) == len(embedding_b):
        return clamp(cosine_sim(embedding_a, embedding_b))
    tokens_a, tokens_b = _tokens(_text(a)), _tokens(_text(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def tech_proximity(a: Any, b: Any, *, config: UserCenterConfig | None = None) -> float:
    """第 2 维：技术接近度 ∈ [0, 1]（加权 Jaccard：Σmin/Σmax）。"""
    cfg = _resolve_config(config)
    signature_a, signature_b = _tech_signature(a, cfg), _tech_signature(b, cfg)
    if not signature_a or not signature_b:
        return 0.0
    # 排序后求和，避免 set 迭代顺序造成跨进程浮点累加次序不同。
    common = sum(min(signature_a[k], signature_b[k])
                 for k in sorted(signature_a.keys() & signature_b.keys()))
    union = sum(max(signature_a[k], signature_b.get(k, 0.0)) for k in sorted(signature_a))
    union += sum(signature_b[k] for k in sorted(signature_b.keys() - signature_a.keys()))
    return clamp(common / union) if union > 0 else 0.0


def event_proximity(a: Any, b: Any, *, config: UserCenterConfig | None = None) -> float:
    """第 3 维：事件接近度 ∈ [0, 1]。

    ``(w_t·时间衰减 + w_s·同会话 + w_g·事件标签 Jaccard) / (w_t+w_s+w_g)``，
    时间衰减 = ``0.5 ** (相差天数 / half_life_days)``。
    """
    cfg = _resolve_config(config)
    denominator = cfg.event_time_weight + cfg.event_session_weight + cfg.event_tag_weight
    if denominator <= 0:
        return 0.0
    created_a, created_b = _created_at(a), _created_at(b)
    if created_a > 0 and created_b > 0:
        delta_days = abs(created_a - created_b) / 86400.0
        temporal = 0.5 ** (delta_days / cfg.event_half_life_days)
    else:
        temporal = 0.0
    session_a, session_b = _session(a), _session(b)
    same_session = 1.0 if session_a and session_a == session_b else 0.0
    tags_a, tags_b = _event_tags(a), _event_tags(b)
    tag_similarity = len(tags_a & tags_b) / len(tags_a | tags_b) if tags_a and tags_b else 0.0
    score = (cfg.event_time_weight * temporal + cfg.event_session_weight * same_session
             + cfg.event_tag_weight * tag_similarity)
    return clamp(score / denominator)


def _task_center_similarity(a: Any, query: Any, cfg: UserCenterConfig,
                            task_anchors: Iterable[Any] | None = None) -> float | None:
    """本次任务中心相似度；无 query 且无任务锚点时返回 None。

    ``task_anchors`` 支持锚点或 ``(anchor, weight)`` 元组；权重按该分量的最大值
    相对归一（与长期技术中心一致），权重 ≤0 的条目跳过。入口先物化为 tuple，
    避免生成器被多次消费后耗尽。
    """
    scores: list[float] = []
    query_view = _query_view(query)
    if query_view is not None:
        scores.append(_text_tech_similarity(a, query_view, cfg))
    anchors = tuple(task_anchors or ())
    weights: list[float] = []
    for item in anchors:
        if isinstance(item, tuple) and len(item) == 2:
            weights.append(max(0.0, float(item[1])))
        else:
            weights.append(1.0)
    max_weight = max(weights) if weights else 0.0
    if max_weight > 0:
        for item, weight in zip(anchors, weights):
            if weight <= 0:
                continue
            anchor = item[0] if isinstance(item, tuple) and len(item) == 2 else item
            relative = weight / max_weight
            scores.append(relative * _text_tech_similarity(a, anchor, cfg))
    return max(scores) if scores else None


def _tech_center_similarity(a: Any, center: Any, cfg: UserCenterConfig) -> float | None:
    """长期技术中心相似度；无挂载或权重全 0 时返回 None。"""
    mounts = _center_items(center)
    if not mounts:
        return None
    max_weight = max((weight for _, weight in mounts), default=0.0)
    if max_weight <= 0:
        return None
    best = 0.0
    for anchor, weight in mounts:
        relative = max(0.0, weight) / max_weight
        if relative > 0:
            best = max(best, relative * _text_tech_similarity(a, anchor, cfg))
    return clamp(best)


def user_center_proximity(a: Any, query: Any = None, center: Any = None, *,
                          task_anchors: Iterable[Any] | None = None,
                          config: UserCenterConfig | None = None) -> float:
    """第 4 维：与用户中心的接近度 ∈ [0, 1]。

    用户中心分两层并融合（用户确认 2026-11-04）：

    * **本次任务中心（默认 85%）** —— 与 ``query``（文本/向量/锚点）以及可选的
      ``task_anchors``（keyword-only；本次任务内一次性挂载的记忆，支持
      ``(anchor, weight)`` 且权重按分量内最大值相对归一）的文本+技术相似度；
    * **长期技术中心（默认 15%）** —— 与 ``center`` 挂载记忆的相对权重相似度，
      事件维不参与，避免时间衰减惩罚长期偏好。

    融合公式为 ``(0.85·task + 0.15·tech) / (0.85 + 0.15)``。任一分量不可用时
    按剩余分量归一（不乘 0.85/0.15）；两者都不可用返回 0.0。记忆每次任务只调用
    一次，因此任务中心为主、技术中心为辅。
    """
    cfg = _resolve_config(config, center)
    task_score = _task_center_similarity(a, query, cfg, task_anchors)
    tech_score = _tech_center_similarity(a, center, cfg)
    if task_score is None and tech_score is None:
        return 0.0
    if task_score is None:
        return clamp(tech_score)
    if tech_score is None:
        return clamp(task_score)
    total = cfg.task_center_weight + cfg.tech_center_weight
    if total <= 0:
        return 0.0
    fused = (cfg.task_center_weight * task_score
             + cfg.tech_center_weight * tech_score) / total
    return clamp(fused)


def combined_proximity(a: Any, b: Any, center: Any = None, query: Any = None, *,
                       task_anchors: Iterable[Any] | None = None,
                       config: UserCenterConfig | None = None) -> float:
    """四维加权接近度 ∈ [0, 1]（按权重和归一化；权重全 0 返回 0.0）。"""
    cfg = _resolve_config(config, center)
    weights = (cfg.w_text, cfg.w_tech, cfg.w_event, cfg.w_center)
    if sum(weights) <= 0:
        return 0.0
    score = (cfg.w_text * text_proximity(a, b, config=cfg)
             + cfg.w_tech * tech_proximity(a, b, config=cfg)
             + cfg.w_event * event_proximity(a, b, config=cfg)
             + cfg.w_center * user_center_proximity(
                 a, query=query, center=center,
                 task_anchors=task_anchors, config=cfg))
    return clamp(score / sum(weights))


# ── 记忆链 ──────────────────────────────────────────────────────────────
@dataclass
class ChainStep:
    """记忆链上的一步：锚点 + 与上一步的四维分数。"""

    index: int
    anchor: Any
    anchor_id: str
    previous_id: str
    text_score: float
    tech_score: float
    event_score: float
    center_score: float
    combined_score: float

    def scores(self) -> dict[str, float]:
        """四维分数 + 综合分（便于日志与断言）。"""
        return {"text": self.text_score, "tech": self.tech_score,
                "event": self.event_score, "center": self.center_score,
                "combined": self.combined_score}

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典（不含锚点正文）。"""
        return {"index": self.index, "id": self.anchor_id,
                "previous_id": self.previous_id, "scores": self.scores()}


@dataclass
class MemoryChain:
    """有序记忆链：``seed``（起点）+ ``steps``（逐步追加）+ 停止原因。"""

    seed: Any = None
    steps: list[ChainStep] = field(default_factory=list)
    stop_reason: str = STOP_CANDIDATES_EXHAUSTED
    config: UserCenterConfig = field(default_factory=UserCenterConfig)

    @property
    def seed_id(self) -> str | None:
        """种子锚点 id；无种子返回 None。"""
        return _safe_id(self.seed)

    @property
    def ids(self) -> list[str]:
        """链上各步的锚点 id（不含种子，按扩展顺序）。"""
        return [step.anchor_id for step in self.steps]

    def node_ids(self) -> list[str]:
        """含种子在内的完整节点 id 序列。"""
        seed_id = self.seed_id
        return ([seed_id] if seed_id else []) + self.ids

    def to_dict(self) -> dict[str, Any]:
        """序列化整条链（含配置快照与停止原因）。"""
        return {"seed_id": self.seed_id, "stop_reason": self.stop_reason,
                "steps": [step.to_dict() for step in self.steps],
                "config": self.config.as_dict()}


def build_memory_chain(seed: Any = None, candidates: Iterable[Any] = (), center: Any = None,
                       query: Any = None, config: UserCenterConfig | None = None, *,
                       task_anchors: Iterable[Any] | None = None) -> MemoryChain:
    """以 seed（或用户中心 / query）起步，逐步扩展记忆链直到停止。

    停止条件（按检查优先级）：

    1. 与当前节点相似度 < ``chain_similarity_threshold``；
    2. 与用户中心（第 4 维）关联度 < ``stop_center_threshold``（x）；
    3. 节点数（含种子）达到 ``max_chain_length``；
    4. 候选耗尽（正常收敛）。

    第 4 维使用"本次任务中心 85% + 长期技术中心 15%"的融合分（用户确认
    2026-11-04）：每次任务只调用一次记忆，因此本次任务为主；``task_anchors``
    （keyword-only）可传入本次任务内的一次性记忆，支持 ``(anchor, weight)`` 元组，
    权重按分量内最大值相对归一。完全确定：候选按传入顺序遍历，同分取前者；
    不使用随机数与时钟。空候选、空中心、无 embedding 均能运行（逐维降级）。
    """
    cfg = _resolve_config(config, center)
    task_anchors = tuple(task_anchors or ())
    chain = MemoryChain(seed=seed, stop_reason=STOP_CANDIDATES_EXHAUSTED, config=cfg)
    previous_id = _safe_id(seed) or ""
    remaining = _dedupe_candidates(candidates, exclude=previous_id)
    current = seed if seed is not None else _query_view(query)
    center_gate = cfg.w_center > 0 and (
        bool(_center_items(center)) or _query_view(query) is not None or bool(task_anchors))

    while True:
        if len(chain.steps) + (1 if seed is not None else 0) >= cfg.max_chain_length:
            chain.stop_reason = STOP_MAX_LENGTH
            break
        if not remaining:
            chain.stop_reason = STOP_CANDIDATES_EXHAUSTED
            break
        index, best = _pick_candidate(remaining, current, center, query, task_anchors, cfg)
        link_score = best["text"] if cfg.chain_gate == "text" else best["combined"]
        if link_score < cfg.chain_similarity_threshold:
            chain.stop_reason = STOP_SIMILARITY_BELOW_THRESHOLD
            break
        if center_gate and best["center"] < cfg.stop_center_threshold:
            chain.stop_reason = STOP_CENTER_BELOW_THRESHOLD
            break
        anchor = remaining.pop(index)
        anchor_id = _safe_id(anchor) or ""
        chain.steps.append(ChainStep(
            index=len(chain.steps) + 1, anchor=anchor, anchor_id=anchor_id,
            previous_id=previous_id, text_score=best["text"], tech_score=best["tech"],
            event_score=best["event"], center_score=best["center"],
            combined_score=best["combined"]))
        current, previous_id = anchor, anchor_id
    return chain


# ── 内部工具 ────────────────────────────────────────────────────────────
def _resolve_config(config: UserCenterConfig | None = None,
                    center: Any = None) -> UserCenterConfig:
    if config is not None:
        return config
    if isinstance(center, UserCenter):
        return center.config
    return UserCenterConfig.from_config()


def _lookup(section: Any, path: str) -> Any:
    current = section
    for part in path.split("."):
        current = None if current is None else _field(current, part)
    return current


def _config_section(source: object = None) -> Any:
    """把 Config / _DotDict / Mapping / None 统一成 ``user_center`` 段。"""
    if source is None:
        from .config import Config
        source = Config.get()
    if isinstance(source, Mapping):
        section = source.get("user_center", source)
        return section if isinstance(section, Mapping) else None
    section = getattr(source, "get_path", lambda *_: None)("user_center", None)
    if section is None and callable(getattr(source, "to_dict", None)):
        data = source.to_dict()
        section = data.get("user_center") if isinstance(data, Mapping) else None
    if section is None:
        section = getattr(source, "user_center", None)
    return section


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """duck-typed 读取：Mapping 用 get，其余用 getattr。"""
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _anchor_id(anchor: Any) -> str:
    if isinstance(anchor, str):
        if anchor:
            return anchor
        raise ValueError("anchor id must be a non-empty string")
    value = _field(anchor, "id", None)
    if value is None or str(value) == "":
        raise ValueError("anchor has no usable 'id' attribute")
    return str(value)


def _safe_id(anchor: Any) -> str | None:
    try:
        return _anchor_id(anchor)
    except ValueError:
        return None


def _text(anchor: Any) -> str:
    value = _field(anchor, "text", "")
    return value if isinstance(value, str) else str(value or "")


def _embedding(anchor: Any) -> list[float] | None:
    raw = _field(anchor, "embedding", None)
    if raw is None:
        return None
    try:
        vector = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    return vector or None


def _tags(anchor: Any) -> tuple[str, ...]:
    raw = _field(anchor, "tags", ()) or ()
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(tag).strip().lower() for tag in raw if str(tag).strip())


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _created_at(anchor: Any) -> float:
    try:
        return float(_field(anchor, "created_at", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _session(anchor: Any) -> str:
    return str(_field(anchor, "source_session", "") or "").strip()


def _event_tags(anchor: Any) -> set[str]:
    return {tag for tag in _tags(anchor) if tag.startswith(_EVENT_TAG_PREFIXES)}


def _keyword_present(keyword: str, tokens: set[str], haystack: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if keyword in tokens:
        return True
    if " " in keyword or any(ord(char) > 127 for char in keyword):
        return keyword in haystack
    return False


def _tech_signature(anchor: Any, cfg: UserCenterConfig) -> dict[str, float]:
    signature: dict[str, float] = {}
    segments = [part for part in
                _CORTEX_SPLIT_RE.split(str(_field(anchor, "cortex_path", "") or "").lower())
                if part]
    for tag in _tags(anchor):
        signature[f"tag:{tag}"] = max(signature.get(f"tag:{tag}", 0.0), cfg.tech_tag_weight)
    for segment in segments:
        key = f"path:{segment}"
        signature[key] = max(signature.get(key, 0.0), cfg.tech_path_weight)
    if cfg.tech_keywords:
        haystack = " ".join([_text(anchor).lower(), *_tags(anchor), *segments])
        tokens = _tokens(haystack)
        for keyword in cfg.tech_keywords:
            if _keyword_present(keyword, tokens, haystack):
                key = f"kw:{keyword.strip().lower()}"
                signature[key] = max(signature.get(key, 0.0), cfg.tech_keyword_weight)
    return {key: weight for key, weight in signature.items() if weight > 0}


def _text_tech_similarity(a: Any, b: Any, cfg: UserCenterConfig) -> float:
    denominator = cfg.w_text + cfg.w_tech
    if denominator <= 0:
        return 0.0
    score = (cfg.w_text * text_proximity(a, b, config=cfg)
             + cfg.w_tech * tech_proximity(a, b, config=cfg))
    return clamp(score / denominator)


def _center_items(center: Any) -> tuple[tuple[Any, float], ...]:
    if center is None:
        return ()
    if isinstance(center, UserCenter):
        return center.mounted_items()
    mounted_items = getattr(center, "mounted_items", None)
    if callable(mounted_items):
        return tuple(mounted_items())
    if isinstance(center, Mapping):
        iterable: Iterable[Any] = center.values()
    elif isinstance(center, Sequence) and not isinstance(center, (str, bytes)):
        iterable = center
    else:
        return ()
    items: list[tuple[Any, float]] = []
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            items.append((item[0], float(item[1])))
        else:
            items.append((item, 1.0))
    return tuple(items)


def _query_view(query: Any) -> Any:
    """把 str / embedding / 锚点统一成可被 duck-typed 读取的视图。"""
    if query is None:
        return None
    if isinstance(query, str):
        return {"text": query}
    if isinstance(query, (list, tuple)) and query and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in query):
        return {"embedding": [float(value) for value in query]}
    if isinstance(query, Mapping) or hasattr(query, "text") or hasattr(query, "embedding"):
        return query
    return None


def _dedupe_candidates(candidates: Iterable[Any], exclude: str = "") -> list[Any]:
    ordered: list[Any] = []
    seen: set[str] = {exclude} if exclude else set()
    for candidate in candidates or ():
        anchor_id = _safe_id(candidate)
        if anchor_id is None or anchor_id in seen:
            continue
        seen.add(anchor_id)
        ordered.append(candidate)
    return ordered


def _score_link(anchor: Any, current: Any, center: Any, query: Any,
                task_anchors: Iterable[Any] | None,
                cfg: UserCenterConfig) -> dict[str, float]:
    if current is None:
        text = tech = event = 0.0
    else:
        text = text_proximity(anchor, current, config=cfg)
        tech = tech_proximity(anchor, current, config=cfg)
        event = event_proximity(anchor, current, config=cfg)
    center_score = user_center_proximity(
        anchor, query=query, center=center, task_anchors=task_anchors, config=cfg)
    denominator = cfg.w_text + cfg.w_tech + cfg.w_event + cfg.w_center
    combined = clamp((cfg.w_text * text + cfg.w_tech * tech
                      + cfg.w_event * event + cfg.w_center * center_score) / denominator
                     ) if denominator > 0 else 0.0
    return {"text": text, "tech": tech, "event": event,
            "center": center_score, "combined": combined}


def _pick_candidate(remaining: list[Any], current: Any, center: Any, query: Any,
                    task_anchors: Iterable[Any] | None,
                    cfg: UserCenterConfig) -> tuple[int, dict[str, float]]:
    best_index = 0
    best_scores = _score_link(remaining[0], current, center, query, task_anchors, cfg)
    for index in range(1, len(remaining)):
        scores = _score_link(remaining[index], current, center, query, task_anchors, cfg)
        if (scores["combined"], scores["text"]) > (best_scores["combined"], best_scores["text"]):
            best_index, best_scores = index, scores
    return best_index, best_scores
