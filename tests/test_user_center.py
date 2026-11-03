"""用户中心记忆链（四维接近度）测试 —— Issue #11。

覆盖容量/淘汰、权重可配置、四维边界与单调性、第 4 维 85/15 融合与单分量归一、
链增长与三类停止条件、空中心回退、无 embedding 降级、确定性、配置读取与序列化。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from star_graph import MemoryChain, UserCenter, UserCenterConfig, build_memory_chain
from star_graph.config import Config
from star_graph.memory_core.anchor import Anchor
from star_graph.user_center import (
    STOP_CANDIDATES_EXHAUSTED,
    STOP_CENTER_BELOW_THRESHOLD,
    STOP_MAX_LENGTH,
    STOP_SIMILARITY_BELOW_THRESHOLD,
    combined_proximity,
    event_proximity,
    tech_proximity,
    text_proximity,
    user_center_proximity,
)


def make_anchor(anchor_id: str, text: str, *, embedding: list[float] | None = None,
                tags: tuple[str, ...] = (), session: str = "",
                created_at: float = 0.0, cortex_path: str = "") -> Anchor:
    """直接构造 Anchor（不经 embedder 注册表，快速且确定）。"""
    return Anchor(id=anchor_id, text=text, embedding=embedding, tags=list(tags),
                  source_session=session, created_at=created_at,
                  last_activated_at=created_at, cortex_path=cortex_path)


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestUserCenterCapacity:
    def test_default_capacity_and_runtime_resize(self) -> None:
        center = UserCenter()
        assert center.capacity == 32
        for index in range(5):
            assert center.mount(make_anchor(f"a{index}", f"m{index}"))
        center.capacity = 2
        assert len(center.mounted_ids) == 2 and center.config.capacity == 2

    def test_zero_capacity_and_mount_state(self) -> None:
        assert UserCenter(UserCenterConfig(capacity=0)).mount(make_anchor("a", "x")) is False
        center = UserCenter(UserCenterConfig(capacity=2))
        anchor = make_anchor("a", "hello", embedding=[1.0, 0.0])
        assert center.is_mounted("a") is False and "a" not in center
        assert center.mount(anchor) is True and center.mount(anchor, weight=0.5) is True
        assert center.weight_of("a") == 0.5 and center.is_mounted("a") is True
        assert center.unmount("a") is True and center.unmount("a") is False

    def test_eviction_by_weight_and_recency(self) -> None:
        clock = FakeClock()
        center = UserCenter(UserCenterConfig(capacity=2), clock=clock)
        assert center.mount(make_anchor("low", "l"), weight=0.1)
        assert center.mount(make_anchor("high", "h"), weight=1.0)
        assert center.mount(make_anchor("new", "n"), weight=0.9)
        assert center.mounted_ids == ("high", "new")
        clock.advance(3600.0)
        assert center.mount(make_anchor("newer", "n2"))
        clock.advance(3600.0)
        assert center.mount(make_anchor("fresh", "n3"))
        assert center.mounted_ids == ("newer", "fresh")

    def test_top_and_center_embedding(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=4), clock=FakeClock())
        center.mount(make_anchor("low", "l"), weight=0.2)
        center.mount(make_anchor("high", "h", embedding=[1.0, 0.0]), weight=3.0)
        center.mount(make_anchor("mid", "m", embedding=[0.0, 1.0]), weight=1.0)
        assert [anchor.id for anchor in center.top(2)] == ["high", "mid"]
        assert center.top(0) == []
        assert center.center_embedding(normalize=False) == pytest.approx([0.75, 0.25])
        normalized = center.center_embedding()
        assert normalized is not None and normalized[0] / normalized[1] == pytest.approx(3.0)
        assert UserCenter(UserCenterConfig(capacity=4)).center_embedding() is None
        center.mount(make_anchor("plain", "没有向量"))
        assert center.center_embedding(normalize=False) == pytest.approx([0.75, 0.25])


class TestConfig:
    def test_reads_defaults_yaml_and_overrides(self) -> None:
        raw = Config.defaults()
        assert raw.user_center.capacity == 32
        assert raw.get_path("user_center.weights.text") == 0.40
        config = UserCenterConfig.from_config(raw)
        assert config.capacity == 32 and config.w_text == pytest.approx(0.40)
        assert config.chain_similarity_threshold == pytest.approx(0.70)
        assert config.stop_center_threshold == pytest.approx(0.35)
        assert config.max_chain_length == 12 and "python" in config.tech_keywords
        assert raw.get_path("user_center.center_split.task_weight") == pytest.approx(0.85)
        assert raw.get_path("user_center.center_split.tech_weight") == pytest.approx(0.15)
        assert config.task_center_weight == pytest.approx(0.85)
        assert config.tech_center_weight == pytest.approx(0.15)
        assert config.as_dict()["center_split"] == {"task_weight": 0.85,
                                                    "tech_weight": 0.15}
        custom = UserCenterConfig.from_config(
            {"user_center": {"center_split": {"task_weight": 0.7, "tech_weight": 0.3}}})
        assert custom.task_center_weight == pytest.approx(0.7)
        assert custom.tech_center_weight == pytest.approx(0.3)
        assert UserCenterConfig.from_config(capacity=4, w_center=0.5).capacity == 4
        raw.override("user_center.capacity", 7)
        raw.override("user_center.weights.text", 0.55)
        patched = UserCenterConfig.from_config(raw)
        assert patched.capacity == 7 and patched.w_text == pytest.approx(0.55)

    def test_as_dict_roundtrip_and_clamps(self) -> None:
        config = UserCenterConfig(capacity=5, w_text=0.5, chain_gate="text")
        restored = UserCenterConfig.from_config({"user_center": config.as_dict()})
        assert restored.as_dict() == config.as_dict()
        broken = UserCenterConfig(capacity=-3, chain_similarity_threshold=2.0,
                                  chain_gate="bogus", w_text=-1.0)
        assert broken.capacity == 0 and broken.chain_similarity_threshold == 1.0
        assert broken.chain_gate == "combined" and broken.w_text == 0.0

    def test_weights_are_configurable(self) -> None:
        a = make_anchor("a", "alpha", embedding=[1.0, 0.0])
        b = make_anchor("b", "beta", embedding=[1.0, 0.0])
        text_only = UserCenterConfig(w_text=1.0, w_tech=0.0, w_event=0.0, w_center=0.0)
        assert combined_proximity(a, b, config=text_only) == pytest.approx(1.0)
        assert combined_proximity(a, b) == pytest.approx(0.40)


class TestFourDimensions:
    def test_text_proximity_bounds_and_fallback(self) -> None:
        same = make_anchor("a", "python pytest", embedding=[1.0, 0.0])
        assert text_proximity(same, same) == pytest.approx(1.0)
        orthogonal = make_anchor("b", "rust", embedding=[0.0, 1.0])
        assert text_proximity(same, orthogonal) == pytest.approx(0.0)
        assert text_proximity(make_anchor("c", ""), make_anchor("d", "")) == 0.0
        a = make_anchor("e", "python 测试 用例")
        # 中文按单字切分：{python, 测, 试, 用, 例} vs {python, 测, 试}
        assert text_proximity(a, make_anchor("f", "python 测试")) == pytest.approx(3 / 5)
        mismatch = make_anchor("g", "python 测试", embedding=[1.0, 0.0, 0.0])
        assert text_proximity(a, mismatch) == pytest.approx(3 / 5)
        base = make_anchor("h", "python pytest fixture")
        near = make_anchor("i", "python pytest")
        far = make_anchor("j", "python java")
        assert text_proximity(base, near) > text_proximity(base, far)

    def test_tech_proximity_tags_paths_keywords(self) -> None:
        tagged = make_anchor("a", "x", tags=("python",))
        assert tech_proximity(tagged, tagged) == pytest.approx(1.0)
        assert tech_proximity(tagged, make_anchor("b", "x", tags=("javascript",))) == 0.0
        assert tech_proximity(make_anchor("c", "用 Python 写服务"),
                              make_anchor("d", "python 服务脚本")) == pytest.approx(1.0)
        path_a = make_anchor("p1", "x", cortex_path="dev.python.gui")
        path_b = make_anchor("p2", "x", cortex_path="dev.python.cli")
        assert 0.0 < tech_proximity(path_a, path_b) < 1.0

    def test_event_proximity_decay_session_tags(self) -> None:
        base = make_anchor("a", "x", created_at=1_000_000.0, session="s1")
        near = make_anchor("b", "x", created_at=1_000_000.0 + 3600.0, session="s1")
        far = make_anchor("c", "x", created_at=1_000_000.0 + 30 * 86400.0, session="s2")
        assert event_proximity(base, near) > event_proximity(base, far)
        assert event_proximity(base, near) <= 1.0
        tagged = make_anchor("d", "x", created_at=1_000_000.0 + 30 * 86400.0,
                             session="s2", tags=("event:deploy",))
        assert event_proximity(tagged, tagged) > event_proximity(base, far)

    def test_user_center_proximity_weighted_and_fallback(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=4), clock=FakeClock())
        center.mount(make_anchor("py", "python", embedding=[1.0, 0.0]), weight=1.0)
        center.mount(make_anchor("js", "javascript", embedding=[0.0, 1.0]), weight=0.1)
        candidate = make_anchor("c", "python", embedding=[1.0, 0.0])
        assert user_center_proximity(candidate, center=center) == pytest.approx(1.0)
        assert user_center_proximity(candidate) == pytest.approx(0.0)
        empty = UserCenter(UserCenterConfig(capacity=4))
        plain = make_anchor("p", "python pytest")
        assert user_center_proximity(plain, query="pytest", center=empty) > 0.0
        assert user_center_proximity(plain, query=None, center=empty) == 0.0
        assert 0.0 <= combined_proximity(candidate, candidate, center=center,
                                         query="python") <= 1.0
        assert combined_proximity(candidate, candidate) >= 0.0

    def test_center_split_defaults_85_15(self) -> None:
        config = UserCenterConfig()
        assert config.task_center_weight == pytest.approx(0.85)
        assert config.tech_center_weight == pytest.approx(0.15)
        assert config.task_center_weight + config.tech_center_weight == pytest.approx(1.0)

    def test_center_split_fuses_task_and_tech(self) -> None:
        cfg = UserCenterConfig(capacity=4)
        tech = UserCenter(cfg, clock=FakeClock())
        tech.mount(make_anchor("py", "python", embedding=[1.0, 0.0], tags=("python",)))
        candidate = make_anchor("c", "python", embedding=[1.0, 0.0], tags=("python",))
        # 只有技术中心 → 归一为技术分（不乘 0.15）
        assert user_center_proximity(candidate, center=tech) == pytest.approx(1.0)
        # 只有本次任务中心 → 归一为任务分（不乘 0.85）
        empty = UserCenter(UserCenterConfig(capacity=4), clock=FakeClock())
        assert user_center_proximity(candidate, query=candidate,
                                     center=empty) == pytest.approx(1.0)
        # 两侧都可用：任务 0 / 技术 1 → 0.15
        assert user_center_proximity(candidate, query="javascript", center=tech) == pytest.approx(0.15)
        # 两侧都可用：任务 1 / 技术 0 → 0.85
        unrelated = UserCenter(UserCenterConfig(capacity=4), clock=FakeClock())
        unrelated.mount(make_anchor("js", "javascript", embedding=[0.0, 1.0],
                                    tags=("javascript",)))
        assert user_center_proximity(candidate, query=candidate,
                                     center=unrelated) == pytest.approx(0.85)

    def test_center_split_is_configurable_and_can_disable(self) -> None:
        cfg = UserCenterConfig(capacity=4, task_center_weight=0.5, tech_center_weight=0.5)
        tech = UserCenter(cfg, clock=FakeClock())
        tech.mount(make_anchor("js", "javascript", embedding=[0.0, 1.0],
                               tags=("javascript",)))
        candidate = make_anchor("c", "python", embedding=[1.0, 0.0], tags=("python",))
        assert user_center_proximity(candidate, query=candidate, center=tech,
                                     config=cfg) == pytest.approx(0.5)
        zero = UserCenterConfig(capacity=4, task_center_weight=0.0, tech_center_weight=0.0)
        assert user_center_proximity(candidate, query="python", center=tech,
                                     config=zero) == 0.0

    def test_task_anchors_extend_current_task_center(self) -> None:
        cfg = UserCenterConfig(capacity=4)
        task_memory = make_anchor("task", "修复 pytest 用例", embedding=[1.0, 0.0],
                                  tags=("pytest",))
        candidate = make_anchor("c", "pytest 用例", embedding=[1.0, 0.0],
                                tags=("pytest",))
        # query 无关，但本次任务锚点命中 → 任务分量仍为 1.0
        assert user_center_proximity(candidate, query="banana", task_anchors=[task_memory],
                                     config=cfg) == pytest.approx(1.0)


class TestMemoryChain:
    def test_chain_grows_until_similarity_drops(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=8), clock=FakeClock())
        seed = make_anchor("seed", "python 测试", embedding=[1.0, 0.0], tags=("python",))
        center.mount(seed)
        similar = make_anchor("similar", "python 测试 用例", embedding=[0.9, 0.1],
                              tags=("python",))
        unrelated = make_anchor("unrelated", "banana 香蕉", embedding=[0.0, 1.0])
        chain = build_memory_chain(seed, [similar, unrelated], center=center,
                                   query="python 测试")
        assert chain.ids == ["similar"]
        assert chain.stop_reason == STOP_SIMILARITY_BELOW_THRESHOLD
        assert chain.steps[0].scores()["combined"] >= 0.7
        assert chain.steps[0].previous_id == "seed" and chain.seed_id == "seed"

    def test_chain_stops_when_center_below_x(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=8), clock=FakeClock())
        center.mount(make_anchor("rust", "rust 服务", embedding=[0.0, 1.0], tags=("rust",)))
        seed = make_anchor("seed", "python 测试", embedding=[1.0, 0.0], tags=("python",))
        candidate = make_anchor("cand", "python 测试", embedding=[1.0, 0.0],
                                tags=("python",))
        config = UserCenterConfig(chain_similarity_threshold=0.20, stop_center_threshold=0.35)
        chain = build_memory_chain(seed, [candidate], center=center,
                                   query=[0.0, 1.0], config=config)
        assert chain.steps == [] and chain.node_ids() == ["seed"]
        assert chain.stop_reason == STOP_CENTER_BELOW_THRESHOLD

    def test_chain_center_gate_uses_fused_85_15_score(self) -> None:
        config = UserCenterConfig(chain_similarity_threshold=0.20,
                                  stop_center_threshold=0.35)
        center = UserCenter(config, clock=FakeClock())
        center.mount(make_anchor("rust", "rust 服务", embedding=[0.0, 1.0], tags=("rust",)))
        seed = make_anchor("seed", "python 测试", embedding=[1.0, 0.0], tags=("python",))
        candidate = make_anchor("cand", "python 测试", embedding=[1.0, 0.0],
                                tags=("python",))
        # query 与 rust 技术中心一致 → 任务 0 / 技术 1 → 融合 0.15 < x → 停
        chain = build_memory_chain(seed, [candidate], center=center,
                                   query=[0.0, 1.0], config=config)
        assert chain.stop_reason == STOP_CENTER_BELOW_THRESHOLD and chain.steps == []
        # query 与候选一致 → 任务 1 / 技术 0 → 融合 0.85 >= x → 继续
        chain2 = build_memory_chain(seed, [candidate], center=center,
                                    query=[1.0, 0.0], config=config)
        assert chain2.ids == ["cand"]

    def test_chain_stops_at_max_length_and_text_gate(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=8), clock=FakeClock())
        seed = make_anchor("seed", "python 测试", embedding=[1.0, 0.0], tags=("python",))
        center.mount(seed)
        candidates = [make_anchor(f"c{i}", "python 测试", embedding=[1.0, 0.0],
                                  tags=("python",)) for i in range(5)]
        chain = build_memory_chain(seed, candidates, center=center, query="python",
                                   config=UserCenterConfig(max_chain_length=3))
        assert len(chain.steps) == 2 and len(chain.node_ids()) == 3
        assert chain.stop_reason == STOP_MAX_LENGTH
        single = build_memory_chain(
            seed, candidates[:1], center=center, query="python",
            config=UserCenterConfig(chain_gate="text", chain_similarity_threshold=0.9))
        assert single.ids == ["c0"] and single.steps[0].text_score >= 0.9

    def test_chain_empty_no_seed_and_no_embeddings(self) -> None:
        seed = make_anchor("seed", "python 测试 用例")
        empty = build_memory_chain(seed, [])
        assert empty.steps == [] and empty.stop_reason == STOP_CANDIDATES_EXHAUSTED
        chain = build_memory_chain(None, [make_anchor("a", "python 测试")],
                                   query="python 测试")
        assert chain.seed is None and chain.steps[0].previous_id == ""
        degraded = build_memory_chain(
            seed, [make_anchor("a", "python 测试"), make_anchor("b", "rust 编译器")],
            query="测试", config=UserCenterConfig(chain_similarity_threshold=0.5))
        assert isinstance(degraded, MemoryChain) and degraded.ids == ["a"]
        assert 0.0 <= degraded.steps[0].combined_score <= 1.0

    def test_chain_is_deterministic(self) -> None:
        center = UserCenter(UserCenterConfig(capacity=8), clock=FakeClock())
        seed = make_anchor("seed", "python 测试", embedding=[1.0, 0.0], tags=("python",))
        center.mount(seed)
        candidates = [
            make_anchor("a", "python 测试", embedding=[1.0, 0.0], tags=("python",)),
            make_anchor("b", "python pytest", embedding=[0.9, 0.1], tags=("python",)),
            make_anchor("c", "rust", embedding=[0.0, 1.0]),
        ]
        first = build_memory_chain(seed, candidates, center=center, query="python")
        second = build_memory_chain(seed, list(candidates), center=center, query="python")
        assert first.to_dict() == second.to_dict()


class TestSerialization:
    def test_memory_chain_to_dict_is_json_friendly(self) -> None:
        chain = build_memory_chain(
            make_anchor("seed", "python 测试"), [make_anchor("a", "python 测试 用例")],
            config=UserCenterConfig(chain_similarity_threshold=0.45))
        payload = chain.to_dict()
        assert payload["seed_id"] == "seed"
        assert payload["stop_reason"] == STOP_CANDIDATES_EXHAUSTED
        assert json.loads(json.dumps(payload)) == payload
        assert payload["steps"][0]["scores"]["combined"] >= 0.0

    def test_default_import_does_not_load_user_center(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code = "import sys, star_graph;print('star_graph.user_center' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=root,
                                capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(root)})
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"
