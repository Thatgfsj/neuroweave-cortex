"""Observatory — observer-dependent luminosity engine for star-graph memory.

核心范式迁移:
  Before: query → embed → cosine similarity → sorted list
  After:  lantern → illuminate star field → real-time luminosity → discovery path

每颗星的亮度不是存储属性，而是每次观测时根据提灯位置实时计算的。
这是迷雾、地标、虫洞和透镜投影的基础。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── 向量工具（纯 math，无 numpy 依赖） ─────────────────────

def _vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _vec_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    na = _vec_norm(a)
    nb = _vec_norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(0.0, _vec_dot(a, b) / (na * nb))


# ── 提灯：观测者的光源 ────────────────────────────────────

@dataclass
class Lantern:
    """观测者的光源——决定了星空中什么可见。

    每次查询由 perception.py 生成。携带观测者在各记忆层的位置、
    强度（查询清晰度）、光谱滤镜（语义焦点）和焦距（近/远视野）。

    Attributes:
        layer: 当前照亮哪一层 working / core / short / long
        position: 在该层坐标空间中的位置（embedding 向量）
        intensity: 0..1 光强（查询确定性）
        color_filter: 光谱标签集（要放大的语义标签）
        focal_depth: 0=仅近场可见 1=远场也可见
        source: query / mood / goal / free_associate
    """
    layer: str
    position: list[float]
    intensity: float = 1.0
    color_filter: set[str] = field(default_factory=set)
    focal_depth: float = 0.5
    source: str = "query"

    @classmethod
    def from_query(cls, query: str, query_embedding: list[float],
                   layer: str = "core",
                   tags: list[str] | None = None,
                   intensity: float | None = None) -> Lantern:
        """从用户查询创建提灯。

        长查询 = 更强的光、更宽的视野。
        短/模糊查询 = 弱光，仅照亮高反射率的星。
        """
        if intensity is None:
            word_count = len(query.split())
            intensity = min(1.0, 0.3 + word_count * 0.07)

        color_filter = set(tags or [])
        tech_kw = re.findall(r'\b([A-Z]\w*(?:thon|js|py|rb|sh|ql|db|sql|ui|api|sdk|cli|http|ssh|git))\b', query)
        color_filter.update(k.lower() for k in tech_kw)

        return cls(
            layer=layer,
            position=query_embedding,
            intensity=intensity,
            color_filter=color_filter,
            focal_depth=0.5,
            source="query",
        )


# ── 亮度计算函数 ──────────────────────────────────────────

def compute_luminosity(
    star_coord: list[float],
    lantern: Lantern,
    reflectivity: float = 0.5,
    spectral_tags: set[str] | None = None,
    mass: float = 1.0,
) -> float:
    """计算一颗星从提灯位置看过去的感知亮度。

    这不是 cosine similarity。这是观测者依赖的亮度模型：

    luminosity = reflectivity × intensity
                 × alignment × depth_focal
                 × color_match

    Args:
        star_coord: 星在提灯当前层的坐标
        lantern: 光源
        reflectivity: 0..1 星的固有反射率
        spectral_tags: 星的语义标签
        mass: 星的质量（影响深度曲线）

    Returns:
        0..1 感知亮度
    """
    if lantern.intensity < 0.01:
        return 0.0

    # 1. 对齐度：提灯光束打到这颗星的角度
    alignment = _cosine_sim(lantern.position, star_coord)

    # 2. 距离因子：星离提灯多远
    diff = _vec_sub(lantern.position, star_coord)
    distance = _vec_norm(diff)
    depth_factor = _depth_focal(distance, lantern.focal_depth, mass)

    # 3. 色匹配：光谱标签重叠度
    cm = _color_match(spectral_tags or set(), lantern.color_filter)

    # 4. 最终亮度
    luminosity = (
        reflectivity
        * lantern.intensity
        * alignment
        * depth_factor
        * (0.7 + 0.3 * cm)
    )

    return min(1.0, max(0.0, luminosity))


def _depth_focal(distance: float, focal_depth: float, mass: float = 1.0) -> float:
    """景深：给定距离的星在多深焦平面上可见。

    focal_depth=0 → 仅近处星可见（窄焦）
    focal_depth=1 → 远处星也可见（广角）

    mass 起引力透镜作用：大质量星在同距离下显得更亮。
    """
    if distance < 0.01:
        return 1.0

    effective_distance = distance / max(0.1, math.sqrt(mass))
    half_life = 0.3 + focal_depth * 2.0
    return math.exp(-effective_distance / half_life)


def _color_match(star_tags: set[str], filter_tags: set[str]) -> float:
    """星的谱标签与提灯滤镜的匹配度。

    Returns:
        0..1, 1=完全匹配
    """
    if not filter_tags or not star_tags:
        return 0.5

    star_lower = {t.lower() for t in star_tags}
    filter_lower = {t.lower() for t in filter_tags}
    intersection = star_lower & filter_lower

    if not intersection:
        return 0.3

    jaccard = len(intersection) / max(1, len(filter_lower))
    return min(1.0, 0.5 + jaccard * 0.5)


# ── 照亮结果 ──────────────────────────────────────────────

@dataclass
class IlluminationResult:
    """一次观测的结果。"""
    star_id: str
    luminosity: float
    source: str
    layer: str
    star_text: str = ""


# ── 天文台：协调多提灯照亮 ───────────────────────────────

class Observatory:
    """天文台——协调星空照亮。

    管理多个提灯（query、mood、goal），
    计算每颗星的亮度，返回最亮的星及其发现路径。
    """

    def __init__(self):
        self.lanterns: list[Lantern] = []

    def set_lanterns(self, lanterns: list[Lantern]) -> None:
        """设置本次观测的提灯。"""
        self.lanterns = lanterns

    def illuminate(self, star_id: str,
                   coord: list[float],
                   reflectivity: float = 0.5,
                   spectral_tags: set[str] | None = None,
                   mass: float = 1.0,
                   text: str = "") -> IlluminationResult:
        """在所有活跃提灯下计算星的亮度。

        多提灯规则：
        - 取最高亮度
        - 当多个提灯同时照到（都>0.3），1.5x 汇合奖励
        """
        if not self.lanterns:
            return IlluminationResult(
                star_id=star_id, luminosity=0.0,
                source="no_lantern", layer="", star_text=text)

        best_lum = 0.0
        best_source = ""
        best_layer = ""
        concurrent = 0

        for lantern in self.lanterns:
            lum = compute_luminosity(
                star_coord=coord,
                lantern=lantern,
                reflectivity=reflectivity,
                spectral_tags=spectral_tags,
                mass=mass,
            )
            if lum > best_lum:
                best_lum = lum
                best_source = lantern.source
                best_layer = lantern.layer
            if lum > 0.3:
                concurrent += 1

        if concurrent >= 2:
            best_lum = min(1.0, best_lum * 1.5)

        return IlluminationResult(
            star_id=star_id, luminosity=best_lum,
            source=best_source, layer=best_layer, star_text=text,
        )

    def illuminate_batch(self, stars: list[tuple],
                          top_k: int = 20) -> list[IlluminationResult]:
        """照亮多颗星，返回最亮的 top-k。

        stars: [(star_id, coord, reflectivity, spectral_tags, mass, text), ...]
        """
        results: list[IlluminationResult] = []
        for item in stars:
            sid, coord, ref, tags, mass_val, text = item
            if not coord or _vec_norm(coord) < 1e-10:
                continue
            result = self.illuminate(sid, coord, ref, tags, mass_val, text)
            results.append(result)

        results.sort(key=lambda r: -r.luminosity)
        return results[:top_k]
