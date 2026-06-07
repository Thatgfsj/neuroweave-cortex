"""Gravity Wells — query inertia & conversational continuity for star-graph memory.

连续查询不应独立计算。引力井模拟思维惯性：
- 前一个焦点会"吸引"下一个查询
- 对话中断或话题切换时，惯性断裂
- 情绪性的引力井比中性的衰减更慢
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GravityWell:
    """一个引力井——拖拽提灯位置的记忆或概念。

    Attributes:
        star_id: 引力井关联的星
        layer: 所在记忆层
        coord: 该层中的坐标
        strength: 引力强度（质量×情绪系数）
        decay_rate: 每轮对话后的衰减速度
        well_type: emotional / habitual / contextual
        created_at: 创建时间
        last_pulled: 最后一次被使用的时间
    """
    star_id: str
    layer: str
    coord: list[float]
    strength: float = 0.5
    decay_rate: float = 0.2  # 每轮衰减
    well_type: str = "contextual"
    created_at: float = field(default_factory=time.time)
    last_pulled: float = field(default_factory=time.time)


class GravityWellManager:
    """管理当前对话中的引力井。

    每轮对话维护一组活跃的引力井。
    新查询时，提灯位置被引力井拉向之前的焦点。
    """

    def __init__(self, inertia: float = 0.4, break_threshold: float = 0.8):
        self.inertia = inertia          # 惯性系数：0=完全重置 1=完全保持
        self.break_threshold = break_threshold  # 新query与焦点的距离超过此值→断裂
        self.wells: dict[str, GravityWell] = {}  # star_id → well
        self.current_focal_point: list[float] | None = None
        self._conversation_turns: int = 0

    def add_well(self, star_id: str, layer: str, coord: list[float],
                 strength: float = 0.5, well_type: str = "contextual") -> GravityWell:
        """创建或更新一个引力井。

        相同 star_id 的引力井会叠加强度。
        """
        if star_id in self.wells:
            well = self.wells[star_id]
            well.strength = min(1.0, well.strength + strength * 0.3)
            well.last_pulled = time.time()
            return well

        well = GravityWell(
            star_id=star_id, layer=layer, coord=coord,
            strength=strength, well_type=well_type,
        )
        self.wells[star_id] = well
        return well

    def apply_inertia(self, query_vector: list[float],
                      layer: str = "core") -> list[float]:
        """将引力井的拉力应用到新的查询向量上。

        新提灯位置 = 0.6 × query + 0.4 × 当前焦点 + Σ 附近引力井拉力

        如果新 query 与当前焦点的距离超过 break_threshold，则重置。
        """
        self._conversation_turns += 1

        if self.current_focal_point is None:
            self.current_focal_point = query_vector.copy()
            return query_vector

        # 检查是否话题断裂
        diff = [a - b for a, b in zip(query_vector, self.current_focal_point)]
        distance = math.sqrt(sum(d * d for d in diff))
        if distance > self.break_threshold:
            self.wells.clear()
            self.current_focal_point = query_vector.copy()
            return query_vector

        # 衰减所有引力井
        decayed = list(self.wells.keys())
        for sid in decayed:
            well = self.wells[sid]
            well.strength *= (1.0 - well.decay_rate * 0.1)
            if well.strength < 0.05:
                del self.wells[sid]

        # 收集同层中强度足够的引力井拉力
        pull_x = [0.0] * len(query_vector)
        total_strength = 0.0
        for well in self.wells.values():
            if well.layer != layer:
                continue
            # 距离越近引力越强
            d = [a - b for a, b in zip(well.coord, query_vector)]
            dist = math.sqrt(sum(x * x for x in d))
            if dist < 0.001:
                grav_strength = well.strength
            else:
                grav_strength = well.strength / (1.0 + dist * 2.0)

            for i in range(len(pull_x)):
                pull_x[i] += well.coord[i] * grav_strength
            total_strength += grav_strength

        # 融合: 新位置 = 惯性保留焦点 + 引力拉拽
        if total_strength > 0.01:
            for i in range(len(pull_x)):
                pull_x[i] /= total_strength
            blend = min(0.4, self.inertia + total_strength * 0.1)
            new_pos = [
                (1 - blend) * q + blend * p
                for q, p in zip(query_vector, pull_x)
            ]
        else:
            new_pos = [
                (1 - self.inertia) * q + self.inertia * f
                for q, f in zip(query_vector, self.current_focal_point)
            ]

        self.current_focal_point = new_pos
        return new_pos

    def reset(self) -> None:
        """强制重置——用户明确切换话题时调用。"""
        self.wells.clear()
        self.current_focal_point = None
        self._conversation_turns = 0

    def get_well_count(self) -> int:
        """当前活跃的引力井数量。"""
        return len(self.wells)

    def decay_all(self, rate_multiplier: float = 1.0) -> int:
        """批量衰减所有引力井，返回移除的数量。"""
        removed = 0
        expired = [sid for sid, w in self.wells.items()
                   if w.strength < 0.05]
        for sid in expired:
            del self.wells[sid]
            removed += 1
        return removed
