"""
BasisRegistry: 基函数注册表，管理所有基函数（BasisFunction）的 CRUD 操作。
基函数是"世界基本逻辑维度"的可解释符号表示，如因果关系、情感色调等。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BasisFunction:
    """基函数：世界基本逻辑维度的符号表示。

    每个基函数代表一种理解事实的"视角"或"滤镜"，
    事实在该基函数上的投影系数构成频谱的一部分。

    stats 字段用于基函数动力学（GA+CA 演化）：
      activation_count: 累计被激活次数
      strength_sum: 累计激活强度（用于计算均值）
      strength_history: 最近 N 次激活强度（用于检测分裂信号）
      created_at: 创建时间戳
      last_activated: 最后激活时间戳
      parent_ids: 来源基函数 ID（分裂/合并产生时记录）
      generation: 演化代数
    """
    id: str = field(default_factory=lambda: f"basis_{id(BasisFunction)}")
    name: str = ""
    description: str = ""
    embedding: List[float] = field(default_factory=list)

    # 动力学统计
    activation_count: int = 0
    strength_sum: float = 0.0
    strength_history: List[float] = field(default_factory=list)
    created_at: float = 0.0
    last_activated: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    def record_activation(self, strength: float) -> None:
        """记录一次激活事件。"""
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        self.activation_count += 1
        self.strength_sum += strength
        self.strength_history.append(strength)
        if len(self.strength_history) > 100:
            self.strength_history = self.strength_history[-100:]
        self.last_activated = now

    @property
    def avg_strength(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.strength_sum / self.activation_count

    @property
    def strength_variance(self) -> float:
        """激活强度的方差 — 高方差是分裂的信号。"""
        if len(self.strength_history) < 3:
            return 0.0
        import numpy as np
        return float(np.var(self.strength_history))


class BasisRegistry:
    """基函数注册表。

    管理系统中所有活跃的基函数，提供注册、查询、列表等功能。
    基函数是整个 M-Engine 的"世界观底座"——它们定义了系统从哪些维度
    理解事实、解读问题、调制偏好。
    """

    def __init__(self):
        self._basis: Dict[str, BasisFunction] = {}

    def register(self, basis: BasisFunction) -> None:
        """注册一个新的基函数。同名基函数将被覆盖。"""
        self._basis[basis.id] = basis
        logger.info("Registered basis: %s (%s)", basis.name, basis.id)

    def get(self, basis_id: str) -> Optional[BasisFunction]:
        """按 ID 获取基函数。"""
        return self._basis.get(basis_id)

    def list_all(self) -> List[BasisFunction]:
        """列出所有已注册的基函数。"""
        return list(self._basis.values())

    def get_embedding(self, basis_id: str) -> List[float]:
        """获取基函数的向量表示，用于相似度计算。"""
        b = self._basis.get(basis_id)
        if b is None:
            return []
        return b.embedding

    def get_names(self) -> List[str]:
        """获取所有基函数名称列表（按注册顺序）。"""
        return [b.name for b in self._basis.values()]

    def load_from_json(self, filepath: str) -> None:
        """从 JSON 文件批量加载基函数定义。

        JSON 格式：数组，每个元素包含 id, name, description, embedding。
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Basis definition file not found: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            bf = BasisFunction(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                embedding=item.get("embedding", []),
            )
            self.register(bf)
        logger.info("Loaded %d basis functions from %s", len(data), filepath)

    def remove(self, basis_id: str) -> bool:
        """移除一个基函数（用于死亡/合并操作）。"""
        if basis_id in self._basis:
            del self._basis[basis_id]
            logger.info("Removed basis: %s", basis_id)
            return True
        return False

    def record_activation(self, basis_id: str, strength: float) -> None:
        """记录基函数的激活事件（用于动力学统计）。"""
        b = self._basis.get(basis_id)
        if b is not None:
            b.record_activation(strength)

    def get_stats(self) -> List[Dict]:
        """获取所有基函数的统计信息。"""
        now = time.time()
        stats = []
        for b in self._basis.values():
            stats.append({
                "id": b.id,
                "name": b.name,
                "activation_count": b.activation_count,
                "avg_strength": round(b.avg_strength, 4),
                "variance": round(b.strength_variance, 4),
                "generation": b.generation,
                "age_seconds": round(now - b.created_at, 1) if b.created_at > 0 else 0,
                "idle_seconds": round(now - b.last_activated, 1) if b.last_activated > 0 else 0,
                "parent_ids": b.parent_ids,
            })
        return stats

    def get_by_name(self, name: str) -> Optional[BasisFunction]:
        """按名称查找基函数。"""
        for b in self._basis.values():
            if b.name == name:
                return b
        return None

    def __len__(self) -> int:
        return len(self._basis)
