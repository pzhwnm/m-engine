"""
KnowledgeExchange: AI 间直接知识交换协议。

允许两个 M-Engine 实例直接交换事实的频谱系数，
实现高带宽、精确无歧义的知识迁移。

协议：
  1. 导出：将事实序列化为 {id, text, spectrum, basis_definitions}
  2. 导入：从序列化格式恢复事实
  3. 共识：对相同文本的多个频谱取加权平均
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from .basis_registry import BasisRegistry, BasisFunction
from .fact_bus import Fact, FactBus

logger = logging.getLogger(__name__)


class KnowledgeExchange:
    """AI-AI 知识交换协议。"""

    def __init__(self, basis_registry: BasisRegistry, fact_bus: FactBus):
        self.basis = basis_registry
        self.fact_bus = fact_bus

    def export_fact(self, fact_id: str) -> Optional[Dict]:
        """导出一个事实的完整可交换表示。"""
        fact = self.fact_bus.get_fact(fact_id)
        if fact is None:
            return None

        # 附上基函数定义（接收方可能需要）
        basis_defs = []
        for b in self.basis.list_all():
            basis_defs.append({
                "id": b.id,
                "name": b.name,
                "description": b.description,
            })

        return {
            "protocol_version": "0.1.0",
            "fact": {
                "id": fact.id,
                "raw_text": fact.raw_text,
                "spectrum": fact.spectrum,
            },
            "basis_definitions": basis_defs,
        }

    def export_all(self) -> List[Dict]:
        """导出所有事实。"""
        return [
            self.export_fact(f.id)
            for f in self.fact_bus.list_all()
            if self.export_fact(f.id) is not None
        ]

    def import_fact(self, data: Dict) -> Optional[Fact]:
        """从交换格式导入一个事实。

        如果已存在相同文本的事实，执行共识合并而非覆盖。
        """
        fact_data = data.get("fact", {})
        text = fact_data.get("raw_text", "")
        spectrum = fact_data.get("spectrum", {})

        if not text:
            return None

        # 检查是否已存在相同文本的事实
        existing = self._find_by_text(text)
        if existing is not None:
            return self._consensus_merge(existing, spectrum)

        # 新事实
        fact = Fact(
            id=fact_data.get("id", ""),
            raw_text=text,
            spectrum=spectrum,
        )
        self.fact_bus.add_fact(fact)
        logger.info("Imported fact: %s", fact.id)
        return fact

    def import_all(self, data_list: List[Dict]) -> int:
        """批量导入事实。"""
        count = 0
        for data in data_list:
            if self.import_fact(data) is not None:
                count += 1
        return count

    def _find_by_text(self, text: str) -> Optional[Fact]:
        """按文本查找已有事实（用于共识合并）。"""
        for f in self.fact_bus.list_all():
            if f.raw_text.strip() == text.strip():
                return f
        return None

    def _consensus_merge(
        self,
        existing: Fact,
        new_spectrum: Dict[str, float],
        weight_existing: float = 0.5,
        weight_new: float = 0.5,
    ) -> Fact:
        """共识合并：对同一事实的两个频谱取加权平均。

        这实现了蓝图中的"两个 AI 智能体达成共识时直接交换频谱值"。
        """
        all_keys = set(existing.spectrum.keys()) | set(new_spectrum.keys())
        for key in all_keys:
            old_val = existing.spectrum.get(key, 0.0)
            new_val = new_spectrum.get(key, 0.0)
            existing.spectrum[key] = old_val * weight_existing + new_val * weight_new
        logger.info(
            "Consensus merged fact '%s': %d dims averaged",
            existing.raw_text[:40], len(all_keys)
        )
        return existing

    def compute_consensus(
        self,
        spectra: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """计算多个频谱的共识（简单平均）。"""
        if not spectra:
            return {}
        all_keys = set()
        for s in spectra:
            all_keys.update(s.keys())
        consensus = {}
        for key in all_keys:
            values = [s.get(key, 0.0) for s in spectra]
            consensus[key] = float(np.mean(values))
        return consensus
