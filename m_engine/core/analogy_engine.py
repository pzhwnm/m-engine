"""
AnalogyEngine: 跨记忆类比推理引擎。

利用所有事实在同一基函数空间中的频谱向量，
计算事实间的谱相似度，发现结构相似的记忆模式，
生成自然语言类比报告。

核心操作：频谱向量的余弦相似度 + 共享高激活维度提取。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_registry import BasisRegistry
from .fact_bus import Fact

logger = logging.getLogger(__name__)


@dataclass
class Analogy:
    """一对事实之间的类比关系。"""
    fact_a_id: str
    fact_a_text: str
    fact_b_id: str
    fact_b_text: str
    similarity: float                    # 总体谱相似度
    shared_dimensions: List[Tuple[str, float, float]]  # (维度名, a谱值, b谱值)
    narrative: str = ""                  # 自然语言类比描述


class AnalogyEngine:
    """跨记忆类比推理引擎。

    同一基函数空间中的不同事实频谱向量可以互相比较，
    发现"分手"和"失业"在【失去】【自我价值重估】等维度上的结构相似性。
    """

    def __init__(self, basis_registry: BasisRegistry, similarity_threshold: float = 0.7):
        self.basis = basis_registry
        self.threshold = similarity_threshold

    def compute_similarity_matrix(self, facts: List[Fact]) -> np.ndarray:
        """计算事实-事实频谱余弦相似度矩阵。"""
        n = len(facts)
        if n < 2:
            return np.zeros((n, n))

        basis_ids = [b.id for b in self.basis.list_all()]
        if not basis_ids:
            return np.zeros((n, n))

        # 构建频谱矩阵 (n_facts, d_B)
        spec_matrix = np.zeros((n, len(basis_ids)), dtype=np.float32)
        for i, f in enumerate(facts):
            for j, bid in enumerate(basis_ids):
                spec_matrix[i, j] = f.spectrum.get(bid, 0.0)

        # 余弦相似度
        norms = np.linalg.norm(spec_matrix, axis=1, keepdims=True) + 1e-8
        spec_norm = spec_matrix / norms
        sim_matrix = spec_norm @ spec_norm.T

        return sim_matrix

    def find_analogies(self, facts: List[Fact], top_k: int = 10) -> List[Analogy]:
        """发现事实间的类比关系。

        Returns:
            按相似度降序的类比列表
        """
        if len(facts) < 2:
            return []

        sim = self.compute_similarity_matrix(facts)
        basis_list = self.basis.list_all()
        basis_ids = [b.id for b in basis_list]

        analogies = []
        n = len(facts)

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    # 找出共享的高激活维度
                    shared = []
                    for k, b in enumerate(basis_list):
                        a_val = facts[i].spectrum.get(b.id, 0.0)
                        b_val = facts[j].spectrum.get(b.id, 0.0)
                        if a_val > 0.05 and b_val > 0.05:
                            shared.append((b.name, round(a_val, 3), round(b_val, 3)))

                    shared.sort(key=lambda x: x[1] + x[2], reverse=True)

                    analogy = Analogy(
                        fact_a_id=facts[i].id,
                        fact_a_text=facts[i].raw_text,
                        fact_b_id=facts[j].id,
                        fact_b_text=facts[j].raw_text,
                        similarity=round(float(sim[i, j]), 4),
                        shared_dimensions=shared[:5],
                    )

                    if shared:
                        analogy.narrative = self._build_narrative(analogy)
                    analogies.append(analogy)

        analogies.sort(key=lambda a: a.similarity, reverse=True)
        return analogies[:top_k]

    def _build_narrative(self, a: Analogy) -> str:
        """生成自然语言类比描述。"""
        dims = [d[0] for d in a.shared_dimensions[:3]]
        if not dims:
            return ""
        dim_str = "、".join(dims)
        return (
            f"这两段记忆在【{dim_str}】维度上具有高度相似的结构模式。"
            f"它们的频谱相似度为 {a.similarity:.2f}，"
            f"表明尽管具体情节不同，但底层的认知骨架高度一致。"
        )

    def get_analogy_summary(self, facts: List[Fact]) -> List[Dict]:
        """获取类比摘要（供 CLI 使用）。"""
        analogies = self.find_analogies(facts)
        return [
            {
                "fact_a": a.fact_a_text[:60],
                "fact_b": a.fact_b_text[:60],
                "similarity": a.similarity,
                "dimensions": [d[0] for d in a.shared_dimensions[:3]],
                "narrative": a.narrative,
            }
            for a in analogies
        ]
