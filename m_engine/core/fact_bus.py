"""
FactBus: 事实总线，事实的内存数据库。
负责事实存储、频谱更新、相似度检索和激活追踪。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """事实：一段原始文本及其在基函数空间上的频谱。

    spectrum: basis_id -> score (float, 0~1)
        事实在各基函数上的投影系数，代表该基函数维度
        在此事实中的"能量"或"显著程度"。
    embedding: 事实的整体向量表示，用于相似度检索。
    activation: question_type -> activation_count, 追踪每种问题
        类型对该事实的激活次数，赫布式学习的基础。
    """
    id: str = field(default_factory=lambda: f"fact_{id(Fact)}")
    raw_text: str = ""
    spectrum: Dict[str, float] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    activation: Dict[str, int] = field(default_factory=dict)

    def get_spectrum_vector(self, basis_ids: List[str]) -> np.ndarray:
        """按基函数 ID 顺序返回频谱向量（numpy 数组）。"""
        return np.array([self.spectrum.get(bid, 0.0) for bid in basis_ids],
                        dtype=np.float32)


class FactBus:
    """事实总线：事实的内存数据库。

    核心职责：
    1. 存储和管理事实
    2. 基于向量相似度检索最相关的事实
    3. 更新事实频谱（支持赫布式学习和反馈调整）
    4. 追踪事实的激活历史
    """

    def __init__(self, embedder=None):
        self._facts: Dict[str, Fact] = {}
        self._embedder = embedder  # 可注入的嵌入函数

    def add_fact(self, fact: Fact) -> None:
        """添加一个新事实到总线。"""
        # 如果未提供嵌入，尝试通过 embedder 生成
        if not fact.embedding and self._embedder is not None:
            try:
                fact.embedding = list(self._embedder(fact.raw_text))
            except Exception as e:
                logger.warning("Failed to embed fact %s: %s", fact.id, e)
        self._facts[fact.id] = fact
        logger.info("Added fact: %s (text preview: %.50s...)", fact.id, fact.raw_text)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """按 ID 获取事实。"""
        return self._facts.get(fact_id)

    def list_all(self) -> List[Fact]:
        """列出所有事实。"""
        return list(self._facts.values())

    def update_spectrum(self, fact_id: str, basis_id: str, score: float) -> None:
        """更新事实在某个基函数上的频谱系数。
        score 被裁剪到 [0, 1] 范围。
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            logger.warning("update_spectrum: fact %s not found", fact_id)
            return
        fact.spectrum[basis_id] = max(0.0, min(1.0, score))
        logger.debug("Updated spectrum: fact=%s basis=%s score=%.3f",
                     fact_id, basis_id, fact.spectrum[basis_id])

    def update_activation(self, fact_id: str, question_type: str, delta: int = 1) -> None:
        """更新事实在某种问题类型下的激活计数（赫布式学习）。"""
        fact = self._facts.get(fact_id)
        if fact is None:
            return
        fact.activation[question_type] = fact.activation.get(question_type, 0) + delta

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[Fact]:
        """基于余弦相似度检索最相关的 top_k 个事实。

        如果所有事实都没有嵌入，则返回最近添加的事实。
        """
        if not self._facts:
            return []

        candidates = []
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)

        for fact in self._facts.values():
            if not fact.embedding or q_norm == 0:
                # 无嵌入时给一个小的均匀分数
                candidates.append((fact, 0.5))
                continue
            f_vec = np.array(fact.embedding, dtype=np.float32)
            f_norm = np.linalg.norm(f_vec)
            if f_norm == 0:
                candidates.append((fact, 0.5))
                continue
            sim = np.dot(q_vec, f_vec) / (q_norm * f_norm)
            candidates.append((fact, float(sim)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in candidates[:top_k]]

    def __len__(self) -> int:
        return len(self._facts)
