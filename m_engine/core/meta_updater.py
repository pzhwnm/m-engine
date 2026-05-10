"""
MetaUpdater: 元更新网络。
根据用户反馈（reinforcement signal）局部修复事实频谱中的投影系数。
MVP 阶段使用简单的线性层实现。
"""

import logging
from typing import Dict, List

import numpy as np

from .fact_bus import FactBus

logger = logging.getLogger(__name__)


class MetaUpdater:
    """元更新网络。

    职责：接收用户反馈信号（-1.0 ~ 1.0），计算频谱修正量
    delta_s，并实际修改 FactBus 中事实的频谱值。

    MVP 阶段：简化为线性映射。
        delta_s = tanh(W @ [beta; feedback] + b)
    其中 beta 是本次交互的滤波权重向量，
    feedback 是标量反馈值。

    更新规则（赫布式）：
        s_new = s_old + lr * delta_s * feedback
    正反馈增强相关维度的系数，负反馈压制。

    同时更新事实激活计数，高激活 + 正反馈 = 该维度在该事实上的频谱被强化。
    """

    def __init__(self, d_B: int, rank: int = 16, hidden_dim: int = 64, lr: float = 0.1):
        """
        Args:
            d_B: 基函数空间的维度
            rank: 低秩分解的秩（MVP阶段保留接口，实际使用完整线性层）
            hidden_dim: 隐藏层维度
            lr: 学习率
        """
        self.d_B = d_B
        self.rank = rank
        self.lr = lr

        # 简单线性层：输入 = [beta; feedback], 输出 = delta_s
        # 使用 Xavier 初始化
        input_dim = d_B + 1
        rng = np.random.RandomState(42)
        limit = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-limit, limit, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.uniform(-limit, limit, (d_B, hidden_dim))
        self.b2 = np.zeros(d_B)

    def compute_delta(self, beta: List[float], feedback: float) -> np.ndarray:
        """计算频谱修正量 delta_s。

        Args:
            beta: 当前交互的滤波权重向量
            feedback: 用户反馈值（-1.0 ~ 1.0）

        Returns:
            delta_s: 各基函数维度的修正量
        """
        beta_arr = np.array(beta, dtype=np.float32)
        if len(beta_arr) != self.d_B:
            # 对齐：截断或补零
            if len(beta_arr) < self.d_B:
                beta_arr = np.pad(beta_arr, (0, self.d_B - len(beta_arr)))
            else:
                beta_arr = beta_arr[:self.d_B]

        x = np.append(beta_arr, feedback).astype(np.float32)
        h = np.tanh(self.W1 @ x + self.b1)
        delta = self.W2 @ h + self.b2
        return delta

    def update(
        self,
        fact_bus: FactBus,
        fact_id: str,
        q_type: str,
        beta: List[float],
        feedback: float,
        basis_ids: List[str],
    ) -> Dict:
        """执行一次记忆更新。

        1. 计算频谱修正量 delta_s
        2. 将修正量写入 FactBus 中对应事实的频谱
        3. 更新激活计数

        Args:
            fact_bus: 事实总线实例
            fact_id: 要更新的事实 ID
            q_type: 问题类型 ID
            beta: 滤波权重向量
            feedback: 用户反馈值（-1.0 ~ 1.0）
            basis_ids: 基函数 ID 列表（用于维度对齐）

        Returns:
            包含更新统计信息的字典
        """
        fact = fact_bus.get_fact(fact_id)
        if fact is None:
            logger.warning("MetaUpdater.update: fact %s not found", fact_id)
            return {"status": "error", "message": f"Fact {fact_id} not found"}

        # 裁剪反馈值
        feedback = max(-1.0, min(1.0, feedback))

        # 计算修正量
        delta = self.compute_delta(beta, feedback)

        # 应用修正
        changes = {}
        for i, bid in enumerate(basis_ids):
            if i >= len(delta):
                break
            old_score = fact.spectrum.get(bid, 0.0)
            new_score = old_score + self.lr * delta[i] * feedback
            new_score = max(0.0, min(1.0, new_score))  # 裁剪到 [0,1]
            fact_bus.update_spectrum(fact_id, bid, new_score)
            changes[bid] = {"old": round(old_score, 4), "new": round(new_score, 4)}

        # 更新激活计数
        fact_bus.update_activation(fact_id, q_type, delta=1)

        logger.info(
            "MetaUpdater: fact=%s q_type=%s feedback=%.2f changed=%d dims",
            fact_id, q_type, feedback, len(changes),
        )

        return {
            "status": "ok",
            "fact_id": fact_id,
            "feedback": feedback,
            "changes": changes,
        }
