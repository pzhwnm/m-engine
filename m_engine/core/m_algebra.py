"""
MAlgebraCore: M-代数核心引擎。
执行完整的张量收缩运算，使用可学习的低秩矩阵 W、G、S
实现记忆的压缩、调制与重构。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fact_bus import Fact
from .basis_registry import BasisRegistry

logger = logging.getLogger(__name__)


class MAlgebraCore:
    """M-代数核心引擎。

    完整公式：
        r = (W @ beta) ⊙ exp(G @ pref) ⊙ avg_spectrum

    其中：
        W: d_B × d_B  问题投影矩阵 — 学习"哪种问题类型激活哪些基函数"
        G: d_B × d_B  偏好投影矩阵 — 学习"用户/模型偏好如何调制基函数"
        beta:          问题滤波权重向量 (d_B)
        pref:          偏好向量 (d_B, 已合并 user_gain * model_gain)
        avg_spectrum:  事实平均频谱向量 (d_B)
        ⊙:            逐元素乘法 (Hadamard product)

    学习规则（赫布式）：
        正反馈 → 加强 beta[i] 和 activation[j] 之间的 W[i,j] 连接
        正反馈 → 加强 pref[i] 和 activation[j] 之间的 G[i,j] 连接
    """

    def __init__(self, basis_registry: BasisRegistry, rank: int = 16):
        self._basis = basis_registry
        self.d_B = 0  # 延迟初始化
        self.rank = rank

        # 可学习矩阵（初始化后由 set_dim 设置）
        self.W: Optional[np.ndarray] = None  # d_B × d_B
        self.G: Optional[np.ndarray] = None  # d_B × d_B
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        d = len(self._basis)
        if d == 0:
            return
        self.set_dim(d)

    def set_dim(self, d_B: int):
        """根据基函数数量初始化投影矩阵。"""
        self.d_B = d_B
        rng = np.random.RandomState(42)
        # W 初始化为近单位矩阵（问题类型默认直接映射到对应基函数）
        self.W = np.eye(d_B, dtype=np.float32) + rng.normal(0, 0.01, (d_B, d_B)).astype(np.float32)
        # G 初始化为近零矩阵（偏好默认不引入调制）
        self.G = rng.normal(0, 0.01, (d_B, d_B)).astype(np.float32)
        self._initialized = True
        logger.info("MAlgebraCore initialized: d_B=%d", d_B)

    def compute_activation(
        self,
        facts: List[Fact],
        filter_beta: List[float],
        preference: List[float],
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """使用完整 M-代数公式计算基函数激活强度。

        Args:
            facts: 检索到的事实列表（已按相关性排序）
            filter_beta: 滤波权重向量（已包含偏好调制后的最终 beta）
            preference: 偏好向量 (user_gain ⊙ model_gain)

        Returns:
            (activation_vector, top_activations)
        """
        self._ensure_init()
        if self.d_B == 0 or not facts:
            return np.zeros(0), []

        beta = self._align_vector(np.array(filter_beta, dtype=np.float32), self.d_B)
        pref = self._align_vector(np.array(preference, dtype=np.float32), self.d_B)

        basis_ids = [b.id for b in self._basis.list_all()]

        # 1. 计算平均频谱: S^T @ f (加权平均)
        avg_spectrum = np.zeros(self.d_B, dtype=np.float32)
        total_weight = 0.0
        for rank, fact in enumerate(facts):
            s_vec = fact.get_spectrum_vector(basis_ids[:self.d_B])
            s_vec = self._align_vector(s_vec, self.d_B)
            weight = 1.0 / (rank + 1)  # 排名衰减
            avg_spectrum += s_vec * weight
            total_weight += weight
        if total_weight > 0:
            avg_spectrum /= total_weight

        # 2. 问题投影: W @ beta
        q_proj = self.W @ beta

        # 3. 偏好调制: exp(G @ pref)，用 softplus 避免数值爆炸
        p_mod = self._safe_exp(self.G @ pref)

        # 4. 三者逐元素乘积
        activation = q_proj * p_mod * avg_spectrum

        # 构建 top-k 激活列表
        basis_list = self._basis.list_all()
        activations: List[Tuple[str, float]] = []
        for i, b in enumerate(basis_list):
            if i < len(activation):
                activations.append((b.name, float(activation[i])))
        activations.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Top activations: %s",
                     [(n, round(v, 4)) for n, v in activations[:5]])
        return activation, activations

    def build_prompt(
        self,
        facts: List[Fact],
        activations: List[Tuple[str, float]],
        user_query: str,
        top_k: int = 5,
    ) -> str:
        """构造给 LLM 的结构化提示词。"""
        top_dims = activations[:top_k]
        dim_lines = []
        for name, strength in top_dims:
            if strength > 0.001:
                bar = "█" * max(1, int(min(strength * 20, 20)))
                dim_lines.append(f"  [{bar}] {name} (强度: {strength:.4f})")

        dim_section = "\n".join(dim_lines) if dim_lines else "  (无显著激活维度)"

        fact_section_parts = []
        for i, f in enumerate(facts, 1):
            fact_section_parts.append(f"事实 {i}: {f.raw_text}")
        fact_section = "\n".join(fact_section_parts)

        prompt = f"""你是一个具备多维度分析能力的记忆助手。你会从多个认知维度审视事实，并给出结构化的回答。

【当前激活的认知维度】（问题类型和偏好决定了哪些维度被激活）
{dim_section}

【相关事实】
{fact_section}

【用户问题】
{user_query}

请从上述激活维度出发，综合分析后回答问题。如果某些维度激活很弱，可以忽略它们。回答保持简洁，2-5句话为宜。"""

        return prompt

    def forward(
        self,
        facts: List[Fact],
        filter_beta: List[float],
        user_gain: List[float],
        model_gain: List[float],
        user_query: str,
        decoder,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """M-代数前向传播。

        1. 计算偏好向量 = user_gain * model_gain
        2. 完整公式计算激活强度
        3. 构造 prompt → 调用 Decoder
        """
        self._ensure_init()

        # 合成偏好向量
        ug = np.array(user_gain, dtype=np.float32) if user_gain else np.ones(self.d_B, dtype=np.float32)
        mg = np.array(model_gain, dtype=np.float32) if model_gain else np.ones(self.d_B, dtype=np.float32)
        ug = self._align_vector(ug, self.d_B)
        mg = self._align_vector(mg, self.d_B)
        pref = ug * mg

        activation, attention = self.compute_activation(facts, filter_beta, pref.tolist())
        prompt = self.build_prompt(facts, attention, user_query)
        # 兼容旧式 decoder（返回 str）和新式 decoder（返回 DecodeResult）
        raw = decoder.decode(prompt, activation.tolist(),
                            [b.name for b in self._basis.list_all()])
        answer = raw.text if hasattr(raw, 'text') else raw
        return answer, attention

    def update_matrices(
        self,
        beta: List[float],
        preference: List[float],
        activation: np.ndarray,
        feedback: float,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """赫布式更新 W 和 G 矩阵。

        正反馈：加强 beta[i]→activation[j] 和 pref[i]→activation[j] 的关联
        负反馈：削弱这些关联

        Args:
            beta: 当前滤波权重
            preference: 偏好向量
            activation: 本次计算的激活向量
            feedback: 反馈信号 (-1.0 ~ 1.0)
            lr: 学习率

        Returns:
            {"delta_W_norm": ..., "delta_G_norm": ...}
        """
        self._ensure_init()

        b = self._align_vector(np.array(beta, dtype=np.float32), self.d_B)
        p = self._align_vector(np.array(preference, dtype=np.float32), self.d_B)
        a = self._align_vector(activation.astype(np.float32), self.d_B)

        # 使用 L2 规范化后的外积更新
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        p_norm = p / (np.linalg.norm(p) + 1e-8)

        # W: 问题空间 → 激活空间 的映射
        delta_W = lr * feedback * np.outer(b_norm, a_norm)
        self.W += delta_W

        # G: 偏好空间 → 激活空间 的指数调制
        delta_G = lr * feedback * np.outer(p_norm, a_norm)
        self.G += delta_G

        # 裁剪防止数值爆炸
        self.W = np.clip(self.W, -5.0, 5.0)
        self.G = np.clip(self.G, -5.0, 5.0)

        dw = float(np.linalg.norm(delta_W))
        dg = float(np.linalg.norm(delta_G))

        logger.debug("M-Algebra update: |dW|=%.4f |dG|=%.4f feedback=%.2f", dw, dg, feedback)
        return {"delta_W_norm": dw, "delta_G_norm": dg}

    @staticmethod
    def _align_vector(v: np.ndarray, target_len: int) -> np.ndarray:
        if len(v) == target_len:
            return v.astype(np.float32)
        if len(v) < target_len:
            return np.pad(v.astype(np.float32), (0, target_len - len(v)))
        return v[:target_len].astype(np.float32)

    @staticmethod
    def _safe_exp(x: np.ndarray) -> np.ndarray:
        """安全的指数函数：对输入做裁剪防止溢出。"""
        x = np.clip(x, -10.0, 10.0)
        return np.exp(x)
