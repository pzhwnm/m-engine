"""
PreferenceModem: 偏好调制器。
负责读取用户和模型的增益参数，并将其施加到基问题的滤波权重上，
形成最终的"问题 · 偏好"复合权重向量。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """用户画像：每个用户对基函数维度的固有权重增益。

    gain: basis_id -> gain (float, 0.1~5.0)
        增益 > 1.0 表示该用户对该维度更敏感，提问时该维度权重会被放大。
        增益 < 1.0 表示该用户对该维度不敏感，权重会被压缩。
        默认值为 1.0（不改变）。
    """
    id: str = field(default_factory=lambda: f"user_{id(UserProfile)}")
    gain: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelConstraints:
    """模型约束：模型对各基函数维度的固有增益和安全阈值。

    gain: basis_id -> gain (float)
        模型自身的频率响应特性，类似于 LLM 的"性格"。
    safety_thresholds: basis_id -> max_activation (float)
        某些维度（如情感、道德）的激活上限，防止回答过激。
    """
    gain: Dict[str, float] = field(default_factory=dict)
    safety_thresholds: Dict[str, float] = field(default_factory=dict)


class PreferenceModem:
    """偏好调制器。

    偏好 = 系统的"频率响应曲线"。
    用户增益（UserProfile.gain）和模型增益（ModelConstraints.gain）
    叠加到基问题权重上，形成个性化的信息提取方向。

    公式：
        beta_final[i] = base_beta[i] * user_gain[i] * model_gain[i]
    然后再做 L1 归一化。
    """

    def get_user_gain(self, user: UserProfile) -> List[float]:
        """获取用户的增益向量。"""
        return list(user.gain.values()) if user.gain else []

    def get_model_gain(self, constraints: ModelConstraints) -> List[float]:
        """获取模型的增益向量。"""
        return list(constraints.gain.values()) if constraints.gain else []

    def apply_gain(
        self,
        base_beta: List[float],
        user_gain: List[float],
        model_gain: List[float],
    ) -> List[float]:
        """将用户和模型增益施加到基问题上。

        所有向量按位置对齐（假定已通过 QuestionRouter 保证维度一致）。
        缺失的用户/模型增益默认视为 1.0。
        """
        beta = np.array(base_beta, dtype=np.float32)
        n = len(beta)

        # 扩展或裁剪增益向量以匹配 base_beta 维度
        u_gain = self._align_gain(user_gain, n, default=1.0)
        m_gain = self._align_gain(model_gain, n, default=1.0)

        result = beta * u_gain * m_gain

        # L1 归一化
        total = np.sum(result)
        if total > 0:
            result = result / total

        logger.debug("Applied gain: base=%s user=%s model=%s -> %s",
                     beta, u_gain, m_gain, result)
        return result.tolist()

    @staticmethod
    def _align_gain(gain: List[float], target_len: int, default: float = 1.0) -> np.ndarray:
        """将增益向量对齐到目标长度。"""
        if not gain:
            return np.full(target_len, default, dtype=np.float32)
        if len(gain) < target_len:
            padded = list(gain) + [default] * (target_len - len(gain))
            return np.array(padded, dtype=np.float32)
        return np.array(gain[:target_len], dtype=np.float32)
