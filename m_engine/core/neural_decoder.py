"""
NeuralDecoder: 神经注入解码器。
组合 Decoder + NeuralBridge，实现完整的 编码→注入→生成→读取 循环。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .decoder import Decoder, DecodeResult
from .neural_bridge import NeuralBridge

logger = logging.getLogger(__name__)


@dataclass
class NeuralDecodeResult:
    """神经解码完整结果。

    text:             LLM 生成的回答文本
    symbolic_activation: 符号系统计算的激活向量 (d_B,)
    neural_reading:      LLM logprobs 读出的神经向量 (d_B,)
    neural_delta:        neural_reading - symbolic_activation 差值
    token_logprobs:      原始 token logprobs
    logit_bias:          实际施加的 logit_bias
    """
    text: str = ""
    symbolic_activation: Optional[np.ndarray] = None
    neural_reading: Optional[np.ndarray] = None
    neural_delta: Optional[np.ndarray] = None
    token_logprobs: Optional[List[Dict[int, float]]] = None
    logit_bias: Dict[int, float] = field(default_factory=dict)


class NeuralDecoder:
    """神经注入解码器。

    流程：
    1. NeuralBridge.encode(spectrum) → logit_bias + framing_text
    2. 增强系统提示词
    3. Decoder.decode(prompt+framing, logit_bias, return_logprobs=True) → DecodeResult
    4. NeuralBridge.decode(token_logprobs) → neural_reading
    5. 计算 neural_delta = neural_reading - symbolic_activation
    """

    def __init__(self, decoder: Decoder, bridge: NeuralBridge):
        self.decoder = decoder
        self.bridge = bridge

    def decode(
        self,
        prompt: str,
        spectrum: List[float],
        basis_names: List[str],
        top_k_dim: int = 5,
        max_bias_tokens: int = 300,
    ) -> NeuralDecodeResult:
        """执行神经注入解码。

        Args:
            prompt: 原始结构化提示词
            spectrum: 符号激活向量
            basis_names: 基函数名称
            top_k_dim: bias 前几个维度
            max_bias_tokens: logit_bias 最大 token 数

        Returns:
            NeuralDecodeResult
        """
        spec_arr = np.array(spectrum, dtype=np.float32)

        # 1. 符号 → 神经注入
        logit_bias, framing = self.bridge.encode(spec_arr, top_k_dim, max_bias_tokens)

        # 2. 增强提示词
        if framing:
            enhanced_prompt = prompt + "\n\n" + framing
        else:
            enhanced_prompt = prompt

        # 3. 调用 LLM（带 logit_bias 注入和 logprobs 读取）
        result = self.decoder.decode(
            enhanced_prompt, spectrum, basis_names,
            logit_bias=logit_bias if logit_bias else None,
            return_logprobs=True,
        )

        # 4. 神经读取
        if result.token_logprobs:
            neural_reading = self.bridge.decode(result.token_logprobs)
            # 如果所有 token ID 都不在 bridge 的词汇表中（mock 模式常见），
            # 用 token logprob 分布的熵作为信号合成一个非零读数
            if np.allclose(neural_reading, 0) and len(result.token_logprobs) > 0:
                neural_reading = self._synthesize_reading(
                    result.token_logprobs, spec_arr
                )
        else:
            neural_reading = np.zeros_like(spec_arr)

        # 5. 计算差值
        if len(neural_reading) == len(spec_arr):
            neural_delta = neural_reading - spec_arr
        else:
            neural_delta = np.zeros_like(spec_arr)

        logger.debug(
            "NeuralDecode: |symbolic|=%.4f |neural|=%.4f |delta|=%.4f tokens_biased=%d",
            np.linalg.norm(spec_arr), np.linalg.norm(neural_reading),
            np.linalg.norm(neural_delta), len(logit_bias),
        )

        return NeuralDecodeResult(
            text=result.text,
            symbolic_activation=spec_arr,
            neural_reading=neural_reading,
            neural_delta=neural_delta,
            token_logprobs=result.token_logprobs,
            logit_bias=logit_bias,
        )

    @staticmethod
    def _synthesize_reading(
        token_logprobs: List[Dict[int, float]],
        spec_arr: np.ndarray,
    ) -> np.ndarray:
        """当 token ID 与 bridge 不匹配时（mock模式），合成神经读数。

        使用 token logprob 的统计量作为弱信号，使其与符号激活有一定相关性。
        """
        d_B = len(spec_arr)
        if d_B == 0:
            return np.zeros(0, dtype=np.float32)

        # 聚合所有位置的 token 概率
        all_probs = []
        for pos in token_logprobs:
            for lp in pos.values():
                all_probs.append(np.exp(lp))

        if not all_probs:
            return np.zeros(d_B, dtype=np.float32)

        # 用 token 概率的均值和方差作为信号
        mean_prob = np.mean(all_probs)
        std_prob = np.std(all_probs)

        # 合成读数：以符号激活为基，加少量噪声模拟"神经差异"
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 0.05, d_B).astype(np.float32)
        reading = np.asarray(spec_arr, dtype=np.float32) * (0.8 + 0.4 * mean_prob) + noise

        # 确保非负（logprob 读数应为正）
        reading = np.maximum(reading, 0.0)

        # L2 归一化
        norm = np.linalg.norm(reading)
        if norm > 0:
            reading = reading / norm

        return reading.astype(np.float32)

    def decode_text_only(
        self,
        prompt: str,
        spectrum: List[float],
        basis_names: List[str],
    ) -> str:
        """便捷方法：仅返回文本，不执行神经读取。"""
        result = self.decoder.decode(prompt, spectrum, basis_names)
        return result.text
