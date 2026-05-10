"""
Decoder: 解码器，调用 LLM API 将结构化 prompt 转换为自然语言回答。
支持 logit_bias 神经注入和 logprobs 神经读取。
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    """LLM 解码结果。

    text: 生成的回答文本
    token_logprobs: 每个生成位置 top token 的 {token_id: logprob} 列表。
                   仅在 return_logprobs=True 时非 None。
    """
    text: str = ""
    token_logprobs: Optional[List[Dict[int, float]]] = None


class Decoder:
    """LLM 解码器。

    支持：
    - OpenAI chat API (gpt-4o-mini 默认)
    - logit_bias 神经注入
    - logprobs 神经读取
    - mock 回退模式
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def decode(
        self,
        prompt: str,
        spectrum: List[float],
        basis_names: List[str],
        logit_bias: Optional[Dict[int, float]] = None,
        return_logprobs: bool = False,
    ) -> DecodeResult:
        """将 M-代数输出解码为自然语言回答。

        Args:
            prompt: 结构化提示词
            spectrum: 激活强度向量
            basis_names: 基函数名称列表
            logit_bias: token ID → bias 值 dict（神经注入）
            return_logprobs: 是否返回 token 级 logprobs（神经读取）

        Returns:
            DecodeResult(text, token_logprobs)
        """
        if not self.api_key:
            logger.warning("No API key configured, returning mock response")
            return self._mock_decode(prompt, spectrum, basis_names, return_logprobs)

        try:
            return self._call_api(prompt, logit_bias, return_logprobs)
        except Exception as e:
            err_msg = str(e)
            # Ollama 等兼容 API 不支持 logit_bias/logprobs → 自动降级重试
            if logit_bias or return_logprobs:
                if "logit_bias" in err_msg or "logprobs" in err_msg or "unexpected" in err_msg.lower():
                    logger.info("Neural features not supported by API, retrying without")
                    try:
                        return self._call_api(prompt, None, False)
                    except Exception as e2:
                        logger.error("Fallback also failed: %s", e2)
                else:
                    logger.info("Retrying once...")
                    try:
                        return self._call_api(prompt, logit_bias, return_logprobs)
                    except Exception as e2:
                        logger.error("Retry failed: %s", e2)
                        try:
                            return self._call_api(prompt, None, False)
                        except Exception as e3:
                            logger.error("All retries failed: %s", e3)
            return DecodeResult(
                text=f"[LLM调用失败: {err_msg[:100]}]",
                token_logprobs=None,
            )

    def _call_api(
        self, prompt: str, logit_bias=None, return_logprobs: bool = False
    ) -> DecodeResult:
        """单次 API 调用。自动检测 Ollama vs OpenAI。"""
        # Ollama 原生 API（无 logit_bias/logprobs 支持）
        if self.base_url and "11434" in str(self.base_url):
            return self._call_ollama(prompt)

        # OpenAI 兼容 API
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个多维度记忆分析助手，用中文回答。"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if logit_bias:
            kwargs["logit_bias"] = logit_bias
        if return_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5

        response = client.chat.completions.create(**kwargs)
        answer = response.choices[0].message.content or ""
        logger.info("LLM response received (%d chars)", len(answer))

        token_logprobs = None
        if return_logprobs:
            token_logprobs = self._extract_logprobs(response)

        return DecodeResult(text=answer, token_logprobs=token_logprobs)

    def _call_ollama(self, prompt: str) -> DecodeResult:
        """Ollama 原生 /api/chat 端点（兼容 gemma4 等 thinking 模型）。"""
        import json as _json
        import urllib.request

        # gemma 系列模型对中文支持差，使用英文提示
        if "gemma" in self.model.lower():
            system_msg = "You are a multi-dimensional memory analysis assistant. Answer concisely in 2-5 sentences."
            user_msg = prompt
        else:
            system_msg = "你是一个多维度记忆分析助手，用中文回答。保持简洁，2-5句话。"
            user_msg = prompt

        body = _json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }).encode("utf-8")

        url = str(self.base_url).replace("/v1", "") + "/api/chat"
        if not url.startswith("http"):
            url = "http://localhost:11434/api/chat"

        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
        })

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                # /api/chat 返回 message.content
                msg = data.get("message", {})
                answer = msg.get("content", "") or data.get("response", "")
                logger.info("Ollama response received (%d chars)", len(answer))
                return DecodeResult(text=answer, token_logprobs=None)
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            return DecodeResult(text=f"[Ollama调用失败: {str(e)[:100]}]", token_logprobs=None)

    @staticmethod
    def _extract_logprobs(response) -> List[Dict[int, float]]:
        """从 OpenAI 响应中提取每位置的 top token logprobs。"""
        result = []
        try:
            logprob_content = response.choices[0].logprobs.content
            if logprob_content:
                for pos_data in logprob_content:
                    pos_probs = {}
                    for top_lp in pos_data.top_logprobs:
                        # token 可能是字符串，需要通过 tokenizer 获取 ID
                        # 这里用字节编码的 hash 作为近似 ID
                        token_str = top_lp.token
                        tid = abs(hash(token_str)) % 100000
                        pos_probs[tid] = top_lp.logprob
                    result.append(pos_probs)
        except Exception as e:
            logger.warning("Failed to extract logprobs: %s", e)
        return result

    def _mock_decode(
        self,
        prompt: str,
        spectrum: List[float],
        basis_names: List[str],
        return_logprobs: bool = False,
    ) -> DecodeResult:
        """无 API 时的模拟解码。合成与 spectrum 相关的 logprobs。"""
        spec = spectrum if spectrum else []
        top_indices = sorted(
            range(len(spec)), key=lambda i: spec[i], reverse=True
        )[:3]

        top_dims = []
        for i in top_indices:
            if i < len(spec) and i < len(basis_names) and spec[i] > 0.001:
                top_dims.append(f"{basis_names[i]}({spec[i]:.4f})")

        if not top_dims:
            return DecodeResult(
                text="（无显著激活维度，无法生成回答。请先存储相关事实。）",
                token_logprobs=None,
            )

        text = f"[模拟回答] 基于以下认知维度的分析：{'、'.join(top_dims)}。\n（设置 OPENAI_API_KEY 环境变量以启用真实 LLM 解码）"

        # 合成 mock logprobs：高激活维度获得人工高概率 token
        # token ID 范围与 NeuralBridge 的 hash 回退方案对齐 (0-2000)
        token_logprobs = None
        if return_logprobs:
            token_logprobs = []
            for i in top_indices[:3]:
                if i < len(spec) and spec[i] > 0:
                    mock_tokens = {}
                    # 使用维度名称的 hash 生成 token ID，与 bridge fallback 对齐
                    for j in range(5):
                        tid = abs(hash(f"{basis_names[i]}_signal_{j}")) % 2000
                        mock_tokens[tid] = np.log(max(0.05, 0.4 + 0.4 * spec[i]))
                    token_logprobs.append(mock_tokens)

        return DecodeResult(text=text, token_logprobs=token_logprobs if token_logprobs else None)

    def decode_text(
        self,
        prompt: str,
        spectrum: List[float],
        basis_names: List[str],
    ) -> str:
        """便捷方法：仅返回文本，兼容旧接口。"""
        return self.decode(prompt, spectrum, basis_names).text
