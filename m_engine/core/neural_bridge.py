"""
NeuralBridge: 符号空间 ↔ 神经空间 双向投影桥。

将 10 个可解释基函数维度映射到 LLM 的 token 词汇空间，
通过 logit_bias（注入）和 logprobs（读取）实现双空间同步。

方向：
  encode: 符号频谱 → token bias + framing text（注入 LLM 生成）
  decode: token logprobs → 神经读数（从 LLM 响应中读取）
  update: 赫布式学习更新双向投影矩阵
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_registry import BasisRegistry

logger = logging.getLogger(__name__)

# ---- 基函数 → 中文种子关键词映射 ----
# 每个基函数 15-30 个关键词，由人工定义以保证可解释性
SIGNATURE_SEEDS: Dict[str, List[str]] = {
    "basis_causality": [
        "因为", "所以", "因此", "导致", "由于", "造成", "引起",
        "于是", "因而", "故而", "之所以", "是因为", "结果",
        "后果", "影响", "引发", "触发", "使得", "促使", "驱使",
    ],
    "basis_temporal": [
        "首先", "然后", "之后", "接着", "最后", "之前", "当时",
        "现在", "未来", "过去", "后来", "随后", "最终", "起初",
        "一开始", "紧接着", "随即", "此后", "同时", "期间",
    ],
    "basis_emotion": [
        "快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "喜悦",
        "焦虑", "感动", "痛苦", "高兴", "难过", "生气", "害怕",
        "开心", "伤心", "担忧", "激动", "沮丧", "欣慰", "委屈",
        "后悔", "兴奋", "失望", "紧张",
    ],
    "basis_motivation": [
        "想", "希望", "打算", "意图", "目的", "为了", "追求",
        "渴望", "期待", "企图", "动机", "目标", "愿望", "企图",
        "设法", "试图", "努力", "争取", "计划", "决心", "志向",
    ],
    "basis_spatial": [
        "上面", "下面", "左边", "右边", "里面", "外面", "前面",
        "后面", "旁边", "中间", "远处", "近处", "周围", "附近",
        "这里", "那里", "对面", "隔壁", "楼上", "楼下",
    ],
    "basis_logic": [
        "矛盾", "一致", "逻辑", "合理", "不合理", "必然", "可能",
        "不可能", "显然", "当然", "必定", "未必", "也许", "或许",
        "应当", "按理", "按理说", "自然", "势必", "未必",
    ],
    "basis_intent": [
        "猜测", "推测", "估计", "大概", "也许", "可能", "暗示",
        "意味", "暗含", "隐喻", "言外之意", "弦外之音", "潜台词",
        "似乎", "好像", "看来", "想必", "莫非", "八成", "多半",
    ],
    "basis_social": [
        "朋友", "家人", "同事", "同学", "邻居", "亲戚", "伴侣",
        "上司", "下属", "同伴", "盟友", "对手", "敌人", "陌生人",
        "关系", "交情", "往来", "相处", "合作", "帮助", "支持",
    ],
    "basis_contrast": [
        "不同", "区别", "差异", "对比", "比较", "相反", "反而",
        "但是", "然而", "可是", "不过", "却", "虽然", "尽管",
        "比起", "相对于", "截然", "迥异", "相似", "同样",
    ],
    "basis_moral": [
        "应该", "不应该", "正确", "错误", "对", "错", "道德",
        "正义", "公正", "公平", "善良", "邪恶", "好坏", "善恶",
        "对得起", "对不起", "愧疚", "责备", "谴责", "赞扬", "褒贬",
    ],
}


class NeuralBridge:
    """符号空间 ↔ 神经 token 空间 双向投影桥。

    P_decode (d_B × n_tokens): 频谱强度 → token bias 大小
    P_encode (n_tokens × d_B): token logprob → 频谱读数
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.d_B = 0
        self.n_tokens = 0

        # 投影矩阵（延迟初始化）
        self.P_decode: Optional[np.ndarray] = None
        self.P_encode: Optional[np.ndarray] = None

        # token 映射
        self.token_ids: List[int] = []
        self.token_strs: List[str] = []
        self._token_to_idx: Dict[int, int] = {}

        # 每个基函数的 token 签名集
        self.basis_signatures: Dict[str, List[int]] = {}

        self._initialized = False

    def build_signatures(self, basis_registry: BasisRegistry) -> int:
        """根据基函数定义和种子关键词构建 token 签名表。

        Args:
            basis_registry: 基函数注册表

        Returns:
            采集到的 token 总数
        """
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model_name)
        except ImportError:
            logger.warning("tiktoken not available, using fallback token IDs")
            return self._build_signatures_fallback(basis_registry)
        except KeyError:
            logger.warning("Model %s not recognized by tiktoken, using cl100k_base",
                         self.model_name)
            enc = tiktoken.get_encoding("cl100k_base")

        basis_list = basis_registry.list_all()
        self.d_B = len(basis_list)

        all_token_ids: set = set()
        self.basis_signatures = {}

        for bf in basis_list:
            keywords = SIGNATURE_SEEDS.get(bf.id, [bf.name])
            sig_tokens = []
            for kw in keywords:
                ids = enc.encode(kw)
                sig_tokens.extend(ids)
                all_token_ids.update(ids)
            self.basis_signatures[bf.id] = list(set(sig_tokens))

        self.token_ids = sorted(all_token_ids)
        self.n_tokens = len(self.token_ids)
        self._token_to_idx = {tid: i for i, tid in enumerate(self.token_ids)}

        # 尝试解码 token 为字符串（用于调试）
        try:
            self.token_strs = [enc.decode([tid]) for tid in self.token_ids]
        except Exception:
            self.token_strs = [str(tid) for tid in self.token_ids]

        self._init_matrices()
        self._initialized = True
        logger.info("NeuralBridge built: d_B=%d n_tokens=%d model=%s",
                    self.d_B, self.n_tokens, self.model_name)
        return self.n_tokens

    def _build_signatures_fallback(self, basis_registry: BasisRegistry) -> int:
        """无 tiktoken 时的回退方案：用关键词的 hash 作为伪 token ID。"""
        basis_list = basis_registry.list_all()
        self.d_B = len(basis_list)

        # 为每个关键词分配一个伪 token ID（hash 取模到 0~2000 范围）
        all_ids: set = set()
        self.basis_signatures = {}
        for bf in basis_list:
            keywords = SIGNATURE_SEEDS.get(bf.id, [bf.name])
            sig_ids = []
            for kw in keywords:
                tid = abs(hash(kw)) % 2000
                sig_ids.append(tid)
                all_ids.add(tid)
            self.basis_signatures[bf.id] = list(set(sig_ids))

        self.token_ids = sorted(all_ids)
        self.n_tokens = len(self.token_ids)
        self._token_to_idx = {tid: i for i, tid in enumerate(self.token_ids)}
        self.token_strs = [f"tok_{tid}" for tid in self.token_ids]

        self._init_matrices()
        self._initialized = True
        logger.warning("NeuralBridge: using hash-based fallback tokens (n=%d)", self.n_tokens)
        return self.n_tokens

    def _init_matrices(self):
        """初始化投影矩阵。P_decode 近 0，P_encode 近 0。"""
        rng = np.random.RandomState(42)
        sigma = 0.01 / np.sqrt(self.n_tokens + 1)
        self.P_decode = rng.normal(0, sigma, (self.d_B, self.n_tokens)).astype(np.float32)
        self.P_encode = rng.normal(0, sigma, (self.n_tokens, self.d_B)).astype(np.float32)

    # ---- ENCODE: 符号频谱 → neural injection ----

    def encode(
        self,
        spectrum: np.ndarray,
        top_k_dim: int = 5,
        max_tokens: int = 300,
    ) -> Tuple[Dict[int, float], str]:
        """符号频谱 → logit_bias dict + 维度聚焦提示文本。

        Args:
            spectrum: (d_B,) 基函数激活向量
            top_k_dim: 只 bias 前 k 个激活维度
            max_tokens: logit_bias 最多包含的 token 数（OpenAI 上限 300）

        Returns:
            (logit_bias_dict, framing_text)
        """
        if not self._initialized or self.P_decode is None:
            return {}, ""

        spec = self._align_vector(spectrum)
        P = self.P_decode

        # 每个维度对每个 token 的 bias = spectrum[d] * P_decode[d, token]
        # token 总 bias = sum over dimensions
        token_bias = np.zeros(self.n_tokens, dtype=np.float32)
        for d in range(self.d_B):
            if d < len(spec):
                token_bias += spec[d] * P[d]

        # 选绝对值最大的 max_tokens 个 token
        if self.n_tokens > max_tokens:
            abs_bias = np.abs(token_bias)
            top_indices = np.argpartition(-abs_bias, max_tokens - 1)[:max_tokens]
        else:
            top_indices = np.arange(self.n_tokens)

        # 构造 logit_bias dict，裁剪到 [-100, 100]
        logit_bias: Dict[int, float] = {}
        for idx in top_indices:
            val = float(token_bias[idx])
            if abs(val) > 0.001:
                val_clipped = max(-100.0, min(100.0, val * 10.0))
                logit_bias[self.token_ids[idx]] = val_clipped

        # 构造聚焦提示文本
        framing = self._build_framing(spectrum, top_k_dim)

        return logit_bias, framing

    def _build_framing(self, spectrum: np.ndarray, top_k: int) -> str:
        """构造维度聚焦提示文本。"""
        spec = self._align_vector(spectrum)
        indices = np.argsort(-spec)[:top_k]
        lines = []
        for i in indices:
            if i < self.d_B and spec[i] > 0.001:
                lines.append(f"- 维度 {i}: 强度 {spec[i]:.4f}")
        if not lines:
            return ""
        return "【神经调制】当前生成倾向于以下认知维度：\n" + "\n".join(lines)

    # ---- DECODE: token logprobs → 神经读数 ----

    def decode(self, token_logprobs: List[Dict[int, float]]) -> np.ndarray:
        """LLM 返回的 token-level logprobs → 神经基函数读数。

        Args:
            token_logprobs: LLM 响应中每个位置 top token 的 {token_id: logprob}
                           列表，长度为生成的 token 数

        Returns:
            neural_reading: (d_B,) 归一化向量，表示 LLM 实际激活了哪些维度
        """
        if not self._initialized or self.P_encode is None:
            return np.zeros(self.d_B, dtype=np.float32) if self.d_B > 0 else np.zeros(0)

        if not token_logprobs:
            return np.zeros(self.d_B, dtype=np.float32)

        # 聚合所有位置的 token logprobs → 加权 token 向量
        token_weights = np.zeros(self.n_tokens, dtype=np.float32)
        total_weight = 0.0
        for pos_probs in token_logprobs:
            for tid, lp in pos_probs.items():
                idx = self._token_to_idx.get(tid)
                if idx is not None:
                    # logprob 是负值，用 exp 转为概率
                    prob = np.exp(lp)
                    token_weights[idx] += prob
                    total_weight += prob

        if total_weight > 0:
            token_weights /= total_weight

        # 投影回基函数空间
        neural = self.P_encode.T @ token_weights

        # L2 归一化
        norm = np.linalg.norm(neural)
        if norm > 0:
            neural = neural / norm

        return neural.astype(np.float32)

    # ---- UPDATE: 赫布式学习 ----

    def update(
        self,
        symbolic_activation: np.ndarray,
        neural_reading: np.ndarray,
        token_logprobs: List[Dict[int, float]],
        feedback: float,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """赫布式更新双向投影矩阵。

        正反馈：强化当前 symbolic↔neural 的共现模式
        负反馈：削弱

        Args:
            symbolic_activation: 符号系统计算出的激活向量 (d_B,)
            neural_reading: 从 LLM logprobs 读出的神经向量 (d_B,)
            token_logprobs: 原始 token logprobs（用于更精确的更新）
            feedback: ±1.0
            lr: 学习率

        Returns:
            {"delta_P_decode": norm, "delta_P_encode": norm}
        """
        if not self._initialized or self.P_decode is None:
            return {"delta_P_decode": 0.0, "delta_P_encode": 0.0}

        s = self._align_vector(symbolic_activation)
        n = self._align_vector(neural_reading)

        s_norm = s / (np.linalg.norm(s) + 1e-8)
        n_norm = n / (np.linalg.norm(n) + 1e-8)

        # 聚合 token 权重
        token_w = np.zeros(self.n_tokens, dtype=np.float32)
        for pos_probs in token_logprobs:
            for tid, lp in pos_probs.items():
                idx = self._token_to_idx.get(tid)
                if idx is not None:
                    token_w[idx] += np.exp(lp)
        t_norm = token_w / (np.linalg.norm(token_w) + 1e-8)

        # P_decode: spectrum → token — 学习哪个 token 响应哪个维度
        delta_Pd = lr * feedback * np.outer(s_norm, t_norm)
        self.P_decode += delta_Pd

        # P_encode: token → spectrum — 学习哪个 token 预示哪个维度
        delta_Pe = lr * feedback * np.outer(t_norm, s_norm)
        self.P_encode += delta_Pe

        # 裁剪
        self.P_decode = np.clip(self.P_decode, -1.0, 1.0)
        self.P_encode = np.clip(self.P_encode, -1.0, 1.0)

        dp = float(np.linalg.norm(delta_Pd))
        de = float(np.linalg.norm(delta_Pe))
        logger.debug("NeuralBridge update: dPd=%.4f dPe=%.4f feedback=%.2f", dp, de, feedback)
        return {"delta_P_decode": dp, "delta_P_encode": de}

    # ---- 工具方法 ----

    @staticmethod
    def _align_vector(v: np.ndarray, target_len: int = 0) -> np.ndarray:
        """不对齐时的对齐方法（encode/decode/update 内部使用 self.d_B）。"""
        return v  # 调用方自行处理

    def _align_vector(self, v: np.ndarray) -> np.ndarray:
        """将向量对齐到 self.d_B。"""
        arr = np.asarray(v, dtype=np.float32).flatten()
        if len(arr) == self.d_B:
            return arr
        if len(arr) < self.d_B:
            return np.pad(arr, (0, self.d_B - len(arr)))
        return arr[:self.d_B]
