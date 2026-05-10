"""
QuestionRouter: 问题路由器。
将用户自然语言查询解析为最匹配的 BaseQuestion 及其滤波权重向量。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BaseQuestion:
    """基问题：预定义的问句模板及其对基函数的默认滤波权重。

    filter_weights: basis_id -> weight (float, >=0)
        滤波权重决定了该问题倾向于从哪些基函数维度提取信息。
        权重越高，该基函数维度的激活越强，在最终回答中越突出。
    """
    id: str = ""
    template: str = ""
    filter_weights: Dict[str, float] = field(default_factory=dict)

    def get_weight_vector(self, basis_ids: List[str]) -> np.ndarray:
        """按基函数 ID 顺序返回权重向量。"""
        return np.array([self.filter_weights.get(bid, 0.0) for bid in basis_ids],
                        dtype=np.float32)


class QuestionRouter:
    """问题路由器。

    职责：
    1. 将用户自然语言查询匹配到最相似的基问题模板
    2. 返回基问题的滤波权重向量

    匹配策略（MVP阶段）：
    - 关键词匹配：根据问句中是否包含特定关键词（为什么、感受、关系等）
    - 嵌入相似度：如果提供了 embedder，计算问句向量与基问题模板向量的余弦相似度
    - 加权融合：关键词得分 + 相似度得分 → 最终匹配
    """

    # 关键词映射：中文关键词 -> 问题类型 ID
    KEYWORD_MAP = {
        "为什么": "q_why",
        "为何": "q_why",
        "原因": "q_why",
        "导致": "q_why",
        "感受": "q_how_feel",
        "感觉": "q_how_feel",
        "心情": "q_how_feel",
        "情绪": "q_how_feel",
        "觉得怎样": "q_how_feel",
        "接下来": "q_what_next",
        "然后": "q_what_next",
        "之后": "q_what_next",
        "后来": "q_what_next",
        "会发生": "q_what_next",
        "对吗": "q_moral",
        "应该": "q_moral",
        "对不对": "q_moral",
        "道德": "q_moral",
        "正义": "q_moral",
        "关系": "q_relationship",
        "之间": "q_relationship",
        "什么关系": "q_relationship",
        "不同": "q_compare",
        "区别": "q_compare",
        "差异": "q_compare",
        "对比": "q_compare",
        "比较": "q_compare",
        "在哪里": "q_where",
        "什么地方": "q_where",
        "位置": "q_where",
        "什么时候": "q_when",
        "何时": "q_when",
        "多久": "q_when",
        "想做什么": "q_intent",
        "意图": "q_intent",
        "目的": "q_intent",
        "打算": "q_intent",
        "想干什么": "q_intent",
    }

    def __init__(self):
        self._questions: Dict[str, BaseQuestion] = {}
        self._embedder = None

    def set_embedder(self, embedder) -> None:
        """设置嵌入函数，用于语义相似度匹配。"""
        self._embedder = embedder

    def register(self, q: BaseQuestion) -> None:
        """注册一个基问题模板。"""
        self._questions[q.id] = q

    def get(self, q_id: str) -> Optional[BaseQuestion]:
        return self._questions.get(q_id)

    def load_from_json(self, filepath: str) -> None:
        """从 JSON 文件批量加载基问题定义。"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            bq = BaseQuestion(
                id=item["id"],
                template=item["template"],
                filter_weights=item.get("filter_weights", {}),
            )
            self.register(bq)
        logger.info("Loaded %d base questions from %s", len(data), filepath)

    def parse(self, user_query: str) -> Tuple[BaseQuestion, List[float]]:
        """解析用户查询，返回最匹配的基问题及其滤波权重向量。

        返回值:
            (BaseQuestion, weight_vector): 基问题模板和归一化后的权重向量。
            权重向量按 BasisRegistry 中基函数的注册顺序排列。
            weight_vector 已经过 L1 归一化处理。
        """
        # 步骤 1：关键词匹配
        keyword_scores: Dict[str, float] = {}
        for keyword, q_type in self.KEYWORD_MAP.items():
            if keyword in user_query:
                keyword_scores[q_type] = keyword_scores.get(q_type, 0.0) + 1.0

        # 步骤 2：选择最佳匹配
        if keyword_scores:
            # 得分最高的基问题
            best_q_id = max(keyword_scores, key=keyword_scores.get)
            base_q = self._questions.get(best_q_id)
            logger.info("Query '%s' matched to %s via keywords", user_query, best_q_id)
        else:
            # 无关键词匹配，使用通用问题
            base_q = self._questions.get("q_general")
            logger.info("Query '%s' matched to general question", user_query)

        if base_q is None:
            # 兜底：返回空基问题
            logger.warning("No base question found, returning empty")
            return BaseQuestion(id="q_fallback", template="{query}"), []

        # 步骤 3：提取权重向量
        weights = list(base_q.filter_weights.values())

        # 归一化
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return base_q, weights

    def list_basis_order(self) -> List[str]:
        """获取基函数 ID 的列表，用于权重向量的维度对齐。
        从第一个注册的基问题中推断基函数顺序。
        """
        for q in self._questions.values():
            if q.filter_weights:
                return list(q.filter_weights.keys())
        return []
