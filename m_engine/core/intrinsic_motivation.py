"""
IntrinsicMotivation: 内在动机引擎。

检测认知缺口（事实频谱中未被充分探索的维度），
自动生成探索性问题来填补这些缺口。

不是独立模块——它利用博弈内核中已有的 Novelty 收益项，
在其驱动下主动寻找信息增益最大的探索方向。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_registry import BasisRegistry
from .fact_bus import Fact

logger = logging.getLogger(__name__)


@dataclass
class ExplorationTarget:
    """一个认知缺口的探索目标。"""
    fact_id: str
    fact_text: str
    basis_name: str           # 缺口所在的基函数维度
    gap_score: float          # 0~1, 越大越需要探索
    proposed_question: str = ""


class IntrinsicMotivation:
    """内在动机引擎。

    利用博弈内核中的 Novelty 收益项，检测认知缺口并生成探索性问题。
    """

    def __init__(self, basis_registry: BasisRegistry, gap_threshold: float = 0.05):
        self.basis = basis_registry
        self.gap_threshold = gap_threshold
        self.exploration_history: List[ExplorationTarget] = []

    def detect_gaps(self, facts: List[Fact]) -> List[ExplorationTarget]:
        """扫描所有事实，检测频谱中的认知缺口。

        缺口 = 某事实在某基函数维度上的频谱值接近零
        （说明该维度从未被用来理解该事实）。

        Returns:
            按缺口严重程度排序的探索目标列表
        """
        targets = []
        basis_list = self.basis.list_all()

        for fact in facts:
            for b in basis_list:
                score = fact.spectrum.get(b.id, 0.0)
                if score < self.gap_threshold:
                    gap = self.gap_threshold - score
                    targets.append(ExplorationTarget(
                        fact_id=fact.id,
                        fact_text=fact.raw_text,
                        basis_name=b.name,
                        gap_score=gap / self.gap_threshold,  # 归一化
                    ))

        # 按缺口严重程度排序
        targets.sort(key=lambda t: t.gap_score, reverse=True)
        return targets

    def generate_questions(
        self,
        targets: List[ExplorationTarget],
        decoder=None,
        max_questions: int = 5,
    ) -> List[ExplorationTarget]:
        """为缺口生成探索性问题。

        使用 LLM 生成针对特定基函数维度的问题。
        无 LLM 时使用模板生成。
        """
        for target in targets[:max_questions]:
            if decoder is not None:
                target.proposed_question = self._llm_question(target, decoder)
            else:
                target.proposed_question = self._template_question(target)

        self.exploration_history.extend(targets[:max_questions])
        if len(self.exploration_history) > 200:
            self.exploration_history = self.exploration_history[-200:]

        return targets[:max_questions]

    def _llm_question(self, target: ExplorationTarget, decoder) -> str:
        """用 LLM 生成探索性问题。"""
        prompt = (
            f"有一段记忆：'{target.fact_text[:200]}'。"
            f"我们目前缺少从【{target.basis_name}】维度对它的理解。"
            f"请生成一个中文问题（15字以内），引导用户从这个维度展开思考。"
            f"只输出问题本身，不要解释。"
        )
        try:
            result = decoder.decode(prompt, [], [])
            text = result.text if hasattr(result, 'text') else str(result)
            return text.strip()[:50]
        except Exception:
            return self._template_question(target)

    def _template_question(self, target: ExplorationTarget) -> str:
        """模板生成探索性问题。"""
        templates = {
            "因果关系": "这件事的原因和后果分别是什么？",
            "情感色调": "当时你内心真正感受到的情绪是什么？",
            "角色动机": "是什么驱使他们做出这个决定？",
            "时间顺序": "这件事在时间线上处于什么位置？",
            "空间关系": "事件发生的环境和空间背景是怎样的？",
            "逻辑一致性": "这里面有没有自相矛盾的地方？",
            "意图推测": "对方真正的意图可能是什么？",
            "社会关系": "这涉及了哪些人之间的关系？",
            "对比差异": "和类似的事件相比，有什么不同？",
            "道德判断": "从道德角度看，这代表了什么？",
        }
        return templates.get(
            target.basis_name,
            f"从{target.basis_name}的角度来看，这意味着什么？"
        )

    def get_top_gaps(self, facts: List[Fact], top_k: int = 5) -> List[Dict]:
        """获取当前最紧迫的认知缺口摘要。"""
        targets = self.detect_gaps(facts)[:top_k]
        return [
            {
                "fact_id": t.fact_id,
                "fact_preview": t.fact_text[:60],
                "dimension": t.basis_name,
                "gap": round(t.gap_score, 3),
                "question": t.proposed_question or "(未生成)",
            }
            for t in targets
        ]
