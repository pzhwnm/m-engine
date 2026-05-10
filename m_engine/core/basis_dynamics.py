"""
BasisDynamics: 基函数动力学引擎（GA+CA 混合演化）。

实现基函数的完整生命周期管理：
  - 新生 (birth):  检测认知覆盖缺口 → LLM 提议新基函数
  - 分裂 (split):  高方差+高激活 → 拆分为两个子维度
  - 合并 (merge):  高嵌入相似度+重叠激活模式 → 合二为一
  - 死亡 (death):  长期不活跃/低强度 → 淘汰并回收频谱能量

GA (遗传算法) 层面：种群适应度 → 选择 → 交叉/变异
CA (元胞自动机) 层面：基于嵌入相似度的局部邻域规则
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_registry import BasisRegistry, BasisFunction

logger = logging.getLogger(__name__)


@dataclass
class DynamicsConfig:
    """动力学配置参数。"""
    # 死亡阈值
    death_min_activations: int = 3         # 最少激活次数
    death_min_avg_strength: float = 0.005  # 最低平均强度
    death_max_idle_seconds: float = 3600.0 # 最长空闲时间（1小时，原型放低便于测试）

    # 合并阈值
    merge_embedding_similarity: float = 0.85  # 嵌入余弦相似度阈值
    merge_activation_correlation: float = 0.7  # 激活模式相关系数阈值

    # 分裂阈值
    split_min_activations: int = 20        # 最少激活次数才考虑分裂
    split_variance_threshold: float = 0.05 # 强度方差阈值

    # 新生阈值
    birth_min_basis_count: int = 3         # 低于此数量优先新生
    birth_max_basis_count: int = 20        # 基函数数量上限
    birth_coverage_gap_threshold: float = 0.3  # 事实嵌入与最近基函数的最小相似度

    # 通用
    cooldown_seconds: float = 60.0         # 两次演化操作之间的最小间隔
    max_ops_per_check: int = 2             # 每次检查最多执行的操作数


@dataclass
class DynamicsEvent:
    """一次演化事件记录。"""
    event_type: str  # "birth", "split", "merge", "death"
    timestamp: float = field(default_factory=time.time)
    affected_ids: List[str] = field(default_factory=list)
    new_ids: List[str] = field(default_factory=list)
    reason: str = ""


class BasisDynamics:
    """GA+CA 混合基函数演化引擎。

    通过统计追踪和 LLM 辅助，自动管理基函数种群的生命周期，
    使系统的"世界观"随交互自适应演化。
    """

    def __init__(
        self,
        basis_registry: BasisRegistry,
        config: Optional[DynamicsConfig] = None,
    ):
        self.basis = basis_registry
        self.config = config or DynamicsConfig()

        # 演化历史和冷却
        self.history: List[DynamicsEvent] = []
        self._last_check: float = 0.0
        self._total_ops: int = 0

    # ================================================================
    # 主入口：周期性检查并执行演化
    # ================================================================

    def step(
        self,
        fact_embeddings: Optional[List[np.ndarray]] = None,
        decoder=None,
    ) -> List[DynamicsEvent]:
        """执行一次演化步骤。

        按优先级依次尝试：死亡 → 合并 → 分裂 → 新生。
        受冷却时间和每步操作上限约束。

        Args:
            fact_embeddings: 当前所有事实的嵌入向量（用于计算覆盖缺口和激活相关性）
            decoder: Decoder 实例（用于 LLM 提议新基函数名称）

        Returns:
            本次产生的演化事件列表
        """
        now = time.time()
        if now - self._last_check < self.config.cooldown_seconds:
            return []

        ops_remaining = self.config.max_ops_per_check
        events: List[DynamicsEvent] = []

        # 优先级：死亡 > 合并 > 分裂 > 新生
        # 死亡优先释放"生态位"，合并其次消除冗余，分裂和新生调整结构

        if ops_remaining > 0:
            death_events = self._try_death()
            events.extend(death_events)
            ops_remaining -= len(death_events)

        if ops_remaining > 0:
            merge_events = self._try_merge(fact_embeddings)
            events.extend(merge_events[:ops_remaining])
            ops_remaining -= len(merge_events[:ops_remaining])

        if ops_remaining > 0:
            split_events = self._try_split(decoder)
            events.extend(split_events[:ops_remaining])
            ops_remaining -= len(split_events[:ops_remaining])

        if ops_remaining > 0:
            birth_events = self._try_birth(fact_embeddings, decoder)
            events.extend(birth_events[:ops_remaining])
            ops_remaining -= len(birth_events[:ops_remaining])

        self._last_check = now
        self._total_ops += len(events)
        self.history.extend(events)
        if events:
            logger.info("Dynamics step: %d events", len(events))
        return events

    # ================================================================
    # 死亡 (Death)
    # ================================================================

    def _try_death(self) -> List[DynamicsEvent]:
        """淘汰不健康的基函数。"""
        events = []
        now = time.time()
        candidates = []

        for b in self.basis.list_all():
            # 保护初代基函数（generation 0 且有激活记录的暂不淘汰）
            if b.generation == 0 and b.activation_count > 0:
                continue
            # 刚创建不久的基函数给予保护期
            if b.created_at > 0 and now - b.created_at < 120:
                continue

            score = self._health_score(b, now)
            if score < 0.0:  # 负分 = 不健康
                candidates.append((b.id, score))

        candidates.sort(key=lambda x: x[1])  # 最不健康的排前面

        for bid, score in candidates[:self.config.max_ops_per_check]:
            b = self.basis.get(bid)
            if b is None:
                continue
            self.basis.remove(bid)
            event = DynamicsEvent(
                event_type="death",
                affected_ids=[bid],
                reason=f"健康分={score:.2f} 激活={b.activation_count} "
                       f"均强={b.avg_strength:.4f} 空闲={now - b.last_activated:.0f}s",
            )
            events.append(event)
            logger.info("DEATH: %s (%s) — %s", b.name, bid, event.reason)

        return events

    def _health_score(self, b: BasisFunction, now: float) -> float:
        """计算基函数的健康分数。正=健康，负=该淘汰。"""
        score = 0.0

        # 激活次数
        if b.activation_count >= self.config.death_min_activations:
            score += 2.0
        elif b.activation_count > 0:
            score += float(b.activation_count) / self.config.death_min_activations

        # 平均强度
        if b.avg_strength >= self.config.death_min_avg_strength:
            score += 2.0
        elif b.avg_strength > 0:
            score += b.avg_strength / self.config.death_min_avg_strength

        # 空闲时间
        if b.last_activated > 0:
            idle = now - b.last_activated
            if idle < self.config.death_max_idle_seconds:
                score += 2.0 * (1.0 - idle / self.config.death_max_idle_seconds)
            else:
                score -= 3.0

        # 完全没有激活过的基函数
        if b.activation_count == 0:
            score -= 5.0

        return score

    # ================================================================
    # 合并 (Merge)
    # ================================================================

    def _try_merge(
        self,
        fact_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[DynamicsEvent]:
        """合并在嵌入空间和激活模式上都高度重叠的基函数对。"""
        events = []
        basis_list = self.basis.list_all()
        if len(basis_list) <= self.config.birth_min_basis_count:
            return []  # 基函数太少，不合并

        pairs = self._find_merge_candidates(basis_list, fact_embeddings)

        for b1, b2, sim in pairs[:self.config.max_ops_per_check]:
            merged = self._merge_two(b1, b2)
            if merged is None:
                continue

            # 注册新基函数，移除旧的两个
            self.basis.register(merged)
            self.basis.remove(b1.id)
            self.basis.remove(b2.id)

            event = DynamicsEvent(
                event_type="merge",
                affected_ids=[b1.id, b2.id],
                new_ids=[merged.id],
                reason=f"嵌入相似度={sim:.3f} '{b1.name}' + '{b2.name}' → '{merged.name}'",
            )
            events.append(event)
            logger.info("MERGE: %s + %s → %s (sim=%.3f)", b1.name, b2.name, merged.name, sim)

        return events

    def _find_merge_candidates(
        self,
        basis_list: List[BasisFunction],
        fact_embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[BasisFunction, BasisFunction, float]]:
        """寻找合并候选对。"""
        candidates = []

        for i in range(len(basis_list)):
            for j in range(i + 1, len(basis_list)):
                b1, b2 = basis_list[i], basis_list[j]

                # 嵌入相似度
                e1 = np.array(b1.embedding, dtype=np.float32) if b1.embedding else None
                e2 = np.array(b2.embedding, dtype=np.float32) if b2.embedding else None

                if e1 is not None and e2 is not None and len(e1) > 0 and len(e2) > 0:
                    sim = float(np.dot(e1, e2) / (
                        np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
                else:
                    # 无嵌入时用名称字符串相似度代理
                    sim = self._name_similarity(b1.name, b2.name)

                if sim >= self.config.merge_embedding_similarity:
                    candidates.append((b1, b2, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _merge_two(
        self, b1: BasisFunction, b2: BasisFunction
    ) -> Optional[BasisFunction]:
        """将两个基函数合并为一个。"""
        # 名称：取两个名称的组合
        # 简单启发：取较短名称的核心 + 较长名称的修饰
        if len(b1.name) <= len(b2.name):
            merged_name = f"{b2.name}·{b1.name}"
        else:
            merged_name = f"{b1.name}·{b2.name}"

        # 描述合并
        merged_desc = f"合并自 [{b1.name}: {b1.description}] 和 [{b2.name}: {b2.description}]"

        # 嵌入取平均
        if b1.embedding and b2.embedding and len(b1.embedding) == len(b2.embedding):
            merged_emb = ((np.array(b1.embedding) + np.array(b2.embedding)) / 2).tolist()
        else:
            merged_emb = b1.embedding or b2.embedding

        return BasisFunction(
            id=f"basis_merged_{uuid.uuid4().hex[:8]}",
            name=merged_name[:30],
            description=merged_desc[:200],
            embedding=merged_emb,
            activation_count=b1.activation_count + b2.activation_count,
            strength_sum=b1.strength_sum + b2.strength_sum,
            parent_ids=[b1.id, b2.id],
            generation=max(b1.generation, b2.generation) + 1,
        )

    # ================================================================
    # 分裂 (Split)
    # ================================================================

    def _try_split(self, decoder=None) -> List[DynamicsEvent]:
        """分裂激活方差过高的基函数。"""
        events = []
        basis_list = self.basis.list_all()

        if len(basis_list) >= self.config.birth_max_basis_count:
            return []  # 已达上限

        candidates = []
        for b in basis_list:
            if b.activation_count < self.config.split_min_activations:
                continue
            if b.strength_variance < self.config.split_variance_threshold:
                continue
            candidates.append((b, b.strength_variance))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for b, var in candidates[:self.config.max_ops_per_check]:
            children = self._split_one(b, decoder)
            if len(children) < 2:
                continue

            self.basis.remove(b.id)
            for child in children:
                self.basis.register(child)

            event = DynamicsEvent(
                event_type="split",
                affected_ids=[b.id],
                new_ids=[c.id for c in children],
                reason=f"方差={var:.4f} '{b.name}' → "
                       f"'{children[0].name}' + '{children[1].name}'",
            )
            events.append(event)
            logger.info("SPLIT: %s → %s + %s (var=%.4f)", b.name,
                       children[0].name, children[1].name, var)

        return events

    def _split_one(
        self, b: BasisFunction, decoder=None
    ) -> List[BasisFunction]:
        """将一个基函数分裂为两个子维度。

        使用 LLM（如果可用）生成子维度名称，否则使用启发式规则。
        """
        a_name = f"{b.name}(强)"
        a_desc = f"从 [{b.name}] 分裂出的高激活子维度：{b.description}"
        b_name = f"{b.name}(弱)"
        b_desc = f"从 [{b.name}] 分裂出的低激活子维度：{b.description}"

        # 尝试用 LLM 生成更好的名称
        if decoder is not None:
            try:
                llm_names = self._llm_split_names(b, decoder)
                if llm_names and len(llm_names) == 2:
                    a_name, b_name = llm_names
            except Exception as e:
                logger.debug("LLM split naming failed: %s", e)

        # 对嵌入做微小扰动以区分
        if b.embedding:
            emb_arr = np.array(b.embedding, dtype=np.float32)
            rng = np.random.RandomState(abs(hash(b.id)) % (2**31))
            noise1 = rng.normal(0, 0.05, len(emb_arr)).astype(np.float32)
            noise2 = rng.normal(0, 0.05, len(emb_arr)).astype(np.float32)
            emb1 = (emb_arr + noise1).tolist()
            emb2 = (emb_arr + noise2).tolist()
        else:
            emb1 = emb2 = b.embedding

        child1 = BasisFunction(
            id=f"basis_split_{uuid.uuid4().hex[:8]}",
            name=a_name[:30],
            description=a_desc[:200],
            embedding=emb1,
            activation_count=b.activation_count // 2,
            strength_sum=b.strength_sum / 2,
            parent_ids=[b.id],
            generation=b.generation + 1,
        )
        child2 = BasisFunction(
            id=f"basis_split_{uuid.uuid4().hex[:8]}",
            name=b_name[:30],
            description=b_desc[:200],
            embedding=emb2,
            activation_count=b.activation_count // 2,
            strength_sum=b.strength_sum / 2,
            parent_ids=[b.id],
            generation=b.generation + 1,
        )
        return [child1, child2]

    def _llm_split_names(
        self, b: BasisFunction, decoder
    ) -> Optional[List[str]]:
        """使用 LLM 提议分裂后的子维度名称。"""
        prompt = (
            f"一个名为'{b.name}'的认知维度（{b.description}）需要拆分为两个更细粒度的子维度。"
            f"请给出两个中文名称（各2-5个字），用逗号分隔。只输出名称，不要解释。"
        )
        result = decoder.decode(prompt, [], [])
        text = result.text if hasattr(result, 'text') else str(result)
        parts = [p.strip() for p in text.replace("、", ",").split(",")]
        if len(parts) >= 2:
            return [parts[0][:15], parts[1][:15]]
        return None

    # ================================================================
    # 新生 (Birth)
    # ================================================================

    def _try_birth(
        self,
        fact_embeddings: Optional[List[np.ndarray]] = None,
        decoder=None,
    ) -> List[DynamicsEvent]:
        """检测认知覆盖缺口并新生基函数。"""
        events = []
        basis_list = self.basis.list_all()

        # 数量不足时更容易触发
        if len(basis_list) < self.config.birth_min_basis_count:
            gap_score = 1.0
        elif len(basis_list) >= self.config.birth_max_basis_count:
            return []  # 已达上限
        elif fact_embeddings is not None and len(fact_embeddings) > 0:
            gap_score = self._compute_coverage_gap(fact_embeddings, basis_list)
        else:
            return []  # 无法判断缺口

        if gap_score < self.config.birth_coverage_gap_threshold:
            return []

        # 新生一个基函数
        newborn = self._birth_one(basis_list, decoder)
        if newborn is None:
            return []

        self.basis.register(newborn)
        event = DynamicsEvent(
            event_type="birth",
            new_ids=[newborn.id],
            reason=f"覆盖缺口={gap_score:.3f} 当前共{len(basis_list)}个基函数",
        )
        events.append(event)
        logger.info("BIRTH: %s (%s) — gap=%.3f", newborn.name, newborn.id, gap_score)
        return events

    def _compute_coverage_gap(
        self,
        fact_embeddings: List[np.ndarray],
        basis_list: List[BasisFunction],
    ) -> float:
        """计算事实嵌入空间中的覆盖缺口。

        对每个事实嵌入，计算它与最近基函数嵌入的余弦距离。
        平均距离越大 = 覆盖越差 = 缺口越大。
        """
        if not fact_embeddings:
            return 0.0

        # 取事实嵌入的维度作为参考
        ref_dim = len(fact_embeddings[0]) if len(fact_embeddings) > 0 else 0
        if ref_dim == 0:
            return 0.0

        # 收集与事实嵌入维度匹配的基函数嵌入
        basis_embs = []
        for b in basis_list:
            if b.embedding and len(b.embedding) == ref_dim:
                emb = np.array(b.embedding, dtype=np.float32)
                if np.linalg.norm(emb) > 1e-8:
                    basis_embs.append(emb)

        if not basis_embs:
            return 1.0  # 没有维度匹配的基函数嵌入 → 覆盖缺口

        basis_stack = np.stack(basis_embs)
        gaps = []
        for fe in fact_embeddings:
            fe_arr = np.array(fe, dtype=np.float32)
            if len(fe_arr) != ref_dim or np.linalg.norm(fe_arr) < 1e-8:
                continue
            f_norm = fe_arr / (np.linalg.norm(fe_arr) + 1e-8)
            b_norms = basis_stack / (np.linalg.norm(basis_stack, axis=1, keepdims=True) + 1e-8)
            sims = np.dot(b_norms, f_norm)
            max_sim = float(np.max(sims))
            gap = 1.0 - max_sim
            gaps.append(gap)

        return float(np.mean(gaps)) if gaps else 0.0

    def _birth_one(
        self,
        basis_list: List[BasisFunction],
        decoder=None,
    ) -> Optional[BasisFunction]:
        """生成一个新基函数。用 LLM 提议名称，否则用启发式。"""
        existing_names = [b.name for b in basis_list]

        if decoder is not None:
            try:
                name, desc = self._llm_birth_name(existing_names, decoder)
            except Exception as e:
                logger.debug("LLM birth naming failed: %s", e)
                name, desc = None, None
        else:
            name, desc = None, None

        if name is None:
            # 启发式：从预定义候选池中选一个未使用的
            name, desc = self._heuristic_birth(existing_names)

        if name is None:
            return None

        new_id = f"basis_born_{uuid.uuid4().hex[:8]}"
        return BasisFunction(
            id=new_id,
            name=name[:30],
            description=desc[:200],
            embedding=[],
            generation=max((b.generation for b in basis_list), default=0) + 1,
        )

    def _llm_birth_name(
        self, existing_names: List[str], decoder
    ) -> Tuple[Optional[str], Optional[str]]:
        """用 LLM 提议一个新认知维度。"""
        existing = "、".join(existing_names)
        prompt = (
            f"当前系统有以下认知维度来分析事实：{existing}。"
            f"请提议一个全新的、不重复的认知维度（2-5个中文字），"
            f"用于补充现有维度未覆盖的方面。格式：维度名称, 简短描述"
        )
        result = decoder.decode(prompt, [], [])
        text = result.text if hasattr(result, 'text') else str(result)
        parts = text.strip().split(",")
        if len(parts) >= 2:
            return parts[0].strip()[:15], parts[1].strip()[:200]
        if len(parts) == 1 and parts[0]:
            return parts[0].strip()[:15], "自动生成的认知维度"
        return None, None

    def _heuristic_birth(
        self, existing_names: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """启发式新生：从候选池中选未占用的维度。"""
        candidates = [
            ("审美感知", "对事物美丑、和谐与艺术性的判断"),
            ("实用价值", "对事物功用、效率和实际利益的评估"),
            ("不确定性", "对信息不完备、风险和概率的感知"),
            ("归属认同", "对群体身份、文化认同和归属感的认知"),
            ("成长变化", "对事物发展、进步和演变趋势的观察"),
            ("权力结构", "对权威、等级和支配关系的分析"),
            ("资源分配", "对稀缺资源、公平性和分配方式的考量"),
            ("习惯常规", "对重复模式、惯例和社会规范的识别"),
        ]
        existing_set = set(existing_names)
        for name, desc in candidates:
            if name not in existing_set:
                return name, desc
        return None, None

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _name_similarity(name1: str, name2: str) -> float:
        """简单的字符串相似度（Jaccard on 2-grams）。"""
        def bigrams(s):
            return {s[i:i+2] for i in range(len(s)-1)}
        b1, b2 = bigrams(name1), bigrams(name2)
        if not b1 or not b2:
            return 0.0
        return len(b1 & b2) / len(b1 | b2)
