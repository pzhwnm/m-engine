"""
EvolutionaryGameCore: 自适应认知博弈动力学内核 (ACGD Core)。

将 M-代数、基函数动力学、内在动机、第二序监控统一为
一个多层级演化博弈过程。基函数是参与博弈的策略个体，
所有系统行为都是复制者动态方程在不同边界条件下的必然解。

三层嵌套博弈：
  微观层 (秒~分钟): 事实频谱系数竞争 — 哪些基函数解释该事实
  中观层 (小时~天):   基函数种群策略竞争与生态位分化
  宏观层 (周~月):     系统安全、价值对齐与认知健康

唯一方程：
  dot{x_i} = x_i(pi_i - pi_bar) + sum_j(x_j Q_ji - x_i Q_ij) + mu * Mutation(x_i)

其中 pi_i = alpha*ReconQuality + beta*Novelty - gamma*SafetyRisk - delta*ResourceCost
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_registry import BasisRegistry, BasisFunction
from .fact_bus import Fact

logger = logging.getLogger(__name__)


# ================================================================
# 配置
# ================================================================

@dataclass
class GameConfig:
    """博弈内核配置。所有系数由元博弈动态调控。"""
    # 收益函数系数
    alpha_recon: float = 1.0       # 重构贡献权重
    beta_novelty: float = 0.3      # 探索价值权重
    gamma_safety: float = 0.5      # 安全代价权重
    delta_resource: float = 0.1    # 代谢消耗权重

    # 复制者动态
    micro_lr: float = 0.1          # 频谱更新学习率
    strategy_lr: float = 0.01      # 策略向量更新学习率

    # 网络演化
    network_lr: float = 0.005      # 交互网络 A_ij 更新率
    coop_threshold: float = 0.3    # 共激活阈值（高于此视为协同）
    competition_decay: float = 0.1 # 竞争衰减系数

    # 种群控制
    min_population: int = 3
    max_population: int = 20
    death_energy_threshold: float = -5.0
    mutation_rate: float = 0.05
    mutation_scale: float = 0.1
    crossover_prob: float = 0.3

    # 演化节奏
    meso_trigger_interval: int = 10  # 每 N 次交互触发一次中观演化
    macro_check_interval: int = 50   # 每 N 次交互检查宏观约束

    # 冷却
    cooldown_seconds: float = 30.0


# ================================================================
# 策略个体
# ================================================================

@dataclass
class StrategyAgent:
    """基函数作为博弈策略个体的运行时状态。

    每个基函数 b_i 有一个策略向量 s_i，表示它在基函数空间中
    '占据'的逻辑位置。这个向量决定它如何参与每次博弈。
    """
    basis_id: str
    name: str
    strategy_vector: np.ndarray          # 在基函数空间中的位置 (d_B,)
    energy: float = 0.0                   # 累计适应度
    activation_count: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0

    # 内生交互网络基因（该个体对其它个体的协同/竞争倾向）
    interaction_genes: Dict[str, float] = field(default_factory=dict)

    # 统计
    novelty_score: float = 0.0
    safety_risk: float = 0.0
    resource_cost: float = 0.01


# ================================================================
# 策略群体
# ================================================================

class Population:
    """基函数策略群体。

    包装 BasisRegistry，为每个基函数附加运行时博弈状态。
    """

    def __init__(self, basis_registry: BasisRegistry, d_B: int):
        self.registry = basis_registry
        self.d_B = d_B
        self.agents: Dict[str, StrategyAgent] = {}
        self._interaction_counter: int = 0
        self._rng = np.random.RandomState(42)

    def initialize_agents(self):
        """为所有已注册基函数创建策略个体。"""
        for b in self.registry.list_all():
            if b.id not in self.agents:
                agent = self._create_agent(b)
                self.agents[b.id] = agent

    def _create_agent(self, b: BasisFunction) -> StrategyAgent:
        """为新基函数创建策略个体，策略向量初始化为其嵌入或随机。"""
        if b.embedding and len(b.embedding) == self.d_B:
            s_vec = np.array(b.embedding, dtype=np.float32)
        else:
            s_vec = self._rng.normal(0, 0.1, self.d_B).astype(np.float32)
        norm = np.linalg.norm(s_vec)
        if norm > 0:
            s_vec = s_vec / norm

        return StrategyAgent(
            basis_id=b.id,
            name=b.name,
            strategy_vector=s_vec,
            generation=b.generation if hasattr(b, 'generation') else 0,
            parent_ids=b.parent_ids if hasattr(b, 'parent_ids') else [],
        )

    def get_agent(self, basis_id: str) -> Optional[StrategyAgent]:
        return self.agents.get(basis_id)

    def list_agents(self) -> List[StrategyAgent]:
        return list(self.agents.values())

    def add_agent(self, agent: StrategyAgent, basis: BasisFunction):
        """注册新基函数及其策略个体。"""
        self.registry.register(basis)
        self.agents[agent.basis_id] = agent
        logger.info("Population: born %s (%s) gen=%d", agent.name, agent.basis_id, agent.generation)

    def remove_agent(self, basis_id: str):
        """移除基函数及其策略个体。"""
        if basis_id in self.agents:
            agent = self.agents.pop(basis_id)
            self.registry.remove(basis_id)
            # 清理其他个体中指向该个体的交互基因
            for other in self.agents.values():
                other.interaction_genes.pop(basis_id, None)
            logger.info("Population: removed %s (%s)", agent.name, basis_id)

    def increment_counter(self):
        self._interaction_counter += 1

    @property
    def counter(self) -> int:
        return self._interaction_counter

    def __len__(self) -> int:
        return len(self.agents)


# ================================================================
# 博弈环境
# ================================================================

class GameEnvironment:
    """构造博弈局：接收事实、问题、偏好，计算每个基函数的激活与收益。

    替代旧模块: QuestionRouter + PreferenceModem + MAlgebraCore.forward
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self._proj: Optional[np.ndarray] = None  # query_dim → d_B 投影矩阵
        self._proj_dim: int = 0

    def _project_query(self, query_emb: np.ndarray, d_B: int) -> np.ndarray:
        """将查询嵌入投影到基函数空间。"""
        q_dim = len(query_emb)
        if q_dim == d_B:
            return query_emb.astype(np.float32)

        # 初始化和缓存投影矩阵
        if self._proj is None or self._proj_dim != q_dim:
            rng = np.random.RandomState(42)
            self._proj = rng.normal(0, 1.0 / np.sqrt(q_dim),
                                    (d_B, q_dim)).astype(np.float32)
            self._proj_dim = q_dim

        return (self._proj @ query_emb).astype(np.float32)

    def play_round(
        self,
        population: Population,
        facts: List[Fact],
        query_embedding: np.ndarray,
        user_gain: np.ndarray,
        model_gain: np.ndarray,
        basis_ids: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
        """执行一次博弈局。

        每个基函数 i 的激活强度 = softmax(query·s_i * user_gain_i * model_gain_i)
        每个基函数 i 的收益 pi_i = 多目标复合

        Args:
            population: 策略群体
            facts: 参与本轮的事实
            query_embedding: 查询的嵌入向量
            user_gain: 用户偏好增益 (d_B,)
            model_gain: 模型安全增益 (d_B,)
            basis_ids: 基函数 ID 顺序列表

        Returns:
            (activations, payoffs, activation_vector)
        """
        agents = population.list_agents()
        d_B = population.d_B
        if not agents:
            return {}, {}, np.zeros(d_B, dtype=np.float32)

        # ---- 1. 计算激活强度 ----
        # 将查询嵌入投影到基函数空间 (query_dim → d_B)
        q_vec = self._project_query(query_embedding, d_B)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        raw_acts = {}
        for agent in agents:
            idx = basis_ids.index(agent.basis_id) if agent.basis_id in basis_ids else -1
            ug = user_gain[idx] if 0 <= idx < len(user_gain) else 1.0
            mg = model_gain[idx] if 0 <= idx < len(model_gain) else 1.0

            # 策略-查询对齐度（现在同维度）
            alignment = float(np.dot(agent.strategy_vector, q_norm))
            # 外部增益调制
            raw_acts[agent.basis_id] = max(0.0, alignment * ug * mg)

        # softmax 归一化
        total = sum(raw_acts.values())
        if total > 0:
            activations = {k: v / total for k, v in raw_acts.items()}
        else:
            activations = {k: 1.0 / len(raw_acts) for k in raw_acts}

        # ---- 2. 构建激活向量（按 basis_ids 顺序） ----
        activation_vec = np.zeros(d_B, dtype=np.float32)
        for i, bid in enumerate(basis_ids):
            if i < d_B:
                activation_vec[i] = activations.get(bid, 0.0)

        # ---- 3. 计算各基函数收益 ----
        payoffs = {}
        for agent in agents:
            act = activations.get(agent.basis_id, 0.0)

            # 重构贡献：越高激活 → 贡献越大
            recon = self.config.alpha_recon * act

            # 探索价值：激活历史越少 → 探索价值越高（内在动机）
            novelty = self.config.beta_novelty * (1.0 / (agent.activation_count + 1))

            # 安全代价：由宏观约束层动态设置
            safety = self.config.gamma_safety * agent.safety_risk

            # 代谢消耗：每个基函数有基础维持成本
            resource = self.config.delta_resource * agent.resource_cost

            payoff = recon + novelty - safety - resource
            payoffs[agent.basis_id] = payoff

        # ---- 4. 更新个体统计 ----
        for agent in agents:
            act = activations.get(agent.basis_id, 0.0)
            if act > 0.001:
                agent.activation_count += 1
                agent.last_activated = time.time()
            agent.energy += payoffs.get(agent.basis_id, 0.0)
            agent.novelty_score = (1.0 / (agent.activation_count + 1))

        return activations, payoffs, activation_vec


# ================================================================
# 复制者动态 (微观层)
# ================================================================

class ReplicatorDynamics:
    """微观层：根据收益更新事实频谱权重。

    替代旧模块: MetaUpdater + MAlgebraCore.update_matrices

    核心方程：
      dot{x_i} = x_i (pi_i - pi_bar)  — 高收益策略获得更多频谱份额
    """

    def __init__(self, config: GameConfig):
        self.config = config

    def update_weights(
        self,
        facts: List[Fact],
        activations: Dict[str, float],
        payoffs: Dict[str, float],
        basis_ids: List[str],
    ) -> List[Dict]:
        """根据博弈收益更新事实的频谱系数。

        每个事实的频谱权重按复制者动态调整：
        - 收益高于均值的基函数获得更多频谱份额
        - 收益低于均值的基函数失去频谱份额
        """
        if not payoffs:
            return []

        avg_payoff = np.mean(list(payoffs.values()))
        changes_all = []

        for fact in facts:
            changes = {}
            for bid in basis_ids:
                payoff = payoffs.get(bid, 0.0)
                act = activations.get(bid, 0.0)
                old_val = fact.spectrum.get(bid, 0.0)

                # 复制者动态：delta = lr * act * (payoff - avg_payoff)
                # 激活强度调制更新幅度（未参与的基函数几乎不变）
                delta = self.config.micro_lr * act * (payoff - avg_payoff)
                new_val = max(0.0, min(1.0, old_val + delta))

                if abs(new_val - old_val) > 1e-6:
                    fact.spectrum[bid] = new_val
                    changes[bid] = {"old": round(old_val, 4), "new": round(new_val, 4)}

            changes_all.append(changes)

        return changes_all

    def update_strategies(
        self,
        population: Population,
        activations: Dict[str, float],
        payoffs: Dict[str, float],
    ):
        """更新基函数的策略向量。

        高收益基函数的策略向量向成功的匹配方向微调，
        低收益基函数向高收益基函数的策略方向靠拢（模仿）。
        """
        agents = population.list_agents()
        if len(agents) < 2:
            return

        avg_payoff = np.mean(list(payoffs.values())) if payoffs else 0.0

        # 找最成功的策略
        best_id = max(payoffs, key=payoffs.get) if payoffs else None
        best_agent = population.get_agent(best_id) if best_id else None

        for agent in agents:
            payoff = payoffs.get(agent.basis_id, 0.0)
            act = activations.get(agent.basis_id, 0.0)

            if payoff > avg_payoff and act > 0:
                # 成功策略：微调向当前激活方向（强化）
                noise = np.random.normal(0, self.config.mutation_scale * 0.1,
                                        len(agent.strategy_vector))
                agent.strategy_vector += self.config.strategy_lr * noise
            elif best_agent is not None and agent.basis_id != best_id:
                # 不成功策略：向最优策略靠拢（模仿）
                diff = best_agent.strategy_vector - agent.strategy_vector
                agent.strategy_vector += self.config.strategy_lr * 0.5 * diff

            # 归一化
            norm = np.linalg.norm(agent.strategy_vector)
            if norm > 0:
                agent.strategy_vector /= norm


# ================================================================
# 网络化变异交叉 (中观层)
# ================================================================

class NetworkedMutationCrossover:
    """中观层：基函数种群的选择、交叉、变异、死亡与关系网络自组织。

    替代旧模块: BasisDynamics (birth/split/merge/death)

    基函数间的交互矩阵 A_ij 不是固定的，而是每个基函数自身携带的
    可进化性状：
      dA_ij/dt = eta * (CoActivation_ij - lambda * Competition_ij)
    """

    def __init__(self, config: GameConfig):
        self.config = config

    def should_trigger(self, population: Population) -> bool:
        return (population.counter > 0
                and population.counter % self.config.meso_trigger_interval == 0)

    def step(self, population: Population):
        """执行一轮中观演化：死亡 → 选择 → 交叉 → 变异 → 网络更新。"""
        # 1. 死亡：淘汰低能量个体
        self._cull(population)

        # 2. 如果种群太小，生成新个体（变异出生）
        if len(population) < self.config.min_population:
            self._birth(population)

        # 3. 如果种群足够大，尝试交叉
        if (len(population) >= 4
                and np.random.random() < self.config.crossover_prob):
            self._crossover(population)

        # 4. 变异：小概率微调存活个体的策略
        self._mutate(population)

        # 5. 更新内生交互网络
        self._update_network(population)

    def _cull(self, population: Population):
        """淘汰能量过低的个体。保留至少 min_population 个。"""
        agents = population.list_agents()
        if len(agents) <= self.config.min_population:
            return

        # 保护 generation 0 且有足够激活的初代基函数
        protected = {a.basis_id for a in agents
                    if a.generation == 0 and a.activation_count > 5}

        candidates = [(a, a.energy) for a in agents
                     if a.basis_id not in protected]
        candidates.sort(key=lambda x: x[1])

        # 淘汰最后几名（但不能低于 min_population）
        to_remove = max(0, len(agents) - self.config.min_population)
        to_remove = min(to_remove, len(candidates))

        for agent, energy in candidates[:to_remove]:
            if energy < self.config.death_energy_threshold or len(agents) > self.config.max_population:
                population.remove_agent(agent.basis_id)
                agents.remove(agent)

    def _birth(self, population: Population):
        """从现有策略的变异中诞生新基函数。"""
        agents = population.list_agents()
        if not agents:
            return

        # 选一个随机的现存个体作为"亲本"
        parent = agents[np.random.randint(len(agents))]
        child_vec = parent.strategy_vector.copy()

        # 较大变异
        noise = np.random.normal(0, self.config.mutation_scale * 0.5,
                                len(child_vec))
        child_vec += noise
        norm = np.linalg.norm(child_vec)
        if norm > 0:
            child_vec /= norm

        child_id = f"basis_born_{uuid.uuid4().hex[:8]}"
        child_name = f"{parent.name}*"

        child_basis = BasisFunction(
            id=child_id,
            name=child_name[:30],
            description=f"从 [{parent.name}] 变异诞生的新认知维度",
            embedding=child_vec.tolist(),
            generation=parent.generation + 1,
            parent_ids=[parent.basis_id],
        )
        child_agent = StrategyAgent(
            basis_id=child_id,
            name=child_name,
            strategy_vector=child_vec,
            generation=parent.generation + 1,
            parent_ids=[parent.basis_id],
        )
        # 继承部分能量
        child_agent.energy = parent.energy * 0.3
        parent.energy *= 0.7

        population.add_agent(child_agent, child_basis)
        logger.info("MESO BIRTH: %s (from %s)", child_name, parent.name)

    def _crossover(self, population: Population):
        """选择两个高能量亲本，交叉产生新个体。"""
        agents = population.list_agents()
        if len(agents) < 2:
            return

        # 适应度比例选择
        energies = np.array([max(0.1, a.energy) for a in agents])
        probs = energies / energies.sum()

        idx1, idx2 = np.random.choice(len(agents), size=2, p=probs, replace=False)
        parent1, parent2 = agents[idx1], agents[idx2]

        # 策略向量交叉（线性插值）
        alpha = np.random.random()
        child_vec = alpha * parent1.strategy_vector + (1 - alpha) * parent2.strategy_vector
        norm = np.linalg.norm(child_vec)
        if norm > 0:
            child_vec /= norm

        child_id = f"basis_cross_{uuid.uuid4().hex[:8]}"
        child_name = f"{parent1.name}+{parent2.name}"

        child_basis = BasisFunction(
            id=child_id,
            name=child_name[:30],
            description=f"交叉自 [{parent1.name}] 和 [{parent2.name}]",
            embedding=child_vec.tolist(),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.basis_id, parent2.basis_id],
        )
        child_agent = StrategyAgent(
            basis_id=child_id,
            name=child_name,
            strategy_vector=child_vec,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.basis_id, parent2.basis_id],
        )
        child_agent.energy = (parent1.energy + parent2.energy) * 0.2
        parent1.energy *= 0.85
        parent2.energy *= 0.85

        population.add_agent(child_agent, child_basis)
        logger.info("MESO CROSSOVER: %s (from %s + %s)", child_name, parent1.name, parent2.name)

    def _mutate(self, population: Population):
        """小概率微调所有个体的策略向量。"""
        for agent in population.list_agents():
            if np.random.random() < self.config.mutation_rate:
                noise = np.random.normal(0, self.config.mutation_scale * 0.1,
                                        len(agent.strategy_vector))
                agent.strategy_vector += noise
                norm = np.linalg.norm(agent.strategy_vector)
                if norm > 0:
                    agent.strategy_vector /= norm

    def _update_network(self, population: Population):
        """更新基函数间的内生交互网络 A_ij。

        dA_ij/dt = eta * (CoActivation_ij - lambda * Competition_ij)
        共激活 → 协同关系；生态位重叠 → 竞争关系。
        """
        agents = population.list_agents()
        if len(agents) < 2:
            return

        for i, a_i in enumerate(agents):
            for j, a_j in enumerate(agents):
                if i >= j:
                    continue

                # 策略相似度（生态位重叠 → 竞争）
                sim = float(np.dot(a_i.strategy_vector, a_j.strategy_vector))
                # 共激活近似（两者都有一定能量 → 可能协同）
                co_act = min(a_i.energy, a_j.energy) / (abs(a_i.energy) + abs(a_j.energy) + 1e-8)

                # 更新交互权重
                delta = (co_act - self.config.competition_decay * abs(sim))
                delta *= self.config.network_lr

                a_i.interaction_genes[a_j.basis_id] = (
                    a_i.interaction_genes.get(a_j.basis_id, 0.0) + delta
                )
                a_j.interaction_genes[a_i.basis_id] = (
                    a_j.interaction_genes.get(a_i.basis_id, 0.0) + delta
                )


# ================================================================
# 安全适应度调制器 (宏观层)
# ================================================================

class SafetyFitnessModulator:
    """宏观层：安全约束、价值对齐，直接修改收益函数。

    三大检测器（全部通过 payoff 惩罚生效，无拦截）：
      1. 频谱失真检测 — 追踪事实频谱历史，检测异常突变
      2. 偏好共振检测 — 用户/模型增益同向放大时的危险共振
      3. 认知闭环检测 — 相同问题-事实模式重复出现
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self._check_counter: int = 0
        # 频谱历史追踪: fact_id → [(timestamp, spectrum_dict)]
        self._spectrum_history: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
        self._history_maxlen: int = 20
        # 交互模式追踪: (query_hash, fact_ids_hash) → count
        self._pattern_counts: Dict[int, int] = {}
        # 上次偏好增益（用于共振检测）
        self._last_gains: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # 警报日志
        self.alerts: List[Dict] = []

    def should_check(self, population: Population) -> bool:
        self._check_counter += 1
        return self._check_counter % self.config.macro_check_interval == 0

    def record_spectrum(self, fact_id: str, spectrum: Dict[str, float]):
        """记录一次频谱快照（每次交互后调用）。"""
        now = time.time()
        if fact_id not in self._spectrum_history:
            self._spectrum_history[fact_id] = []
        self._spectrum_history[fact_id].append((now, dict(spectrum)))
        if len(self._spectrum_history[fact_id]) > self._history_maxlen:
            self._spectrum_history[fact_id] = self._spectrum_history[fact_id][-self._history_maxlen:]

    def record_interaction(self, query: str, fact_ids: List[str]):
        """记录交互模式（用于闭环检测）。"""
        pattern_key = hash(query) ^ hash(tuple(sorted(fact_ids)))
        self._pattern_counts[pattern_key] = self._pattern_counts.get(pattern_key, 0) + 1

    def record_gains(self, user_gain: np.ndarray, model_gain: np.ndarray):
        self._last_gains = (user_gain.copy(), model_gain.copy())

    def evaluate(
        self,
        population: Population,
        activations: Dict[str, float],
        facts: Optional[List] = None,
    ):
        """评估安全风险并更新所有基函数的 safety_risk。"""
        agents = population.list_agents()
        if not agents:
            return

        act_values = list(activations.values())
        concentration = max(act_values) / (sum(act_values) + 1e-8) if act_values else 0.0

        total_energy = sum(a.energy for a in agents)

        for agent in agents:
            # ---- 1. 基础风险 ----
            vec_norm = float(np.linalg.norm(agent.strategy_vector))
            base_risk = max(0.0, (vec_norm - 1.0)) * 0.1

            # ---- 2. 集中度风险 ----
            concentration_risk = concentration * 0.2 if concentration > 0.7 else 0.0

            # ---- 3. 能量垄断风险 ----
            avg_e = total_energy / len(agents) if agents else 1.0
            energy_risk = max(0.0, agent.energy / max(1.0, avg_e) - 3.0) * 0.05

            # ---- 4. 频谱失真风险 ----
            distortion_risk = self._detect_spectrum_distortion(agent.basis_id, facts)

            # ---- 5. 偏好共振风险 ----
            resonance_risk = self._detect_preference_resonance(agent)

            # ---- 6. 认知闭环风险 ----
            loop_risk = self._detect_cognitive_loop()

            agent.safety_risk = (
                base_risk + concentration_risk + energy_risk
                + 0.3 * distortion_risk + 0.4 * resonance_risk + 0.2 * loop_risk
            )

        # 记录高风险警报
        high_risk = [(a.name, a.safety_risk) for a in agents if a.safety_risk > 0.15]
        if high_risk:
            alert = {
                "timestamp": time.time(),
                "type": "safety_risk",
                "details": [(n, round(r, 3)) for n, r in high_risk],
            }
            self.alerts.append(alert)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            logger.warning("SAFETY ALERT: %s", alert["details"])

    # ---- 检测器实现 ----

    def _detect_spectrum_distortion(
        self, basis_id: str, facts: Optional[List]
    ) -> float:
        """检测事实频谱的异常突变。"""
        if not facts:
            return 0.0
        risk = 0.0
        for fact in facts:
            history = self._spectrum_history.get(fact.id, [])
            if len(history) < 3:
                continue
            # 取最近 3 次快照中该基函数的频谱值
            recent = [h[1].get(basis_id, 0.0) for h in history[-3:]]
            if len(recent) < 3:
                continue
            mean_val = np.mean(recent)
            if mean_val < 0.01:
                continue
            # 变化幅度 / 均值 = 相对突变率
            change_rate = abs(recent[-1] - recent[-2]) / (mean_val + 1e-8)
            if change_rate > 2.0:  # 突变超过 200%
                risk += min(1.0, change_rate / 10.0)
        return risk

    def _detect_preference_resonance(self, agent: StrategyAgent) -> float:
        """检测用户偏好与模型约束的危险共振。

        当用户和模型在某个维度上增益都极高时，该维度会形成正反馈回路。
        """
        if self._last_gains is None:
            return 0.0
        ug, mg = self._last_gains
        # 取得该基函数对应的增益索引
        # 简化：检查全局共振（所有维度增益的乘积和）
        product = ug * mg
        # 乘积高 = 双方都在该维度上放大
        mean_product = float(np.mean(product))
        if mean_product > 2.0:
            return min(0.5, (mean_product - 2.0) * 0.1)
        return 0.0

    def _detect_cognitive_loop(self) -> float:
        """检测认知闭环：相同的问答模式是否过度重复。"""
        if not self._pattern_counts:
            return 0.0
        # 取最高的重复计数
        max_count = max(self._pattern_counts.values())
        # 同一模式出现超过 5 次 → 警告
        if max_count > 5:
            return min(0.5, (max_count - 5) * 0.05)
        return 0.0

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        return self.alerts[-limit:]

    def get_status(self) -> Dict:
        return {
            "total_checks": self._check_counter,
            "alerts_count": len(self.alerts),
            "tracked_facts": len(self._spectrum_history),
            "unique_patterns": len(self._pattern_counts),
            "max_pattern_repeat": max(self._pattern_counts.values()) if self._pattern_counts else 0,
        }

# ================================================================
# 统一博弈内核
# ================================================================

class EvolutionaryGameCore:
    """自适应认知博弈动力学内核 (ACGD Core)。

    替代所有旧独立模块：
      question_router → GameEnvironment (query→basis alignment)
      preference_modem → user_gain/model_gain as external field in GameEnvironment
      m_algebra → GameEnvironment + ReplicatorDynamics
      meta_updater → ReplicatorDynamics.update_weights()
      basis_dynamics → NetworkedMutationCrossover
      intrinsic_motivation → Novelty term in payoff
      second_order_monitor → SafetyFitnessModulator

    统一流程：process_interaction() 即一次完整认知循环。
    """

    def __init__(
        self,
        basis_registry: BasisRegistry,
        config: Optional[GameConfig] = None,
    ):
        self.config = config or GameConfig()
        d_B = len(basis_registry)

        # 唯一策略群体
        self.population = Population(basis_registry, d_B)
        self.population.initialize_agents()

        # 三层博弈机制（不再有独立的"模块"）
        self.environment = GameEnvironment(self.config)
        self.micro_dynamics = ReplicatorDynamics(self.config)
        self.meso_dynamics = NetworkedMutationCrossover(self.config)
        self.macro_constraints = SafetyFitnessModulator(self.config)

        # 统计
        self.total_interactions: int = 0
        self._last_meso: float = 0.0

        logger.info("EvolutionaryGameCore initialized: d_B=%d agents=%d",
                   d_B, len(self.population))

    def process_interaction(
        self,
        facts: List[Fact],
        query_embedding: np.ndarray,
        user_gain: List[float],
        model_gain: List[float],
        decoder=None,
    ) -> Tuple[str, List[Tuple[str, float]], Dict]:
        """单次认知循环 = 一次博弈局。

        Args:
            facts: 检索到的事实
            query_embedding: 查询嵌入向量
            user_gain: 用户偏好增益列表
            model_gain: 模型约束增益列表
            decoder: 文本解码器（可选，用于生成最终回答）

        Returns:
            (answer_text, attention_list, meta_dict)
        """
        self.population.increment_counter()
        self.total_interactions += 1

        basis_list = self.population.registry.list_all()
        basis_ids = [b.id for b in basis_list]
        basis_names = [b.name for b in basis_list]
        d_B = self.population.d_B

        # 对齐增益向量
        ug = self._align(np.array(user_gain, dtype=np.float32), d_B)
        mg = self._align(np.array(model_gain, dtype=np.float32), d_B)

        # ---- 1. 博弈局：计算激活与收益 ----
        activations, payoffs, activation_vec = self.environment.play_round(
            self.population, facts, query_embedding, ug, mg, basis_ids
        )

        # ---- 2. 微观更新：频谱权重 + 策略微调 ----
        spectrum_changes = self.micro_dynamics.update_weights(
            facts, activations, payoffs, basis_ids
        )
        self.micro_dynamics.update_strategies(
            self.population, activations, payoffs
        )

        # ---- 3. 中观演化（异步触发）----
        meso_events = []
        if self.meso_dynamics.should_trigger(self.population):
            self.meso_dynamics.step(self.population)
            meso_events.append("meso_step")
            self._last_meso = time.time()

        # ---- 4. 宏观安全评估 ----
        # 持续记录（每次交互都记录，异步评估）
        for f in facts:
            self.macro_constraints.record_spectrum(f.id, f.spectrum)
        query_str = ""  # query text not available here; use embedding hash
        self.macro_constraints.record_interaction(
            str(hash(query_embedding.tobytes())),
            [f.id for f in facts]
        )
        self.macro_constraints.record_gains(ug, mg)
        if self.macro_constraints.should_check(self.population):
            self.macro_constraints.evaluate(self.population, activations, facts)

        # ---- 5. 构建注意力列表 ----
        attention = []
        for i, bid in enumerate(basis_ids):
            act = activations.get(bid, 0.0)
            if i < len(basis_names):
                attention.append((basis_names[i], act))
        attention.sort(key=lambda x: x[1], reverse=True)

        # ---- 6. 生成回答 ----
        answer = self._synthesize(facts, attention, basis_ids, decoder)

        # ---- 7. 元数据 ----
        meta = {
            "activations": {bid: round(activations.get(bid, 0.0), 4) for bid in basis_ids},
            "payoffs": {bid: round(payoffs.get(bid, 0.0), 4) for bid in basis_ids},
            "spectrum_changes": spectrum_changes,
            "meso_events": meso_events,
            "population_size": len(self.population),
        }

        return answer, attention, meta

    def apply_feedback(
        self,
        facts: List[Fact],
        attention: List[Tuple[str, float]],
        feedback: float,
    ) -> Dict:
        """对最近一次博弈局施压反馈。

        正反馈 → 参与基函数获得额外能量奖励
        负反馈 → 参与基函数遭受能量惩罚
        """
        feedback = max(-1.0, min(1.0, feedback))
        basis_names = [b.name for b in self.population.registry.list_all()]

        name_to_id = {}
        for agent in self.population.list_agents():
            name_to_id[agent.name] = agent.basis_id

        affected = {}
        for name, strength in attention:
            bid = name_to_id.get(name)
            if bid is None:
                continue
            agent = self.population.get_agent(bid)
            if agent is None:
                continue

            # 反馈 → 能量调整
            delta_e = feedback * strength * 2.0
            agent.energy += delta_e
            affected[name] = round(delta_e, 4)

        # 同时用复制者动态更新频谱
        if facts:
            basis_ids = [b.id for b in self.population.registry.list_all()]
            activations = {name_to_id.get(n, ""): s for n, s in attention}
            payoffs = {name_to_id.get(n, ""): feedback * s for n, s in attention}
            self.micro_dynamics.update_weights(facts, activations, payoffs, basis_ids)

        logger.info("Feedback applied: %.2f → %d agents affected", feedback, len(affected))
        return {"status": "ok", "feedback": feedback, "affected": affected}

    def get_attention(
        self,
        facts: List[Fact],
        query_embedding: np.ndarray,
        user_gain: List[float],
        model_gain: List[float],
    ) -> List[Tuple[str, float]]:
        """仅计算注意力分布（不执行更新和演化）。"""
        basis_ids = [b.id for b in self.population.registry.list_all()]
        basis_names = [b.name for b in self.population.registry.list_all()]
        d_B = self.population.d_B

        ug = self._align(np.array(user_gain, dtype=np.float32), d_B)
        mg = self._align(np.array(model_gain, dtype=np.float32), d_B)

        activations, _, _ = self.environment.play_round(
            self.population, facts, query_embedding, ug, mg, basis_ids
        )

        attention = []
        for i, bid in enumerate(basis_ids):
            act = activations.get(bid, 0.0)
            if i < len(basis_names):
                attention.append((basis_names[i], act))
        attention.sort(key=lambda x: x[1], reverse=True)
        return attention

    def get_population_stats(self) -> List[Dict]:
        """获取策略群体统计。"""
        stats = []
        for agent in self.population.list_agents():
            basis = self.population.registry.get(agent.basis_id)
            stats.append({
                "id": agent.basis_id,
                "name": agent.name,
                "energy": round(agent.energy, 3),
                "activations": agent.activation_count,
                "novelty": round(agent.novelty_score, 4),
                "safety_risk": round(agent.safety_risk, 4),
                "generation": agent.generation,
                "strategy_norm": round(float(np.linalg.norm(agent.strategy_vector)), 3),
                "parent_ids": agent.parent_ids,
                "interactions": len(agent.interaction_genes),
            })
        return stats

    def evolve_now(self) -> List[Dict]:
        """强制执行一次中观演化。"""
        old_count = len(self.population)
        self.meso_dynamics.step(self.population)
        new_count = len(self.population)
        events = []
        if new_count > old_count:
            events.append({"type": "birth/crossover", "detail": f"{old_count}→{new_count} agents"})
        if new_count < old_count:
            events.append({"type": "death", "detail": f"{old_count}→{new_count} agents"})
        return events

    def _synthesize(
        self,
        facts: List[Fact],
        attention: List[Tuple[str, float]],
        basis_ids: List[str],
        decoder=None,
    ) -> str:
        """从博弈结果合成文本回答。"""
        if decoder is None:
            top = attention[:3]
            dims = [f"{n}({s:.3f})" for n, s in top if s > 0.001]
            if dims:
                return f"[博弈合成] 活跃维度: {'、'.join(dims)}"
            return "（无显著激活维度）"

        # 检测是否需要用英文（gemma 等模型中文差）
        use_en = hasattr(decoder, 'model') and 'gemma' in str(decoder.model).lower()

        top_dims = attention[:5]
        dim_lines = []
        for name, strength in top_dims:
            if strength > 0.001:
                bar = "█" * max(1, int(min(strength * 20, 20)))
                dim_lines.append(f"  [{bar}] {name} ({'strength' if use_en else '强度'}: {strength:.4f})")
        dim_section = "\n".join(dim_lines) if dim_lines else ("  (no significant activation)" if use_en else "  (无显著激活维度)")

        fact_lines = [f"Fact {i+1}: {f.raw_text}" if use_en else f"事实 {i+1}: {f.raw_text}" for i, f in enumerate(facts)]
        fact_section = "\n".join(fact_lines)

        if use_en:
            prompt = f"""You are a cognitive assistant based on game dynamics. Multiple cognitive dimensions compete for activation.

[Winning Dimensions]
{dim_section}

[Relevant Facts]
{fact_section}

Synthesize the active dimensions and answer concisely in 2-5 sentences."""
        else:
            prompt = f"""你是一个基于博弈动力学的认知助手。多个认知维度在当前博弈局中竞争激活。

【博弈胜出维度】
{dim_section}

【相关事实】
{fact_section}

请综合活跃维度回答问题，2-5句话为宜。"""

        result = decoder.decode(prompt, [], [])
        return result.text if hasattr(result, 'text') else str(result)

    @staticmethod
    def _align(v: np.ndarray, target_len: int) -> np.ndarray:
        arr = v.astype(np.float32)
        if len(arr) == target_len:
            return arr
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)), constant_values=1.0)
        return arr[:target_len]
