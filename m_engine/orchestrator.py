"""
MEngineOrchestrator: 主控协调器 (Singularity Edition)。

所有核心智能逻辑收敛到唯一内核：EvolutionaryGameCore。
旧模块（QuestionRouter, PreferenceModem, MAlgebraCore, MetaUpdater,
BasisDynamics）不再被导入，其功能全部由博弈动力学自发产生。

保留的模块（数据/I/O/持久化）：
  - BasisRegistry: 基函数数据定义
  - FactBus: 事实存储与检索
  - Embedder: 文本嵌入
  - Decoder: 文本合成
  - NeuralBridge/NeuralDecoder: 双空间桥接
  - Persistence: SQLite 持久化
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.basis_registry import BasisRegistry, BasisFunction
from .core.fact_bus import FactBus, Fact
from .core.decoder import Decoder
from .core.embedder import Embedder
from .core.persistence import Persistence
from .core.neural_bridge import NeuralBridge
from .core.neural_decoder import NeuralDecoder, NeuralDecodeResult
from .core.evolutionary_game_core import (
    EvolutionaryGameCore, GameConfig, Population
)
from .core.intrinsic_motivation import IntrinsicMotivation
from .core.analogy_engine import AnalogyEngine
from .core.knowledge_exchange import KnowledgeExchange

logger = logging.getLogger(__name__)


class MEngineOrchestrator:
    """M-Engine 主控协调器 (Singularity Edition)。

    完整认知循环：
      1. 嵌入查询 → 语义检索事实
      2. EvolutionaryGameCore.process_interaction() — 唯一智能来源
      3. (可选) 神经双空间桥接
      4. 反馈 → game_core.apply_feedback()
      5. 持久化
    """

    def __init__(
        self,
        data_dir: str = "data",
        db_path: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedder_model: Optional[str] = None,
        neural_mode: bool = True,
        game_config: Optional[GameConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.neural_mode = neural_mode

        # 嵌入器
        self.embedder = Embedder(
            model_name=embedder_model, api_key=api_key, base_url=base_url
        )

        # 数据层
        self.basis_registry = BasisRegistry()
        self.fact_bus = FactBus(embedder=self.embedder.encode_single)
        self.decoder = Decoder(model=model, api_key=api_key, base_url=base_url)

        # 唯一博弈内核（延迟初始化：需要基函数加载后才能创建）
        self.game_core: Optional[EvolutionaryGameCore] = None
        self._game_config = game_config or GameConfig()

        # 内在动机引擎（延迟初始化）
        self.motivation: Optional[IntrinsicMotivation] = None

        # 类比推理引擎（延迟初始化）
        self.analogy: Optional[AnalogyEngine] = None

        # 知识交换协议（延迟初始化）
        self.exchange: Optional[KnowledgeExchange] = None

        # 双空间（延迟初始化）
        self.bridge: Optional[NeuralBridge] = None
        self.neural_decoder: Optional[NeuralDecoder] = None

        # 交互历史
        self._last_interaction: Optional[Dict] = None

        # 持久化
        self._db_path = db_path or str(self.data_dir / "m_engine.db")
        self._persistence: Optional[Persistence] = None

    # ================================================================
    # 初始化
    # ================================================================

    def initialize(
        self,
        basis_file: str = "base_basis.json",
        questions_file: str = "base_questions.json",
    ) -> None:
        """加载基函数，创建博弈内核和双空间桥接。"""
        basis_path = self.data_dir / basis_file
        questions_path = self.data_dir / questions_file

        logger.info("Loading basis from: %s", basis_path)
        self.basis_registry.load_from_json(str(basis_path))

        # 加载问题模板（仅用于基问题的 filter_weights 种子）
        import json
        if questions_path.exists():
            with open(questions_path, "r", encoding="utf-8") as f:
                self._question_templates = json.load(f)
        else:
            self._question_templates = []

        d_B = len(self.basis_registry)

        # ---- 唯一博弈内核 ----
        self.game_core = EvolutionaryGameCore(self.basis_registry, self._game_config)
        # 内在动机 + 类比 + 知识交换
        self.motivation = IntrinsicMotivation(self.basis_registry)
        self.analogy = AnalogyEngine(self.basis_registry)
        self.exchange = KnowledgeExchange(self.basis_registry, self.fact_bus)
        logger.info("Motivation, analogy, exchange engines initialized")

        logger.info("Game core initialized: %d basis agents", d_B)

        # ---- 双空间桥接 ----
        if self.neural_mode:
            self.bridge = NeuralBridge(model_name=self.decoder.model)
            n_tokens = self.bridge.build_signatures(self.basis_registry)
            self.neural_decoder = NeuralDecoder(self.decoder, self.bridge)
            logger.info("Neural bridge initialized: %d tokens", n_tokens)

        logger.info("MEngineOrchestrator (Singularity) ready: d_B=%d", d_B)

    # ================================================================
    # 核心循环
    # ================================================================

    def process_query(
        self,
        query: str,
        user_gain: Optional[List[float]] = None,
        model_gain: Optional[List[float]] = None,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """处理用户查询 —— 一次完整的认知博弈局。

        Args:
            query: 用户自然语言查询
            user_gain: 用户偏好增益（可选）
            model_gain: 模型约束增益（可选）

        Returns:
            (answer, attention)
        """
        if self.game_core is None:
            return "（系统未初始化）", []

        d_B = len(self.basis_registry)
        basis_ids = [b.id for b in self.basis_registry.list_all()]

        # 默认增益
        if user_gain is None:
            user_gain = [1.0] * d_B
        if model_gain is None:
            model_gain = [1.0] * d_B

        # 1. 语义检索
        query_embedding_arr = np.array(
            self.embedder.encode_single(query), dtype=np.float32
        )
        facts = self.fact_bus.retrieve(query_embedding_arr.tolist(), top_k=5)

        if not facts:
            answer = "（记忆库中还没有任何事实。请先用 'store <文本>' 命令存储一些事实。）"
            self._last_interaction = {
                "query": query, "answer": answer, "attention": [],
                "facts": [], "meta": {},
            }
            return answer, []

        # 2. 博弈内核：唯一智能来源
        answer, attention, meta = self.game_core.process_interaction(
            facts=facts,
            query_embedding=query_embedding_arr,
            user_gain=user_gain,
            model_gain=model_gain,
            decoder=self.decoder,
        )

        # 3. (可选) 双空间神经桥接
        neural_result = None
        if self.neural_mode and self.neural_decoder is not None:
            try:
                prompt = self._build_prompt_from_attention(facts, attention)
                activation_vec = np.array(
                    [s for _, s in attention], dtype=np.float32
                )
                neural_result = self.neural_decoder.decode(
                    prompt=prompt,
                    spectrum=activation_vec.tolist() if len(activation_vec) > 0 else [0.0]*d_B,
                    basis_names=[b.name for b in self.basis_registry.list_all()],
                )
                answer = neural_result.text
            except Exception as e:
                logger.warning("Neural path failed, using game answer: %s", e)

        # 4. 记录交互
        self._last_interaction = {
            "query": query,
            "answer": answer,
            "attention": attention,
            "facts": facts,
            "meta": meta,
            "neural_result": neural_result,
            "user_gain": user_gain,
        }

        return answer, attention

    def apply_feedback(self, fact_id: str, score: float) -> Dict:
        """对最近一次交互施加反馈。"""
        if self.game_core is None:
            return {"status": "error", "message": "Game core not initialized"}
        if self._last_interaction is None:
            return {"status": "error", "message": "No previous interaction"}

        attention = self._last_interaction.get("attention", [])
        facts = self._last_interaction.get("facts", [])
        target_facts = [f for f in facts if f.id == fact_id] if fact_id else facts

        # 1. 博弈内核反馈
        result = self.game_core.apply_feedback(target_facts, attention, score)

        # 2. 更新神经桥接
        neural_result = self._last_interaction.get("neural_result")
        if (neural_result is not None
                and self.bridge is not None
                and neural_result.token_logprobs):
            d_B = len(self.basis_registry)
            act_vec = np.array([s for _, s in attention], dtype=np.float32)
            if len(act_vec) < d_B:
                act_vec = np.pad(act_vec, (0, d_B - len(act_vec)))
            self.bridge.update(
                symbolic_activation=act_vec,
                neural_reading=neural_result.neural_reading
                if neural_result.neural_reading is not None
                else np.zeros_like(act_vec),
                token_logprobs=neural_result.token_logprobs,
                feedback=score,
            )

        return result

    def get_spectrum(self, fact_id: str) -> Optional[Dict]:
        """查看事实频谱。"""
        fact = self.fact_bus.get_fact(fact_id)
        if fact is None:
            return None
        named_spectrum = {}
        for bid, score in fact.spectrum.items():
            bf = self.basis_registry.get(bid)
            name = bf.name if bf else bid
            named_spectrum[name] = score
        return {
            "id": fact.id,
            "text": fact.raw_text,
            "spectrum": named_spectrum,
            "activation": fact.activation,
        }

    def get_neural_reading(self) -> Optional[Dict]:
        """获取最近神经读数。"""
        if self._last_interaction is None:
            return None
        nr = self._last_interaction.get("neural_result")
        if nr is None or nr.neural_reading is None:
            return None
        basis_names = [b.name for b in self.basis_registry.list_all()]
        reading = {}
        for i, name in enumerate(basis_names):
            s_val = float(nr.symbolic_activation[i]) if nr.symbolic_activation is not None and i < len(nr.symbolic_activation) else 0.0
            n_val = float(nr.neural_reading[i]) if nr.neural_reading is not None and i < len(nr.neural_reading) else 0.0
            d_val = n_val - s_val
            reading[name] = {"symbolic": round(s_val, 4), "neural": round(n_val, 4), "delta": round(d_val, 4)}
        return {"logit_bias_count": len(nr.logit_bias) if nr.logit_bias else 0, "dimensions": reading}

    def get_dynamics_status(self) -> Dict:
        """获取博弈内核状态。"""
        if self.game_core is None:
            return {"status": "not_initialized"}
        pop_stats = self.game_core.get_population_stats()
        return {
            "total_basis": len(self.basis_registry),
            "total_interactions": self.game_core.total_interactions,
            "config": {
                "micro_lr": self.game_core.config.micro_lr,
                "meso_interval": self.game_core.config.meso_trigger_interval,
                "min_pop": self.game_core.config.min_population,
                "max_pop": self.game_core.config.max_population,
            },
            "population": pop_stats,
        }

    def get_safety_status(self) -> Dict:
        """获取第二序监控状态。"""
        if self.game_core is None:
            return {"status": "not_initialized"}
        return self.game_core.macro_constraints.get_status()

    def get_safety_alerts(self, limit: int = 10) -> List[Dict]:
        """获取安全警报。"""
        if self.game_core is None:
            return []
        return self.game_core.macro_constraints.get_alerts(limit)

    def explore_gaps(self, max_questions: int = 5) -> List[Dict]:
        """检测认知缺口并生成探索性问题。"""
        if self.motivation is None:
            return []
        facts = self.fact_bus.list_all()
        if not facts:
            return [{"message": "（记忆库为空，无法探测缺口）"}]
        targets = self.motivation.detect_gaps(facts)
        if not targets:
            return [{"message": "（未检测到显著认知缺口）"}]
        self.motivation.generate_questions(targets, self.decoder, max_questions)
        return self.motivation.get_top_gaps(facts, max_questions)

    def find_analogies(self) -> List[Dict]:
        """发现事实间的类比关系。"""
        if self.analogy is None:
            return []
        facts = self.fact_bus.list_all()
        return self.analogy.get_analogy_summary(facts)

    def export_knowledge(self, fact_id: Optional[str] = None) -> List[Dict]:
        """导出事实为可交换格式。"""
        if self.exchange is None:
            return []
        if fact_id:
            item = self.exchange.export_fact(fact_id)
            return [item] if item else []
        return self.exchange.export_all()

    def import_knowledge(self, data_list: List[Dict]) -> int:
        """从交换格式导入事实。"""
        if self.exchange is None:
            return 0
        return self.exchange.import_all(data_list)

    def evolve_now(self) -> List[Dict]:
        """强制执行一次中观演化。"""
        if self.game_core is None:
            return []
        return self.game_core.evolve_now()

    def store_fact(self, text: str, fact_id: Optional[str] = None) -> Fact:
        """存储事实。"""
        f = Fact(
            id=fact_id or f"fact_{uuid.uuid4().hex[:8]}",
            raw_text=text,
        )
        try:
            f.embedding = self.embedder.encode_single(text)
        except Exception as e:
            logger.warning("Failed to embed fact: %s", e)
        for bf in self.basis_registry.list_all():
            f.spectrum[bf.id] = 0.1
        self.fact_bus.add_fact(f)
        logger.info("Stored fact: %s", f.id)
        return f

    # ================================================================
    # 持久化
    # ================================================================

    def _get_persistence(self) -> Persistence:
        if self._persistence is None:
            self._persistence = Persistence(self._db_path)
            self._persistence.connect()
        return self._persistence

    def save(self, db_path: Optional[str] = None) -> str:
        if db_path:
            self._db_path = db_path
            if self._persistence:
                self._persistence.close()
            self._persistence = None
        p = self._get_persistence()
        facts = self.fact_bus.list_all()
        if facts:
            p.save_facts(facts)
        p.save_game_core(self.game_core)  # 保存博弈内核状态
        p.save_basis_stats(self.basis_registry)
        if self.bridge is not None:
            p.save_neural_bridge(self.bridge)
        p.set_meta("basis_count", str(len(self.basis_registry)))
        p.set_meta("total_interactions", str(
            self.game_core.total_interactions if self.game_core else 0))
        logger.info("State saved to %s", self._db_path)
        return self._db_path

    def load(self, db_path: Optional[str] = None) -> int:
        if db_path:
            self._db_path = db_path
        p = self._get_persistence()
        facts = p.load_facts()
        for f in facts:
            self.fact_bus.add_fact(f)
        p.restore_basis_stats(self.basis_registry)
        p.restore_game_core(self.game_core)
        if self.bridge is not None:
            p.restore_neural_bridge(self.bridge)
        # 重建博弈代理（基于恢复后的统计）
        if self.game_core is not None:
            self.game_core.population = Population(self.basis_registry, len(self.basis_registry))
            self.game_core.population.initialize_agents()
            # 恢复能量统计
            for b in self.basis_registry.list_all():
                agent = self.game_core.population.get_agent(b.id)
                if agent is not None:
                    agent.energy = b.activation_count * b.avg_strength
                    agent.activation_count = b.activation_count
        logger.info("State loaded from %s (%d facts)", self._db_path, len(facts))
        return len(facts)

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _build_prompt_from_attention(facts, attention) -> str:
        top = attention[:5]
        dim_lines = []
        for name, strength in top:
            if strength > 0.001:
                bar = "█" * max(1, int(min(strength * 20, 20)))
                dim_lines.append(f"  [{bar}] {name} (强度: {strength:.4f})")
        dim_section = "\n".join(dim_lines) if dim_lines else "  (无显著激活维度)"
        fact_lines = [f"事实 {i+1}: {f.raw_text}" for i, f in enumerate(facts)]
        return f"""你是一个多维度记忆分析助手。

【活跃维度】
{dim_section}

【事实】
{chr(10).join(fact_lines)}

请综合分析后回答问题，2-5句话。"""
