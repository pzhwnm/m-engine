"""
Persistence: SQLite 持久化层。
负责将事实、矩阵权重、用户画像序列化到磁盘，支持 save/load。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .fact_bus import Fact, FactBus
from .preference_modem import UserProfile
from .meta_updater import MetaUpdater

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    spectrum TEXT NOT NULL,
    embedding TEXT,
    activation TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS meta_weights (
    layer_name TEXT PRIMARY KEY,
    weights TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    id TEXT PRIMARY KEY,
    gain TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Persistence:
    """SQLite 持久化管理器。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    def connect(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("Connected to database: %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    # ---- Facts ----

    def save_facts(self, facts: List[Fact]) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM facts")
        for f in facts:
            conn.execute(
                "INSERT OR REPLACE INTO facts (id, raw_text, spectrum, embedding, activation) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    f.id,
                    f.raw_text,
                    json.dumps(f.spectrum, ensure_ascii=False),
                    json.dumps(f.embedding) if f.embedding else None,
                    json.dumps(f.activation, ensure_ascii=False),
                ),
            )
        conn.commit()
        logger.info("Saved %d facts", len(facts))

    def load_facts(self) -> List[Fact]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, raw_text, spectrum, embedding, activation FROM facts"
        ).fetchall()
        facts = []
        for row in rows:
            f = Fact(
                id=row[0],
                raw_text=row[1],
                spectrum=json.loads(row[2]),
                embedding=json.loads(row[3]) if row[3] else [],
                activation=json.loads(row[4]) if row[4] else {},
            )
            facts.append(f)
        logger.info("Loaded %d facts", len(facts))
        return facts

    # ---- MetaUpdater weights ----

    def save_meta_weights(self, meta: MetaUpdater) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM meta_weights")
        weight_map = {
            "W1": meta.W1.tolist(),
            "b1": meta.b1.tolist(),
            "W2": meta.W2.tolist(),
            "b2": meta.b2.tolist(),
            "d_B": meta.d_B,
            "rank": meta.rank,
            "hidden_dim": getattr(meta, "hidden_dim", 64),
            "lr": meta.lr,
        }
        for name, val in weight_map.items():
            conn.execute(
                "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
                (name, json.dumps(val)),
            )
        conn.commit()
        logger.info("Saved MetaUpdater weights")

    def save_game_core(self, game_core) -> None:
        """Save EvolutionaryGameCore agent population state."""
        conn = self._get_conn()
        agent_data = []
        for agent in game_core.population.list_agents():
            agent_data.append({
                "basis_id": agent.basis_id,
                "energy": agent.energy,
                "activation_count": agent.activation_count,
                "strategy_vector": agent.strategy_vector.tolist(),
                "generation": agent.generation,
                "parent_ids": agent.parent_ids,
                "novelty_score": agent.novelty_score,
                "safety_risk": agent.safety_risk,
            })
        conn.execute(
            "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
            ("game_core_agents", json.dumps(agent_data)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
            ("game_core_config", json.dumps({
                "total_interactions": game_core.total_interactions,
            })),
        )
        conn.commit()
        logger.info("Saved game core state: %d agents", len(agent_data))

    def restore_game_core(self, game_core) -> bool:
        """Restore EvolutionaryGameCore agent population state."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT weights FROM meta_weights WHERE layer_name = ?",
            ("game_core_agents",)
        ).fetchone()
        if not row:
            return False
        agent_data = json.loads(row[0])
        for ad in agent_data:
            agent = game_core.population.get_agent(ad["basis_id"])
            if agent is not None:
                agent.energy = ad.get("energy", 0.0)
                agent.activation_count = ad.get("activation_count", 0)
                agent.generation = ad.get("generation", 0)
                agent.parent_ids = ad.get("parent_ids", [])
                agent.novelty_score = ad.get("novelty_score", 0.0)
                agent.safety_risk = ad.get("safety_risk", 0.0)
        row2 = conn.execute(
            "SELECT weights FROM meta_weights WHERE layer_name = ?",
            ("game_core_config",)
        ).fetchone()
        if row2:
            config_data = json.loads(row2[0])
            game_core.total_interactions = config_data.get("total_interactions", 0)
            game_core.population._interaction_counter = game_core.total_interactions
        logger.info("Restored game core state for %d agents", len(agent_data))
        return True

    def save_algebra_matrices(self, m_algebra) -> None:
        """保存 MAlgebraCore 的 W 和 G 矩阵。"""
        conn = self._get_conn()
        if m_algebra.W is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
                ("algebra_W", json.dumps(m_algebra.W.tolist())),
            )
        if m_algebra.G is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
                ("algebra_G", json.dumps(m_algebra.G.tolist())),
            )
        if m_algebra.d_B > 0:
            conn.execute(
                "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
                ("algebra_d_B", json.dumps(m_algebra.d_B)),
            )
        conn.commit()
        logger.info("Saved M-Algebra matrices")

    def load_meta_weights(self) -> Optional[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT layer_name, weights FROM meta_weights"
        ).fetchall()
        if not rows:
            return None
        result = {}
        for name, val_json in rows:
            result[name] = json.loads(val_json)
        return result

    def restore_meta_weights(self, meta: MetaUpdater) -> bool:
        data = self.load_meta_weights()
        if data is None:
            return False
        if "W1" in data:
            meta.W1 = np.array(data["W1"], dtype=np.float32)
        if "b1" in data:
            meta.b1 = np.array(data["b1"], dtype=np.float32)
        if "W2" in data:
            meta.W2 = np.array(data["W2"], dtype=np.float32)
        if "b2" in data:
            meta.b2 = np.array(data["b2"], dtype=np.float32)
        logger.info("Restored MetaUpdater weights from database")
        return True

    def restore_algebra_matrices(self, m_algebra) -> bool:
        """从数据库恢复 MAlgebraCore 的 W 和 G 矩阵。"""
        data = self.load_meta_weights()
        if data is None:
            return False
        if "algebra_W" in data:
            m_algebra.W = np.array(data["algebra_W"], dtype=np.float32)
        if "algebra_G" in data:
            m_algebra.G = np.array(data["algebra_G"], dtype=np.float32)
        if "algebra_d_B" in data:
            m_algebra.d_B = data["algebra_d_B"]
            m_algebra._initialized = True
        if m_algebra.W is not None:
            logger.info("Restored M-Algebra matrices from database")
            return True
        return False

    # ---- User profiles ----

    def save_user_profiles(self, profiles: List[UserProfile]) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM user_profiles")
        for p in profiles:
            conn.execute(
                "INSERT OR REPLACE INTO user_profiles (id, gain) VALUES (?, ?)",
                (p.id, json.dumps(p.gain, ensure_ascii=False)),
            )
        conn.commit()
        logger.info("Saved %d user profiles", len(profiles))

    def load_user_profiles(self) -> List[UserProfile]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, gain FROM user_profiles"
        ).fetchall()
        profiles = []
        for row in rows:
            p = UserProfile(id=row[0], gain=json.loads(row[1]))
            profiles.append(p)
        logger.info("Loaded %d user profiles", len(profiles))
        return profiles

    # ---- NeuralBridge ----

    def save_neural_bridge(self, bridge) -> None:
        """保存 NeuralBridge 的投影矩阵和 token 签名。"""
        conn = self._get_conn()
        data = {
            "P_decode": bridge.P_decode.tolist() if bridge.P_decode is not None else None,
            "P_encode": bridge.P_encode.tolist() if bridge.P_encode is not None else None,
            "token_ids": bridge.token_ids,
            "token_strs": bridge.token_strs,
            "d_B": bridge.d_B,
            "n_tokens": bridge.n_tokens,
            "model_name": bridge.model_name,
        }
        conn.execute(
            "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
            ("neural_bridge", json.dumps(data)),
        )
        conn.commit()
        logger.info("Saved NeuralBridge (%d tokens)", bridge.n_tokens)

    # ---- Basis stats (for dynamics) ----

    def save_basis_stats(self, basis_registry) -> None:
        """保存基函数的动力学统计信息。"""
        conn = self._get_conn()
        stats = []
        for b in basis_registry.list_all():
            stats.append({
                "id": b.id,
                "activation_count": b.activation_count,
                "strength_sum": b.strength_sum,
                "strength_history": b.strength_history,
                "created_at": b.created_at,
                "last_activated": b.last_activated,
                "parent_ids": b.parent_ids,
                "generation": b.generation,
            })
        conn.execute(
            "INSERT OR REPLACE INTO meta_weights (layer_name, weights) VALUES (?, ?)",
            ("basis_stats", json.dumps(stats)),
        )
        conn.commit()
        logger.info("Saved basis stats for %d functions", len(stats))

    def restore_basis_stats(self, basis_registry) -> bool:
        """恢复基函数的动力学统计信息。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT weights FROM meta_weights WHERE layer_name = ?",
            ("basis_stats",)
        ).fetchone()
        if not row:
            return False
        data = json.loads(row[0])
        for item in data:
            b = basis_registry.get(item["id"])
            if b is not None:
                b.activation_count = item.get("activation_count", 0)
                b.strength_sum = item.get("strength_sum", 0.0)
                b.strength_history = item.get("strength_history", [])
                b.created_at = item.get("created_at", 0.0)
                b.last_activated = item.get("last_activated", 0.0)
                b.parent_ids = item.get("parent_ids", [])
                b.generation = item.get("generation", 0)
        logger.info("Restored basis stats for %d functions", len(data))
        return True

    def restore_neural_bridge(self, bridge) -> bool:
        """从数据库恢复 NeuralBridge 的投影矩阵。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT weights FROM meta_weights WHERE layer_name = ?",
            ("neural_bridge",)
        ).fetchone()
        if not row:
            return False
        data = json.loads(row[0])
        if data.get("P_decode"):
            bridge.P_decode = np.array(data["P_decode"], dtype=np.float32)
        if data.get("P_encode"):
            bridge.P_encode = np.array(data["P_encode"], dtype=np.float32)
        bridge.token_ids = data.get("token_ids", [])
        bridge.token_strs = data.get("token_strs", [])
        bridge.d_B = data.get("d_B", 0)
        bridge.n_tokens = data.get("n_tokens", 0)
        bridge._token_to_idx = {tid: i for i, tid in enumerate(bridge.token_ids)}
        bridge._initialized = True
        logger.info("Restored NeuralBridge (%d tokens)", bridge.n_tokens)
        return True

    # ---- Metadata ----

    def set_meta(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None
