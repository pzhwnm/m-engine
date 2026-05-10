"""
单元测试：对各核心模块进行独立测试，mock LLM 调用。
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import numpy as np

from m_engine.core.basis_registry import BasisRegistry, BasisFunction
from m_engine.core.fact_bus import FactBus, Fact
from m_engine.core.question_router import QuestionRouter, BaseQuestion
from m_engine.core.preference_modem import PreferenceModem, UserProfile, ModelConstraints
from m_engine.core.m_algebra import MAlgebraCore
from m_engine.core.decoder import Decoder
from m_engine.core.meta_updater import MetaUpdater


# ============================================================
# BasisRegistry 测试
# ============================================================

class TestBasisRegistry:
    def test_register_and_get(self):
        registry = BasisRegistry()
        b = BasisFunction(id="b1", name="因果", description="因果链条")
        registry.register(b)
        assert registry.get("b1") is b
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = BasisRegistry()
        registry.register(BasisFunction(id="b1", name="因果"))
        registry.register(BasisFunction(id="b2", name="情感"))
        assert len(registry.list_all()) == 2

    def test_get_embedding(self):
        registry = BasisRegistry()
        b = BasisFunction(id="b1", name="因果", embedding=[0.1, 0.2, 0.3])
        registry.register(b)
        assert registry.get_embedding("b1") == [0.1, 0.2, 0.3]
        assert registry.get_embedding("nonexistent") == []

    def test_get_names(self):
        registry = BasisRegistry()
        registry.register(BasisFunction(id="b1", name="因果"))
        registry.register(BasisFunction(id="b2", name="情感"))
        names = registry.get_names()
        assert "因果" in names
        assert "情感" in names

    def test_load_from_json(self):
        data = [
            {"id": "b1", "name": "因果", "description": "因果链", "embedding": [0.1]},
            {"id": "b2", "name": "情感", "description": "情感色", "embedding": [0.2]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            registry = BasisRegistry()
            registry.load_from_json(tmp_path)
            assert len(registry) == 2
            assert registry.get("b1").name == "因果"
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        registry = BasisRegistry()
        with pytest.raises(FileNotFoundError):
            registry.load_from_json("/nonexistent/path.json")


# ============================================================
# FactBus 测试
# ============================================================

class TestFact:
    def test_get_spectrum_vector(self):
        f = Fact(
            id="f1",
            raw_text="测试",
            spectrum={"basis_a": 0.5, "basis_b": 0.3, "basis_c": 0.8},
        )
        vec = f.get_spectrum_vector(["basis_a", "basis_b", "basis_c"])
        assert vec.tolist() == pytest.approx([0.5, 0.3, 0.8], abs=1e-5)

    def test_get_spectrum_vector_missing(self):
        f = Fact(id="f1", spectrum={"basis_a": 0.5})
        vec = f.get_spectrum_vector(["basis_a", "basis_b"])
        assert vec.tolist() == [0.5, 0.0]


class TestFactBus:
    def test_add_and_get(self):
        bus = FactBus()
        f = Fact(id="f1", raw_text="hello")
        bus.add_fact(f)
        assert bus.get_fact("f1") is f
        assert len(bus) == 1

    def test_update_spectrum(self):
        bus = FactBus()
        f = Fact(id="f1")
        bus.add_fact(f)
        bus.update_spectrum("f1", "basis_a", 0.75)
        assert f.spectrum["basis_a"] == 0.75

    def test_update_spectrum_clamp(self):
        bus = FactBus()
        f = Fact(id="f1")
        bus.add_fact(f)
        bus.update_spectrum("f1", "basis_a", 1.5)  # 应裁剪到 1.0
        assert f.spectrum["basis_a"] == 1.0
        bus.update_spectrum("f1", "basis_b", -0.5)  # 应裁剪到 0.0
        assert f.spectrum["basis_b"] == 0.0

    def test_update_spectrum_nonexistent(self):
        bus = FactBus()
        bus.update_spectrum("nonexistent", "basis_a", 0.5)  # 不该崩溃

    def test_activate(self):
        bus = FactBus()
        f = Fact(id="f1")
        bus.add_fact(f)
        bus.update_activation("f1", "q_why", delta=2)
        assert f.activation["q_why"] == 2
        bus.update_activation("f1", "q_why", delta=1)
        assert f.activation["q_why"] == 3

    def test_retrieve(self):
        bus = FactBus()
        f1 = Fact(id="f1", raw_text="aaa", embedding=[1.0, 0.0, 0.0])
        f2 = Fact(id="f2", raw_text="bbb", embedding=[0.0, 1.0, 0.0])
        f3 = Fact(id="f3", raw_text="ccc", embedding=[0.9, 0.1, 0.0])
        bus.add_fact(f1)
        bus.add_fact(f2)
        bus.add_fact(f3)

        results = bus.retrieve([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "f1"  # 最相似

    def test_retrieve_empty(self):
        bus = FactBus()
        assert bus.retrieve([1.0, 0.0]) == []


# ============================================================
# QuestionRouter 测试
# ============================================================

class TestQuestionRouter:
    def setup_method(self):
        self.router = QuestionRouter()
        self.router.register(BaseQuestion(
            id="q_why", template="为什么...",
            filter_weights={"basis_causality": 1.5, "basis_motivation": 1.0},
        ))
        self.router.register(BaseQuestion(
            id="q_how_feel", template="感觉...",
            filter_weights={"basis_emotion": 2.0, "basis_causality": 0.2},
        ))
        self.router.register(BaseQuestion(
            id="q_general", template="{query}",
            filter_weights={"basis_causality": 0.5, "basis_emotion": 0.5},
        ))

    def test_parse_with_keyword(self):
        base_q, weights = self.router.parse("小明为什么会哭？")
        assert base_q.id == "q_why"

    def test_parse_no_keyword(self):
        base_q, weights = self.router.parse("今天天气如何？")
        assert base_q.id == "q_general"

    def test_parse_emotion_keyword(self):
        base_q, weights = self.router.parse("小明的感受是怎样的？")
        assert base_q.id == "q_how_feel"

    def test_load_from_json(self):
        data = [
            {"id": "q_test", "template": "测试{query}",
             "filter_weights": {"basis_test": 1.0}},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            router = QuestionRouter()
            router.load_from_json(tmp_path)
            assert router.get("q_test") is not None
        finally:
            os.unlink(tmp_path)


# ============================================================
# PreferenceModem 测试
# ============================================================

class TestPreferenceModem:
    def test_apply_gain_unity(self):
        modem = PreferenceModem()
        beta = [0.5, 0.3, 0.2]
        result = modem.apply_gain(beta, [], [])
        assert sum(result) == pytest.approx(1.0, abs=1e-5)

    def test_apply_gain_user_emphasis(self):
        modem = PreferenceModem()
        # base 均匀，用户对 dim0 增益 2.0
        beta = [0.5, 0.5]
        user_gain = [2.0, 1.0]
        result = modem.apply_gain(beta, user_gain, [])
        # dim0 应该被放大
        assert result[0] > result[1]

    def test_apply_gain_model_damping(self):
        modem = PreferenceModem()
        beta = [0.5, 0.5]
        model_gain = [0.5, 1.0]
        result = modem.apply_gain(beta, [], model_gain)
        assert result[0] < result[1]

    def test_apply_gain_normalization(self):
        modem = PreferenceModem()
        beta = [1.0, 2.0, 3.0]
        result = modem.apply_gain(beta, [], [])
        assert sum(result) == pytest.approx(1.0, abs=1e-5)

    def test_get_gain_vectors(self):
        modem = PreferenceModem()
        user = UserProfile(id="u1", gain={"a": 1.5, "b": 0.8})
        assert modem.get_user_gain(user) == [1.5, 0.8]

        model = ModelConstraints(gain={"a": 1.0, "b": 2.0})
        assert modem.get_model_gain(model) == [1.0, 2.0]


# ============================================================
# MAlgebraCore 测试
# ============================================================

class TestMAlgebraCore:
    def setup_method(self):
        self.basis = BasisRegistry()
        self.basis.register(BasisFunction(id="b1", name="因果"))
        self.basis.register(BasisFunction(id="b2", name="情感"))
        self.basis.register(BasisFunction(id="b3", name="逻辑"))
        self.algebra = MAlgebraCore(self.basis)

    def test_compute_activation_basic(self):
        f1 = Fact(id="f1", spectrum={"b1": 0.8, "b2": 0.1, "b3": 0.3})
        f2 = Fact(id="f2", spectrum={"b1": 0.2, "b2": 0.9, "b3": 0.1})
        beta = [1.0, 0.0, 0.5]  # 只看因果，不看情感
        pref = [1.0, 1.0, 1.0]  # 中性偏好

        activation, top = self.algebra.compute_activation([f1, f2], beta, pref)
        assert len(activation) == 3
        # 因果(b1) 应该被激活
        assert top[0][0] == "因果" or activation[0] > 0

    def test_compute_activation_empty(self):
        activation, top = self.algebra.compute_activation([], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        assert len(activation) == 0
        assert top == []

    def test_build_prompt(self):
        f1 = Fact(id="f1", raw_text="事实A")
        activations = [("因果", 0.8), ("情感", 0.2)]
        prompt = self.algebra.build_prompt([f1], activations, "为什么？")
        assert "因果" in prompt
        assert "事实A" in prompt
        assert "为什么？" in prompt


# ============================================================
# Decoder 测试
# ============================================================

class TestDecoder:
    def test_mock_decode(self):
        decoder = Decoder(api_key="")  # 空 key 触发 mock 模式
        result = decoder.decode("测试prompt", [0.8, 0.2], ["因果", "情感"])
        assert "模拟回答" in result.text or "因果" in result.text

    def test_decode_without_key(self):
        decoder = Decoder(api_key=None)
        result = decoder.decode("prompt", [0.0, 0.0], ["a", "b"])
        assert "无显著激活" in result.text


# ============================================================
# MetaUpdater 测试
# ============================================================

class TestMetaUpdater:
    def test_compute_delta_shape(self):
        mu = MetaUpdater(d_B=5)
        delta = mu.compute_delta([0.2, 0.2, 0.2, 0.2, 0.2], feedback=1.0)
        assert len(delta) == 5

    def test_compute_delta_positive_vs_negative(self):
        mu = MetaUpdater(d_B=3)
        d_pos = mu.compute_delta([0.3, 0.3, 0.4], feedback=1.0)
        d_neg = mu.compute_delta([0.3, 0.3, 0.4], feedback=-1.0)
        # 正负反馈应产生不同的修正方向
        assert not np.allclose(d_pos, d_neg)

    def test_update(self):
        mu = MetaUpdater(d_B=3, lr=0.5)
        bus = FactBus()
        f = Fact(id="f1", spectrum={"b1": 0.5, "b2": 0.3, "b3": 0.1})
        bus.add_fact(f)

        result = mu.update(
            fact_bus=bus,
            fact_id="f1",
            q_type="q_why",
            beta=[1.0, 0.5, 0.1],
            feedback=1.0,
            basis_ids=["b1", "b2", "b3"],
        )
        assert result["status"] == "ok"
        assert "changes" in result

    def test_update_nonexistent_fact(self):
        mu = MetaUpdater(d_B=3)
        bus = FactBus()
        result = mu.update(bus, "nonexistent", "q", [0.5, 0.5, 0.5],
                           feedback=1.0, basis_ids=[])
        assert result["status"] == "error"
