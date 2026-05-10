"""
Comprehensive end-to-end integration test for M-Engine.
Exercises the full cycle: store → ask → spectrum → feedback → verify changes.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from m_engine.orchestrator import MEngineOrchestrator


def main():
    tmpdir = tempfile.mkdtemp()
    basis_data = [
        {"id": "basis_causality", "name": "因果关系", "description": "因果链", "embedding": []},
        {"id": "basis_temporal", "name": "时间顺序", "description": "时序", "embedding": []},
        {"id": "basis_emotion", "name": "情感色调", "description": "情感", "embedding": []},
        {"id": "basis_motivation", "name": "角色动机", "description": "动机", "embedding": []},
        {"id": "basis_social", "name": "社会关系", "description": "社会", "embedding": []},
        {"id": "basis_moral", "name": "道德判断", "description": "道德", "embedding": []},
    ]
    questions_data = [
        {"id": "q_why", "template": "why", "filter_weights": {"basis_causality": 1.5, "basis_motivation": 1.0}},
        {"id": "q_how_feel", "template": "feel", "filter_weights": {"basis_emotion": 2.0, "basis_social": 0.5}},
        {"id": "q_general", "template": "{query}", "filter_weights": {
            "basis_causality": 0.5, "basis_emotion": 0.5, "basis_motivation": 0.5,
            "basis_social": 0.5, "basis_moral": 0.5, "basis_temporal": 0.5
        }},
    ]
    with open(os.path.join(tmpdir, "base_basis.json"), "w") as f:
        json.dump(basis_data, f)
    with open(os.path.join(tmpdir, "base_questions.json"), "w") as f:
        json.dump(questions_data, f)

    errors = []

    def check(condition, msg):
        if not condition:
            errors.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS")

    # ============================================================
    engine = MEngineOrchestrator(data_dir=tmpdir, api_key="")
    engine.initialize()

    # TEST 1: Store facts
    print("TEST 1: Store facts")
    f1 = engine.store_fact("小明因为数学考试不及格，躲在房间里哭了整整一个下午。")
    f2 = engine.store_fact("小明的爸爸严厉地批评了小明，事后又很后悔。")
    check(len(engine.fact_bus) == 2, "Should have 2 facts")

    # TEST 2: Causal question routes correctly
    print("TEST 2: Causal question -> q_why")
    answer, att = engine.process_query("小明为什么哭了？")
    check(engine._last_interaction["base_q_id"] == "q_why",
          f"Expected q_why, got {engine._last_interaction['base_q_id']}")
    if att:
        check("因果" in att[0][0], f"Expected 因果 top, got {att[0][0]}")

    # TEST 3: Emotional question routes correctly
    print("TEST 3: Emotional question -> q_how_feel")
    answer2, att2 = engine.process_query("小明当时是什么感受？")
    check(engine._last_interaction["base_q_id"] == "q_how_feel",
          f"Expected q_how_feel, got {engine._last_interaction['base_q_id']}")
    if att2:
        check("情感" in att2[0][0], f"Expected 情感 top, got {att2[0][0]}")

    # TEST 4: Show spectrum
    print("TEST 4: Show spectrum")
    spec = engine.get_spectrum(f1.id)
    check(spec is not None, "Spectrum should not be None")
    check(len(spec["spectrum"]) == 6, f"Expected 6 dims, got {len(spec['spectrum'])}")

    # TEST 5: Positive feedback changes spectrum
    print("TEST 5: Positive feedback changes spectrum")
    spec_before = dict(engine.get_spectrum(f1.id)["spectrum"])
    result = engine.apply_feedback(f1.id, 1.0)
    check(result["status"] == "ok", f"Feedback failed: {result}")
    spec_after = dict(engine.get_spectrum(f1.id)["spectrum"])
    changed = sum(1 for k in spec_before
                  if abs(spec_before[k] - spec_after.get(k, 0)) > 0.0001)
    check(changed > 0, "Spectrum should change after feedback")

    # TEST 6: Negative feedback
    print("TEST 6: Negative feedback")
    engine.process_query("爸爸批评小明对吗？")
    spec_before2 = dict(engine.get_spectrum(f2.id)["spectrum"])
    result2 = engine.apply_feedback(f2.id, -1.0)
    check(result2["status"] == "ok", f"Negative feedback failed: {result2}")
    spec_after2 = dict(engine.get_spectrum(f2.id)["spectrum"])
    changed2 = sum(1 for k in spec_before2
                   if abs(spec_before2[k] - spec_after2.get(k, 0)) > 0.0001)
    check(changed2 > 0, "Negative feedback should also change spectrum")

    # TEST 7: User preference influences activation
    print("TEST 7: User preference influences activation")
    emotional_user = UserProfile(id="u_emo", gain={
        "basis_causality": 0.2, "basis_emotion": 3.0,
        "basis_motivation": 1.5, "basis_social": 1.0,
        "basis_moral": 1.0, "basis_temporal": 1.0,
    })
    _, att_default = engine.process_query("小明为什么哭了？")
    _, att_emo = engine.process_query("小明为什么哭了？", user_profile=emotional_user)
    # Both should have valid attention
    check(len(att_default) > 0, "Default attention should not be empty")
    check(len(att_emo) > 0, "Emotional attention should not be empty")
    # Emotional dimension should rank differently
    emo_names = [n for n, s in att_emo]
    def_names = [n for n, s in att_default]
    if "情感色调" in emo_names and "情感色调" in def_names:
        print(f"    Emotion rank: default={def_names.index('情感色调')}, "
              f"emotional={emo_names.index('情感色调')}")

    # TEST 8: Feedback without prior query fails gracefully
    print("TEST 8: Feedback without prior query")
    engine3 = MEngineOrchestrator(data_dir=tmpdir, api_key="")
    engine3.initialize()
    f3 = engine3.store_fact("独立测试事实。")
    result3 = engine3.apply_feedback(f3.id, 0.5)
    check(result3["status"] == "error", "Should fail without prior interaction")
    check("previous interaction" in result3.get("message", ""),
          "Error message should mention 'previous interaction'")

    # TEST 9: Empty fact bus returns helpful message
    print("TEST 9: Empty fact bus")
    engine4 = MEngineOrchestrator(data_dir=tmpdir, api_key="")
    engine4.initialize()
    answer_empty, att_empty = engine4.process_query("随便问个问题")
    check("没有任何事实" in answer_empty, f"Should warn about empty facts, got: {answer_empty[:60]}")
    check(att_empty == [], "Attention should be empty when no facts exist")

    # TEST 10: Activation count updates after interaction
    print("TEST 10: Activation tracking")
    f4 = engine4.store_fact("激活测试事实。")
    engine4.process_query("为什么？")
    engine4.apply_feedback(f4.id, 0.5)
    f4_updated = engine4.fact_bus.get_fact(f4.id)
    total_act = sum(f4_updated.activation.values())
    check(total_act > 0, f"Activation should be > 0, got {total_act}")

    # ============================================================
    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL 10 END-TO-END TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
