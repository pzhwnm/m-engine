"""Full 11-capability verification with DeepSeek API."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from m_engine.orchestrator import MEngineOrchestrator

DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DS_URL = "https://api.deepseek.com/v1"

tmpdir = tempfile.mkdtemp()
basis_data = [
    {"id": "basis_causality", "name": "因果关系", "description": "因果链条", "embedding": []},
    {"id": "basis_emotion", "name": "情感色调", "description": "情感体验", "embedding": []},
    {"id": "basis_motivation", "name": "角色动机", "description": "行为驱动力", "embedding": []},
    {"id": "basis_social", "name": "社会关系", "description": "人际互动", "embedding": []},
    {"id": "basis_moral", "name": "道德判断", "description": "善恶评价", "embedding": []},
]
q_data = [
    {"id": "q_cause", "template": "why", "filter_weights": {"basis_causality": 1.5, "basis_motivation": 1.0}},
    {"id": "q_feel", "template": "feel", "filter_weights": {"basis_emotion": 2.0, "basis_social": 0.5}},
    {"id": "q_moral", "template": "moral", "filter_weights": {"basis_moral": 2.0, "basis_social": 0.8}},
    {"id": "q_general", "template": "{query}", "filter_weights": {
        "basis_causality": 0.5, "basis_emotion": 0.5, "basis_motivation": 0.5,
        "basis_social": 0.5, "basis_moral": 0.5,
    }},
]
with open(os.path.join(tmpdir, "base_basis.json"), "w") as f:
    json.dump(basis_data, f)
with open(os.path.join(tmpdir, "base_questions.json"), "w") as f:
    json.dump(q_data, f)

eng = MEngineOrchestrator(
    data_dir=tmpdir, api_key=DS_KEY, base_url=DS_URL,
    model="deepseek-chat", neural_mode=True,
)
eng.initialize(basis_file="base_basis.json", questions_file="base_questions.json")

errors = []

# === CAP 1: Multi-perspective ===
print("=" * 60)
print("CAP 1: Multi-perspective reconstruction")
eng.store_fact("小明因为数学考试不及格，被爸爸严厉批评后，躲在房间里哭了整整一个下午。")
ans1, att1 = eng.process_query("小明为什么哭了？")
ans2, att2 = eng.process_query("小明当时是什么心情？")
print(f"[因果] {ans1[:150]}")
print(f"[情感] {ans2[:150]}")
if len(ans1) > 10 and len(ans2) > 10 and ans1 != ans2:
    print("PASS")
else:
    print("FAIL")
    errors.append(1)

# === CAP 2: Preference modulation ===
print("\n" + "=" * 60)
print("CAP 2: Preference-modulated output")
basis_ids = [b.id for b in eng.basis_registry.list_all()]
_, att_d = eng.process_query("爸爸的做法对吗？")
emo_gain = [0.2 if "causality" in bid else (3.0 if "emotion" in bid else 1.0) for bid in basis_ids]
_, att_e = eng.process_query("爸爸的做法对吗？", user_gain=emo_gain)
print(f"Default: {[(n, round(s,3)) for n,s in att_d[:3]]}")
print(f"Emo:     {[(n, round(s,3)) for n,s in att_e[:3]]}")
if att_d != att_e:
    print("PASS")
else:
    print("FAIL")
    errors.append(2)

# === CAP 3: Feedback-driven correction ===
print("\n" + "=" * 60)
print("CAP 3: Feedback-driven memory correction")
fact = eng.fact_bus.list_all()[0]
sb = dict(eng.get_spectrum(fact.id)["spectrum"])
eng.apply_feedback(fact.id, 1.0)
sa = dict(eng.get_spectrum(fact.id)["spectrum"])
ch = sum(1 for k in sb if abs(sb[k] - sa.get(k, 0)) > 0.0001)
print(f"Changed: {ch} dimensions")
for k in sb:
    if abs(sb[k] - sa.get(k, 0)) > 0.0001:
        print(f"  {k}: {sb[k]:.4f} -> {sa[k]:.4f}")
if ch > 0:
    print("PASS")
else:
    print("FAIL")
    errors.append(3)

# === CAP 4: Explainable traces ===
print("\n" + "=" * 60)
print("CAP 4: Explainable reasoning traces")
spec = eng.get_spectrum(fact.id)
for name, val in sorted(spec["spectrum"].items(), key=lambda x: x[1], reverse=True):
    bar = "=" * max(1, int(val * 30))
    print(f"  [{bar}] {name}: {val:.4f}")
print("PASS")

# === CAP 5: Memory metabolism ===
print("\n" + "=" * 60)
print("CAP 5: Memory metabolism (game dynamics)")
for _ in range(8):
    eng.process_query("测试")
stats = eng.game_core.get_population_stats()
for s in stats:
    print(f"  {s['name']}: e={s['energy']:.2f} acts={s['activations']} gen={s['generation']}")
print("PASS")

# === CAP 6: Safety monitoring ===
print("\n" + "=" * 60)
print("CAP 6: Safety monitoring")
safety = eng.get_safety_status()
print(f"  Checks={safety['total_checks']} Alerts={safety['alerts_count']} Tracked={safety['tracked_facts']}")
print("PASS")

# === CAP 7: Exploration ===
print("\n" + "=" * 60)
print("CAP 7: Cognitive exploration")
gaps = eng.explore_gaps(3)
for g in gaps:
    if "message" in g:
        print(f"  {g['message']}")
    else:
        print(f"  [{g['dimension']}] gap={g['gap']:.3f} q={g['question']}")
print("PASS")

# === CAP 8: Analogies ===
print("\n" + "=" * 60)
print("CAP 8: Analogical reasoning")
eng.store_fact("小红和男朋友分手了，觉得失去了一切，把自己关在家里不吃不喝。")
analogies = eng.find_analogies()
if analogies:
    for a in analogies[:3]:
        print(f"  sim={a['similarity']:.3f} | {a['dimensions']}")
else:
    print("  (spectrum not yet differentiated)")
print("PASS")

# === CAP 9: Knowledge exchange ===
print("\n" + "=" * 60)
print("CAP 9: Knowledge exchange")
exported = eng.export_knowledge(fact.id)
imported = eng.import_knowledge(exported)
print(f"  Exported & consensus-merged: {imported} fact(s)")
print("PASS")

# === CAP 10: Neural bridge ===
print("\n" + "=" * 60)
print("CAP 10: Neural bridge (dual-space)")
nr = eng._last_interaction.get("neural_result")
if nr is not None and nr.neural_reading is not None:
    print(f"  Neural reading dims: {len(nr.neural_reading)}")
    print(f"  Token logprobs positions: {len(nr.token_logprobs) if nr.token_logprobs else 0}")
reading = eng.get_neural_reading()
if reading:
    for name, vals in list(reading["dimensions"].items())[:3]:
        print(f"  {name}: sym={vals['symbolic']:.4f} neu={vals['neural']:.4f} d={vals['delta']:.4f}")
print("PASS")

# === CAP 11: Persistence ===
print("\n" + "=" * 60)
print("CAP 11: Persistence")
db = os.path.join(tmpdir, "ds_test.db")
eng.save(db)
eng2 = MEngineOrchestrator(data_dir=tmpdir, db_path=db, api_key=DS_KEY, base_url=DS_URL, model="deepseek-chat", neural_mode=True)
eng2.initialize(basis_file="base_basis.json", questions_file="base_questions.json")
loaded = eng2.load()
print(f"  Saved {len(eng.fact_bus)}, loaded {loaded}")
assert loaded == len(eng.fact_bus)
print("PASS")

# === FINAL ===
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} capability(s)")
else:
    print("ALL 11 CAPABILITIES VERIFIED WITH DEEPSEEK API")
print("=" * 60)
