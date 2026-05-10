"""
Route B-3: Parameter Sensitivity Analysis.
Varies alpha, beta, gamma, delta and measures impact on:
  - Perspective diversity (attention std)
  - Evolution rate (new agents born)
  - Safety alerts triggered
Runs without real API to speed up (only internal game dynamics).
"""
import json, os, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from m_engine.core.evolutionary_game_core import GameConfig

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Setup facts
facts_text = [
    "Xiao Ming failed his math exam and was harshly scolded by his father. He cried all afternoon.",
    "Xiao Hong broke up with her boyfriend and felt she lost everything.",
    "Father regretted yelling but did not know how to apologize.",
    "Xiao Ming scored 85 on the makeup exam and told everyone the good news.",
]

# Quick simulation function (no API calls, just game dynamics)
def simulate(params, n_rounds=30):
    tmpdir = tempfile.mkdtemp()
    basis_data = [
        {"id":f"b_{n}","name":f"{n}","description":"","embedding":[]}
        for n in ["Causality","Emotion","Motivation","Social","Moral"]
    ]
    q_data = [{"id":"q_gen","template":"{q}","filter_weights":{f"b_{n}":0.5 for n in ["Causality","Emotion","Motivation","Social","Moral"]}}]
    with open(os.path.join(tmpdir,"base_basis.json"),"w") as f: json.dump(basis_data,f)
    with open(os.path.join(tmpdir,"base_questions.json"),"w") as f: json.dump(q_data,f)

    from m_engine.orchestrator import MEngineOrchestrator
    config = GameConfig(**params)
    eng = MEngineOrchestrator(data_dir=tmpdir, api_key="", neural_mode=False, game_config=config)
    eng.initialize(basis_file="base_basis.json", questions_file="base_questions.json")
    for t in facts_text:
        eng.store_fact(t)

    diversity_vals = []
    alerts_count = 0
    init_pop = len(eng.basis_registry)

    for r in range(n_rounds):
        q = f"query_{r % 3}"
        _, att = eng.process_query(q)
        att_vals = [s for _, s in att]
        diversity_vals.append(float(np.std(att_vals)) if len(att_vals) > 1 else 0.0)

        safety = eng.get_safety_status()
        alerts_count += safety.get("alerts_count", 0)

    final_pop = len(eng.basis_registry)
    return {
        "mean_diversity": float(np.mean(diversity_vals)),
        "final_diversity": float(diversity_vals[-1]),
        "new_agents": final_pop - init_pop,
        "alerts": alerts_count,
    }

# Parameter grid
default = {"alpha_recon":1.0, "beta_novelty":0.3, "gamma_safety":0.5, "delta_resource":0.1}

param_ranges = {
    "alpha (Reconstruction)": {"key":"alpha_recon", "values":[0.0, 0.5, 1.0, 2.0, 5.0]},
    "beta (Exploration)":    {"key":"beta_novelty", "values":[0.0, 0.2, 0.5, 1.0, 2.0]},
    "gamma (Safety)":        {"key":"gamma_safety", "values":[0.0, 0.25, 0.5, 1.0, 2.0]},
    "delta (Metabolic)":     {"key":"delta_resource","values":[0.0, 0.05, 0.1, 0.5, 1.0]},
}

all_results = {}
print("Running parameter sensitivity analysis...")
for param_name, param_info in param_ranges.items():
    key = param_info["key"]
    print(f"\n  {param_name}:")
    results_for_param = []
    for val in param_info["values"]:
        params = dict(default)
        params[key] = val
        r = simulate(params, n_rounds=30)
        results_for_param.append({"value": val, **r})
        print(f"    {key}={val:.2f}: diversity={r['mean_diversity']:.4f} agents={r['new_agents']} alerts={r['alerts']}")
    all_results[param_name] = results_for_param

# Save
with open("data/sensitivity_results.json","w") as f:
    json.dump(all_results, f, indent=2)
print("\nData saved.")

# ==========================================
# FIGURE: Parameter Sensitivity Grid
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0']

for idx, (param_name, results) in enumerate(all_results.items()):
    ax = axes[idx]
    values = [r["value"] for r in results]
    diversity = [r["mean_diversity"] for r in results]
    agents = [r["new_agents"] for r in results]

    ax2 = ax.twinx()
    line1 = ax.plot(values, diversity, 'o-', color=colors[idx], lw=2, markersize=8, label='Diversity')
    line2 = ax2.plot(values, agents, 's--', color='#FF9800', lw=2, markersize=8, label='New Agents')

    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_ylabel('Mean Diversity', fontsize=11, color=colors[idx])
    ax2.set_ylabel('New Agents Born', fontsize=11, color='#FF9800')
    ax.set_title(param_name, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig("figures/fig_sensitivity.pdf", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/fig_sensitivity.pdf")

# Summary
print("\n=== SENSITIVITY SUMMARY ===")
for param_name, results in all_results.items():
    divs = [r["mean_diversity"] for r in results]
    agents_range = f"{min(r['new_agents'] for r in results)}-{max(r['new_agents'] for r in results)}"
    print(f"  {param_name}: diversity range [{min(divs):.4f}, {max(divs):.4f}], agents: {agents_range}")
