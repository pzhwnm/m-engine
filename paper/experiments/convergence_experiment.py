"""
Route B-1: Convergence Analysis Experiment.
Runs 50+ interactive query-feedback cycles, tracks spectrum evolution,
basis function energy trajectories, and population dynamics.
"""
import json, os, sys, tempfile, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from m_engine.orchestrator import MEngineOrchestrator

DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DS_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")
DS_URL = "https://api.deepseek.com/v1"
N_ROUNDS = 50
SAVE_EVERY = 5  # save spectrum snapshot every N rounds

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Setup engine with real basis functions
tmpdir = tempfile.mkdtemp()
basis_data = [
    {"id":"basis_causality","name":"Causality","description":"Causal chain","embedding":[]},
    {"id":"basis_emotion","name":"Emotion","description":"Emotional tone","embedding":[]},
    {"id":"basis_motivation","name":"Motivation","description":"Drives and goals","embedding":[]},
    {"id":"basis_social","name":"Social","description":"Interpersonal relations","embedding":[]},
    {"id":"basis_moral","name":"Moral","description":"Right/wrong judgment","embedding":[]},
]
q_data = [
    {"id":"q_why","template":"why","filter_weights":{"basis_causality":1.5,"basis_motivation":1.0}},
    {"id":"q_feel","template":"feel","filter_weights":{"basis_emotion":2.0,"basis_social":0.5}},
    {"id":"q_moral","template":"moral","filter_weights":{"basis_moral":2.0,"basis_social":0.8}},
    {"id":"q_general","template":"{query}","filter_weights":{
        "basis_causality":0.5,"basis_emotion":0.5,"basis_motivation":0.5,"basis_social":0.5,"basis_moral":0.5
    }},
]
with open(os.path.join(tmpdir,"base_basis.json"),"w") as f: json.dump(basis_data,f)
with open(os.path.join(tmpdir,"base_questions.json"),"w") as f: json.dump(q_data,f)

eng = MEngineOrchestrator(data_dir=tmpdir, api_key=DS_KEY, base_url=DS_URL, model="deepseek-chat", neural_mode=False)
eng.initialize(basis_file="base_basis.json", questions_file="base_questions.json")

# Store diverse facts
facts_text = [
    "Xiao Ming failed his math exam and was harshly scolded by his father. He hid in his room and cried all afternoon.",
    "Xiao Hong broke up with her boyfriend and felt she lost everything. She locked herself at home without eating.",
    "Father later regretted yelling at Xiao Ming but did not know how to apologize.",
    "A week later, Xiao Ming scored 85 on the makeup exam and immediately told Xiao Hong and his father.",
]
for t in facts_text:
    eng.store_fact(t)

print(f"Initialized: {len(eng.fact_bus)} facts, {len(eng.basis_registry)} basis functions")
print(f"Running {N_ROUNDS} interaction rounds...")

# Tracking data
spectrum_history = {f.id: [] for f in eng.fact_bus.list_all()}
energy_history = []
population_history = []
diversity_history = []
round_times = []

queries_causal = ["Why did Xiao Ming cry?", "What caused this situation?", "Why did father react that way?"]
queries_emotional = ["How did Xiao Ming feel?", "What was Xiao Hong's emotional state?", "How did father feel afterwards?"]
feedbacks = [1.0, 1.0, -0.5, 0.8, 1.0, -0.3, 0.5, 1.0, 1.0, -0.2]

for r in range(N_ROUNDS):
    t0 = time.time()

    # Rotate query types and feedback
    q = queries_causal[r % 3] if r % 2 == 0 else queries_emotional[r % 3]
    fb = feedbacks[r % len(feedbacks)]

    ans, att = eng.process_query(q)

    # Apply feedback to the first fact
    fact = eng.fact_bus.list_all()[0]
    eng.apply_feedback(fact.id, fb)

    # Record data
    snapshot = {}
    for f in eng.fact_bus.list_all():
        snapshot[f.id] = dict(f.spectrum)
    spectrum_history[fact.id].append(dict(fact.spectrum))

    # Population stats
    stats = eng.game_core.get_population_stats()
    energies = {s["name"]: s["energy"] for s in stats}
    energy_history.append(energies)
    population_history.append(len(eng.basis_registry))

    # Diversity: compute std of attention values (higher = more differentiated)
    att_values = [s for _, s in att]
    diversity_history.append(float(np.std(att_values)) if len(att_values) > 1 else 0.0)

    round_times.append(time.time() - t0)

    if (r+1) % 10 == 0:
        print(f"  Round {r+1}/{N_ROUNDS} | Population: {len(eng.basis_registry)} | "
              f"Diversity: {diversity_history[-1]:.4f} | "
              f"Avg time: {np.mean(round_times[-10:]):.1f}s")

# Save raw data
data = {
    "spectrum_history": {k: v for k, v in spectrum_history.items()},
    "energy_history": [{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in e.items()} for e in energy_history],
    "population_history": population_history,
    "diversity_history": diversity_history,
    "round_times": round_times,
}
with open("data/convergence_data.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nRaw data saved to data/convergence_data.json")

# ==========================================
# FIGURE 1: Spectrum Convergence
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: spectrum traces for the first fact
basis_ids_ordered = [b.id for b in eng.basis_registry.list_all()]
basis_names = [b.name for b in eng.basis_registry.list_all()]
colors = ['#FF5722','#E91E63','#9C27B0','#2196F3','#4CAF50','#FF9800','#795548','#607D8B','#00BCD4','#CDDC39']

spec_snapshots = spectrum_history[list(eng.fact_bus.list_all())[0].id]
rounds_with_data = list(range(0, N_ROUNDS))

# Get the last values for each dimension at each recorded round
# Since we recorded every round, just extract
for i, bid in enumerate(basis_ids_ordered):
    vals = [spec_snapshots[r].get(bid, 0.0) for r in range(len(spec_snapshots))]
    axes[0].plot(range(len(vals)), vals, color=colors[i], label=basis_names[i], lw=1.5, alpha=0.8)

axes[0].set_xlabel("Interaction Round", fontsize=11)
axes[0].set_ylabel("Spectrum Value", fontsize=11)
axes[0].set_title("Spectrum Convergence Over 50 Rounds", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].grid(alpha=0.3)

# Right: diversity metric
axes[1].plot(diversity_history, 'b-', lw=1.5, alpha=0.7)
axes[1].set_xlabel("Interaction Round", fontsize=11)
axes[1].set_ylabel("Attention Std (Diversity)", fontsize=11)
axes[1].set_title("Perspective Diversity Over Time", fontsize=12, fontweight='bold')
# Add trend line
z = np.polyfit(range(len(diversity_history)), diversity_history, 1)
p = np.poly1d(z)
axes[1].plot(range(len(diversity_history)), p(range(len(diversity_history))), 'r--', lw=2, label='Trend')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig_convergence.pdf", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/fig_convergence.pdf")

# ==========================================
# FIGURE 2: Agent Energy Trajectories
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))

all_agent_names = list(dict.fromkeys([n for e in energy_history for n in e.keys()]))
for name_idx, name in enumerate(all_agent_names):
    vals = [energy_history[r].get(name, 0.0) for r in range(len(energy_history))]
    if max(vals) > 0.1:
        is_new = name not in basis_names
        style = '--' if is_new else '-'
        ax.plot(vals, style, color=colors[name_idx % len(colors)], label=f"{name}{' (new)' if is_new else ''}", lw=1.5, alpha=0.8)

# Check for new agents
all_names = set()
for e in energy_history:
    all_names.update(e.keys())
new_agents = all_names - set(basis_names)
for name in new_agents:
    vals = [energy_history[r].get(name, 0.0) for r in range(len(energy_history))]
    if max(vals) > 0.1:
        ax.plot(vals, '--', lw=2, label=f"{name} (new)", alpha=0.9)

ax.set_xlabel("Interaction Round", fontsize=11)
ax.set_ylabel("Agent Energy", fontsize=11)
ax.set_title("Basis Function Energy Trajectories", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig_energy_trajectories.pdf", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/fig_energy_trajectories.pdf")

# ==========================================
# FIGURE 3: Population Size Over Time
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(population_history, 'g-', lw=2, marker='o', markersize=3)
ax.set_xlabel("Interaction Round", fontsize=11)
ax.set_ylabel("Number of Basis Functions", fontsize=11)
ax.set_title("Population Size Evolution", fontsize=12, fontweight='bold')
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_population.pdf", dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: figures/fig_population.pdf")

# Summary stats
print(f"\n=== CONVERGENCE SUMMARY ===")
print(f"Final spectrum std: {np.std(list(spec_snapshots[-1].values())):.4f}")
print(f"Initial spectrum std: {np.std(list(spec_snapshots[0].values())):.4f}")
print(f"Final population size: {population_history[-1]}")
print(f"Initial population size: {population_history[0]}")
print(f"Mean diversity: {np.mean(diversity_history):.4f}")
print(f"Final diversity: {diversity_history[-1]:.4f}")
print(f"Mean round time: {np.mean(round_times):.1f}s")
print(f"New agents born: {population_history[-1] - population_history[0]}")
