# M-Engine: Adaptive Cognitive Memory Engine

A cognitive architecture that treats memory as spectral compression over interpretable basis functions and unifies six traditionally independent AI modules through a single replicator dynamics equation.

## Quick Start

```bash
pip install -r m_engine/requirements.txt
export DEEPSEEK_API_KEY=sk-your-key
python -m m_engine.cli
```

## Commands

```
ask <question>        Query the memory from a perspective
store <text>          Store a fact
feedback <id> <score> Give feedback (-1.0 to 1.0)
show spectrum <id>    View fact spectrum
show dynamics         View population state
show safety           View safety monitoring
show neural           View neural-symbolic readings
explore               Detect cognitive gaps
analogies             Find structural analogies
evolve                Force population evolution
save / load           Persist state
```

## Architecture

Three nested game layers: micro (spectrum competition), meso (basis evolution), macro (safety modulation). All behavior emerges from:

$$\dot{x}_i = x_i(\pi_i - \bar{\pi}) + \sum_j(x_j Q_{ji} - x_i Q_{ij}) + \mu \cdot \text{Mutation}(x_i)$$

## Tests

```bash
python -m pytest m_engine/tests/ -v   # 41 automated tests
python m_engine/tests/test_deepseek.py  # Full capability verification
```

## Paper

Ziheng Pan. *Adaptive Cognitive Memory Engine: Unifying Multi-Perspective Recall, Preference Modulation, and Basis Function Evolution via Replicator Dynamics.* Submitted to Cognitive Systems Research, 2026.

## License

MIT License
