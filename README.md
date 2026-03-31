# graph-glad-hep

**Graph-level anomaly detection with contrastive learning for LHC new physics searches.**

GSoC 2026 proof-of-concept package — ML4SCI / GENIE project (GENIE1).

[![CI](https://github.com/pulin-chhangawala/graph-glad-hep/actions/workflows/ci.yml/badge.svg)](https://github.com/pulin-chhangawala/graph-glad-hep/actions)
[![Coverage](https://codecov.io/gh/pulin-chhangawala/graph-glad-hep/branch/main/graph/badge.svg)](https://codecov.io/gh/pulin-chhangawala/graph-glad-hep)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

---

## What this is

This package implements the GLADC framework ([Luo et al., 2022](https://doi.org/10.1038/s41598-022-22086-3)) adapted for particle physics collision data at the LHC. Each collision event is represented as a graph (particles → nodes, spatial proximity → edges) and scored by a dual-encoder contrastive learning model trained only on Standard Model background.

Key additions over the original GLADC paper:
- **HEP-specific graph construction** with k-NN + seed edges and physically motivated node features (pT, η, φ, E, ΔR)
- **DisCo mass decorrelation** to prevent the anomaly score from correlating with dijet invariant mass m_jj
- **SIC curve and IS metric** evaluation (HEP community standard)
- **`MockEventGenerator`** — fully synthetic LHC-like events so the entire pipeline runs in CI with no credentials or real data

---

## Installation

```bash
git clone https://github.com/pulin-chhangawala/graph-glad-hep
cd graph-glad-hep
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install -e ".[dev]"
```

---

## Quick start

```python
from graph_glad_hep import MockEventGenerator, GraphGLADC
from graph_glad_hep.evaluate import evaluate, print_results
import numpy as np

# 1. Generate synthetic events (replace with real LHCO data)
gen = MockEventGenerator(seed=42)
dataset = gen.generate_dataset(n_events=500, signal_fraction=0.1)

background_graphs = [g for g, label in dataset if label == 0]
all_graphs       = [g for g, label in dataset]
labels           = np.array([label for _, label in dataset])

# 2. Train on background only
model = GraphGLADC(hidden=64, out_dim=32, epochs=20, batch_size=32)
model.fit(background_graphs, verbose=True)

# 3. Score all events
scores = model.score_samples(all_graphs)

# 4. Evaluate
results = evaluate(scores, labels)
print_results(results)
```

---

## Running tests

```bash
pytest tests/ --cov=graph_glad_hep --cov-report=term-missing
```

---

## Package structure

```
graph_glad_hep/
  graph_builder.py   — Event → PyG Data graph construction + MockEventGenerator
  encoder.py         — DualEncoder (clean + perturbed GCN branches)
  losses.py          — L1 (reconstruction) + L2 (InfoNCE) + L3 (repr. error)
  decorrelation.py   — DisCo distance correlation regulariser
  evaluate.py        — AUC, SIC curve, IS metric
  model.py           — GraphGLADC sklearn-compatible estimator
tests/
  test_graph_builder.py
  test_encoder.py
  test_losses_and_decorrelation.py
```

---

## Related

- [ML4SCI/Anomaly-Detection](https://github.com/ML4SCI/Anomaly-Detection) — prior GENIE work (Saif 2022)
- [GSoC 2026 proposal](https://ml4sci.org/gsoc/2026/proposal_GENIE1.html)
- [GLADC paper](https://doi.org/10.1038/s41598-022-22086-3)
