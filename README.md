# AI-SWE-Saftey-Lab

Research sandbox for **activation-based safety steering** in open-weight chat models.

The core idea: capture internal hidden-state “modes” associated with **refusal** vs **compliance** (elicited by the same harmful requests but with different forced assistant prefixes), compute a **direction vector** in representation space, then **inject** that vector during generation to see how behavior shifts.

This repo is intentionally small and script-driven (no training), focusing on:
- extracting “refusal directions” from intermediate transformer layers
- steering generation by adding those directions back into activations
- visualizing separation/trajectory in a low-dimensional space (PCA)

## What’s in this repo

### `test_general_saftey.py`
- Loads **TinyLlama/TinyLlama-1.1B-Chat-v1.0**
- Builds paired prompts:
  - same harmful request
  - assistant forced to begin with `"I cannot"` (refusal) vs `"Sure"` (compliance)
- Captures last-token hidden state at a chosen layer, computes:
  - `steering_vector = mean(refusal_states) - mean(compliance_states)` (normalized)
- Registers a forward hook to inject `coeff * steering_vector` during generation
- Tests the effect on a harmful query across different coefficients

### `test_coding_saftey.py`
- Loads **Qwen/Qwen2.5-Coder-3B-Instruct**
- Uses a larger set of harmful prompts to define refusal/compliance states more robustly
- Extracts vectors across **multiple layers** (default: layers 10–25)
- Applies steering across those layers during generation
- Tests on a *harmless* coding prompt to see whether positive steering causes inappropriate refusal

### `plotting_the_figures.py`
- Intended to visualize clusters/trajectories with **PCA**
- Plots refusal vs compliance activations and where steered test prompts land in that space
- Note: as written, this script assumes variables like `activations`, `model`, `tokenizer`, `steering_vector`, and `layer_id` already exist in scope (i.e., it’s currently more like a “notebook cell” than a standalone script).

## Setup

### Environment
Python 3.10+ recommended.

Install deps:
```bash
pip install torch transformers accelerate numpy matplotlib scikit-learn
```

### Hardware
These scripts are easiest to run on a CUDA GPU.
- `device_map="auto"` is used in places, but some scripts also hardcode `"cuda"` in `.to("cuda")`.
- If you’re on CPU/MPS, you’ll need to adjust those `.to(...)` calls accordingly.

## How to run

### 1) General safety steering (TinyLlama)
```bash
python test_general_saftey.py
```
You should see baseline vs steered outputs for the same query at different coefficients.

### 2) Coding model multi-layer steering (Qwen Coder)
```bash
python test_coding_saftey.py
```
You should see:
- a baseline response to a harmless coding prompt
- then responses under stronger positive steering (which may become overly refusant)

### 3) Plot PCA figure (experimental)
`plotting_the_figures.py` currently is not fully standalone. Typical use is:
- run extraction/steering code to populate `activations` and the steering vector
- then run the plotting script in the same interpreter session (or refactor to load saved activations)

## Interpreting coefficients

- `coeff = 0`: no steering (baseline)
- `coeff > 0`: pushes the model toward the “refusal” mode captured by the vector
- `coeff < 0`: pushes toward the opposite direction (often more eager/less safe, sometimes unstable)

There is no universal “correct” coefficient—effective values depend on:
- model architecture/scale
- which layer(s) you steer
- how the vector was constructed
- decoding settings

## Notes on methodology (high level)

This is a simple linear intervention:
1. Collect hidden states for condition A (refusal) and condition B (compliance)
2. Compute a mean-difference direction
3. Add that direction to hidden states at inference time via hooks

This is useful for:
- probing whether refusal behavior corresponds to a consistent internal direction
- stress-testing how easily behavior can be shifted without fine-tuning

## Safety / ethics

These scripts include **harmful prompts** strictly as *measurement probes* to elicit refusal/compliance internal states.
- Do not use this repo to generate or operationalize harmful instructions.
- If you extend the work, prefer evaluating on established safety benchmarks and logging only aggregate metrics.

## Known rough edges / TODO

- Make `plotting_the_figures.py` standalone by saving/loading activations (e.g., `.pt` or `.npz`).
- Standardize device handling (`model.device` vs hardcoded `"cuda"`).
- Add deterministic seeding and configurable CLI args (model id, layers, coeffs, prompts).
- Add basic metrics (refusal rate, toxicity classifier, etc.) instead of only qualitative prints.
