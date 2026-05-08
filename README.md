# Adaptive-layer-skipping

Implementation and experimental code for the thesis:

> **Adaptive layer skipping during large language model inference**

The project investigates adaptive skipping of transformer FFN layers during inference in large language models. The goal is to reduce computation by dynamically routing tokens through the network while preserving model quality.

The repository explores:
- layer importance analysis,
- static layer pruning,
- oracle-based token-level skipping,
- lightweight predictor-based routing,
- gated routing methods,
- and comparisons with the BlockPruner approach.

The experiments were conducted on the SmolLM model using PyTorch and Hugging Face Transformers.

---

## Repository Structure

### `layer_importance.ipynb`
Layer importance analysis, layer removal strategies, and oracle-based token-level skipping.

### `predictor_definitions.py`
Definitions of lightweight predictor models used for routing.

### `predictor.ipynb`
Training and evaluation of predictors estimating transformer block output magnitude.

### `predictor_v2.ipynb`
Iterative predictor training using routing-aware data collection.

### `predictor_v3.ipynb`
Predictors designed for SingleSkip routing.

### `router_evaluation.ipynb`
Integration and evaluation of predictor-based routing methods.

### `router_v2_evaluation.ipynb`
Evaluation using iteratively trained predictors.

### `router_v3_evaluation.ipynb`
Evaluation of SingleSkip routing with iterative predictors.

### `gates.ipynb`
Initial gated routing implementation using cross-entropy training.

### `gates_kl.ipynb`
Gated routing models trained using a KL-divergence objective.

### `gates_evaluation.ipynb`
Training and evaluation of gated models and routing decisions.

### `routing_results.ipynb`
Visualization and comparison of routing methods and baselines.

### `block_pruner.ipynb`
Implementation and evaluation of the BlockPruner baseline.

### `results/`
Tables and outputs from experiments.
