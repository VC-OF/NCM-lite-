# NCM-Lite Interface Contracts

**Status: template — Research Architect must complete and project lead must approve.**

Do not implement against guessed signatures. This file becomes the shared contract.

## Common conventions

- Framework: PyTorch, unless the approved spec justifies otherwise.
- Tensor device: preserve caller device; CPU is the initial supported target.
- Tensor dtype: explicitly document integer code-index dtype and floating activation dtype.
- Batch dimension: first dimension unless the approved spec states otherwise.
- All shapes below must be replaced with exact symbolic dimensions in `spec-v1`.

## Stage 1 — Tokenizer

| Component | Proposed responsibility | Owner |
|---|---|---|
| MultiscaleEncoder | Convert input representation into multiscale latent tensors | Agent 1 |
| ResidualVectorQuantizer | Quantize residuals and return discrete indices plus quantized tensors | Agent 1 |
| MultiscaleDecoder | Reconstruct input from quantized multiscale representations | Agent 1 |
| TokenizerModel | Compose encode, quantize, decode and loss computation | Agent 1 |

Required contract fields:
- input representation shape and preprocessing;
- latent shape at every level;
- code index shape and range for every level;
- quantized latent shape;
- reconstruction shape;
- loss dictionary keys and reduction conventions;
- whether codebook/encoder gradients use straight-through estimation or another method;
- checkpoint schema and version.

## Stage 2 — Next-concept predictor

| Component | Proposed responsibility | Owner |
|---|---|---|
| CodeEmbedding | Embed discrete codes and level identities | Agent 2 |
| NextConceptModel | Predict code distributions at each hierarchy level | Agent 2 |
| PredictionOutput | Return logits and any auxiliary outputs | Agent 2 |

Required contract fields:
- input code shape and level ordering;
- conditioning mask and causal constraints;
- logits shape and codebook vocabulary per level;
- target shape and alignment;
- per-level loss names and reduction;
- parallel dimensions and sequential dependencies;
- frozen-tokenizer assumption.

## Training

| Component | Proposed responsibility | Owner |
|---|---|---|
| TextDataset | Stream/preprocess text examples | Agent 3 |
| TokenizerTrainer | Optimize Stage 1 tokenizer | Agent 3 |
| CodeDataset/Cache | Produce or cache frozen tokenizer codes | Agent 3 |
| GeneratorTrainer | Optimize Stage 2 predictor | Agent 3 |
| CheckpointManager | Save/load resumable state | Agent 3 |

## Inference and evaluation

| Component | Proposed responsibility | Owner |
|---|---|---|
| generate_codes | Generate hierarchy in approved order | Agent 4 |
| decode_codes | Convert generated codes to output representation | Agent 4 |
| evaluate_reconstruction | Measure tokenizer reconstruction | Agent 4 |
| evaluate_generation | Run generation diagnostics and save samples | Agent 4 |

## Interface change proposal template

Before changing a shared interface, submit:

1. Reason for change.
2. Current signature and shapes.
3. Proposed signature and shapes.
4. All affected modules and owners.
5. Migration plan.
6. Tests to add/update.
7. Whether the change affects research fidelity.
