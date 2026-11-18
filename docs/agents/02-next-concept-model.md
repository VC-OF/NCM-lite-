# Agent 02 — Hierarchical Next-Concept Model

## Mission

Implement Stage 2's predictor: a compact transformer that predicts the tokenizer's
hierarchical discrete concept codes according to the approved coarse-to-fine dependency
structure.

## Read first

- `AGENT_RULES.md`
- `docs/research/spec.md`
- `docs/INTERFACES.md`
- `docs/research/deviations.md`

Do not start until `spec-v1` is approved.

## Ownership

Only edit:
- `src/ncm/models/next_concept.py`
- `src/ncm/models/embeddings.py` (if assigned in the interface contract)
- `tests/models/`

Do not edit tokenizer, quantization, training loops, inference orchestration, configs,
or shared interface docs.

## Required implementation

- Model inputs and outputs exactly match the approved interface contract.
- Embeddings for hierarchical code indices, with explicit handling of level identity.
- Coarse-to-fine prediction and conditioning exactly as specified.
- Parallel prediction over positions within a level if that is the approved method.
- Logits for each required codebook/level, with documented tensor shapes.
- Loss-compatible outputs for the training agent.
- Configurable depth, width, heads, dropout, and vocabulary/codebook sizes.

Do not turn the task into ordinary autoregressive next-token language modeling. The
model predicts discrete concept codes, not ordinary text tokens, unless the approved
spec explicitly defines a text-token interface for a particular submodule.

## Development sequence

1. Build shape-only forward tests using mock code indices.
2. Verify level ordering and conditioning masks.
3. Verify output logits align with the correct codebook sizes.
4. Verify loss can be computed without shape broadcasting mistakes.
5. Verify parameter count and CPU forward memory on the smallest config.
6. Add a tiny synthetic overfit test if feasible.

## Tests required

- input validation for batch, sequence, level, and code-index dimensions;
- output shape per level;
- no target leakage across prohibited positions/levels;
- deterministic evaluation mode;
- gradient flow and finite loss;
- parameter-count test against config;
- save/load round-trip if assigned.

## Handoff

Report architecture, parameter count, tensor shapes, tests, memory observations, and
any deviation from the approved spec.
