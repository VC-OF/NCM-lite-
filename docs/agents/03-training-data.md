# Agent 03 — Data Pipeline & Training

## Mission

Build reproducible data preparation and training orchestration for the two distinct
training stages. Stage 1 trains the tokenizer. Stage 2 freezes it and trains the
next-concept predictor on discrete code targets.

## Read first

- `AGENT_RULES.md`
- `docs/research/spec.md`
- `docs/INTERFACES.md`
- `configs/laptop_cpu.yaml`

Do not start until `spec-v1` is approved.

## Ownership

Only edit:
- `src/ncm/training/`
- `src/ncm/data/`
- `scripts/`
- `tests/training/`
- `tests/data/`

Do not edit model internals, tokenizer internals, inference, shared configs, or
interface docs. Propose config changes rather than silently changing them.

## Required implementation

- Explicit, opt-in dataset preparation; no implicit network download.
- Streaming or memory-conscious text loading.
- Deterministic train/validation split and seeded shuffling.
- Stage 1 tokenizer training loop.
- Stage 2 generator training loop with tokenizer frozen and in evaluation mode.
- Loss aggregation and logging.
- Atomic checkpoints with model, optimizer, scheduler, step, RNG state, config, and
  version metadata as appropriate.
- Resume-from-checkpoint support.
- Tiny smoke-test commands that finish quickly on CPU.

## Development sequence

1. Implement synthetic dataset and dataloader.
2. Implement generic training-step tests with mock models.
3. Add tokenizer stage loop.
4. Add frozen-tokenizer code extraction / caching strategy.
5. Add next-concept stage loop.
6. Add checkpoint/resume tests.
7. Add optional TinyStories preparation only after license, source, and download behavior
   are documented and approved.

## Tests required

- deterministic split and seeded batches;
- no full-dataset materialization by default;
- tokenizer parameters update in Stage 1;
- tokenizer parameters do not update in Stage 2;
- generator parameters update in Stage 2;
- loss values are finite;
- checkpoint save/load/resume restores step and optimizer state;
- tiny smoke test runs with CPU and a very small sample.

## Handoff

Report exact commands, test results, memory use if measured, checkpoint format, data
provenance, and any deviation from the approved spec.
