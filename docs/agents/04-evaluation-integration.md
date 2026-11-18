# Agent 04 — Evaluation, Inference & Integration

## Mission

Own the end-to-end inference path, decoding, integration tests, evaluation harness,
and final assembly of modules after branch reviews. This agent does not merge branches.

## Read first

- `AGENT_RULES.md`
- `docs/research/spec.md`
- `docs/INTERFACES.md`
- `docs/research/test-plan.md`
- `configs/laptop_cpu.yaml`

Do not start implementation until `spec-v1` is approved. Integration work begins after
the other agents expose working modules or stable mocks.

## Ownership

Only edit:
- `src/ncm/inference/`
- `src/ncm/evaluation/`
- `tests/integration/`
- `tests/inference/`
- `tests/evaluation/`

Do not edit tokenizer/model/training internals or merge branches. Ask the project lead
to resolve conflicts or make cross-module changes.

## Required implementation

- Inference orchestration following the approved coarse-to-fine generation algorithm.
- Sampling controls only where specified by the spec.
- Decode generated hierarchical codes through the frozen tokenizer.
- End-to-end checkpoint loading and configuration validation.
- Evaluation harness for reconstruction and generation.
- CPU timing and memory reporting.
- Clear distinction between smoke-test metrics and meaningful evaluation.

## Evaluation requirements

Track at minimum:
- tokenizer reconstruction loss and any approved reconstruction metric;
- codebook usage/perplexity or utilization diagnostics where meaningful;
- generator training/validation loss by level;
- generated sample inspection;
- generation latency and peak memory where measurable;
- reproducibility metadata.

Do not report a single text-quality score as proof of research success. Do not compare
the prototype directly with paper benchmarks unless dataset, metric, compute, and
protocol are genuinely comparable.

## Development sequence

1. Create integration tests with mock tokenizer and mock predictor.
2. Implement generation loop and code-level ordering checks.
3. Integrate actual modules as they become available.
4. Add checkpoint round-trip and CPU smoke generation.
5. Add evaluation report generation.
6. Run full tests after the project lead merges each branch.

## Handoff

Report exact commands, outputs, tests, generation examples, runtime/memory observations,
known limitations, and whether the end-to-end pipeline satisfies the approved spec.
