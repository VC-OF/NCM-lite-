# Agent 01 — Multiscale Tokenizer & Residual VQ

## Mission

Implement Stage 1: the prototype's multiscale encoder, residual vector quantization
(RVQ) codebooks, quantization/dequantization path, and decoder/reconstruction path.

## Read first

- `AGENT_RULES.md`
- `docs/research/spec.md`
- `docs/INTERFACES.md`
- `docs/research/deviations.md`

Do not start until the project lead has approved `spec-v1`.

## Ownership

Only edit:
- `src/ncm/tokenizer/`
- `src/ncm/quantization/`
- `tests/tokenizer/`
- `tests/quantization/`

Do not edit training loops, next-concept model, inference orchestration, configs, or
shared interface docs.

## Required implementation

- Multiscale representation encoder as specified in the approved spec.
- Residual vector quantizer with explicit codebook lookup and code-index outputs.
- Quantize/dequantize operations with well-defined shapes and dtypes.
- Decoder/reconstruction path.
- Loss components required by the approved spec.
- Serialization/load support for tokenizer checkpoints if assigned by interface contract.

## Development sequence

1. Implement tensor-shape validation and deterministic synthetic tests.
2. Implement a minimal RVQ codebook and verify nearest-code lookup.
3. Implement residual stages and verify residual decreases or behaves as specified.
4. Implement multiscale encoder and decoder.
5. Verify encode → quantize → decode on tiny synthetic batches.
6. Add a tiny overfit test. Do not launch full training.

## Tests required

- codebook lookup indices have expected shape/range;
- quantized tensors have expected shape/dtype/device;
- gradients flow to intended encoder/decoder/codebook parameters;
- no unintended gradient path through frozen or detached values;
- empty/invalid inputs fail clearly;
- encode/decode and checkpoint round-trip work;
- deterministic behavior under fixed seed where expected.

## Handoff

Report exact APIs, shapes, loss definitions, tests run, numerical observations, and
any divergence from the approved research specification. Do not change public
interfaces without a proposal.
