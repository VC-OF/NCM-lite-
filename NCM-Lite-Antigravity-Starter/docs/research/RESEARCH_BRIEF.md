# Research Brief — Stanford Next-Concept Modeling (NCM)

## Objective

Produce a source-grounded technical specification for a small CPU-first NCM-inspired
prototype. The goal is to understand and implement the central algorithmic ideas, not
to claim reproduction of the paper's large-scale results.

## Primary sources to inspect

1. Paper: https://arxiv.org/abs/2610.14162
2. Official Hazy Research project blog:
   https://hazyresearch.stanford.edu/blog/2026-10-12-ncm
3. Official repository: https://github.com/HazyResearch/ncm
4. Checkpoints: https://huggingface.co/hazyresearch/ncm-checkpoints

Verify that these sources are accessible and that the paper/repository correspond to
the intended Next-Concept Modeling work. Record access date and exact commit/revision
where relevant. If any source is unavailable or has changed, report that instead of
guessing.

## Questions to answer from the paper and code

### 1. Problem and motivation
- What problem does NCM address?
- What is the paper's definition/intuition for a “concept”?
- What limitations of the compared approaches motivate this design?
- Which claims are theoretical, empirical, or implementation-specific?

### 2. Stage 1: multiscale tokenizer
- Draw the encoder → multiscale representation → residual quantization → decoder path.
- Explain how scales/resolutions are formed.
- Explain residual quantization, codebook lookup, and reconstruction.
- Record codebook counts/sizes/dimensions and architectural details only when directly
  supported by the paper or code.
- Explain all losses and regularization, including nested dropout or other multiscale
  mechanisms if present.
- Explain tokenizer training data, optimizer, schedule, and preprocessing where given.
- Identify what the official repository actually implements and what remains absent.

### 3. Stage 2: next-concept predictor
- Explain what discrete codes are targets.
- Explain the hierarchy and level ordering.
- Explain which positions are predicted in parallel and which levels are sequentially
  conditioned.
- Record transformer architecture, attention masks, embeddings, and loss.
- Explain any training-time corruption/noising of conditioning codes and why it is used.
- Explain how the tokenizer is frozen and how code targets are generated/cached.

### 4. Inference
- Give a numbered inference algorithm from prompt/input through hierarchical code
  generation to decoder output.
- Identify sampling/temperature/top-k/top-p behavior only if supported.
- State which computation is parallel and which is sequential.
- Explain stopping, length control, and decoding details.

### 5. Experimental evidence
- List datasets, baselines, metrics, compute hardware, and training budgets.
- Reproduce the reported result table accurately with units and context.
- Explicitly state that paper benchmark speedups are not predictions for this laptop.
- Distinguish language modeling, summarization, and other task results.

### 6. Official implementation audit
Inspect repository structure, setup instructions, scripts, configs, model definitions,
data requirements, checkpoint loading, and documented commands.

Classify each capability:
- available and documented;
- present but undocumented or incomplete;
- not found;
- unclear.

Do not assume that a repository containing evaluation code also provides a complete
end-to-end training pipeline.

## Prototype constraints

Initial target:
- CPU-first;
- approximately 10–15M generator parameters as an upper target, subject to memory tests;
- total process memory budget should be conservative on a machine with about 13.3 GB
  usable RAM;
- no assumption of CUDA;
- begin with synthetic tensors and tiny data;
- use a small text dataset only after the pipeline works.

The parameter target is not a memory guarantee. Include estimates for weights,
gradients, optimizer states, activations, temporary tensors, dataloader memory, Python,
and IDE overhead. Recommend a smaller fallback if needed.

## Required output files

### `docs/research/paper-notes.md`
Source-grounded notes with section/page or code-path references. Use short quotations
only when necessary; otherwise paraphrase.

### `docs/research/spec.md`
Implementable architecture specification containing:
- scope and non-goals;
- Stage 1 and Stage 2 diagrams/data flow;
- exact shapes, dtypes, and dimensions;
- loss definitions;
- training sequence;
- inference algorithm;
- checkpoint formats;
- configuration;
- CPU/memory budget;
- acceptance criteria.

### `docs/INTERFACES.md`
For every module, define:
- Python class/function signature;
- input/output tensor shapes;
- dtypes and device behavior;
- train/eval/frozen behavior;
- exceptions and validation;
- serialization format;
- owner agent.

### `docs/research/test-plan.md`
List unit, integration, numerical, overfit, checkpoint, and end-to-end tests.

### `docs/research/deviations.md`
For each simplification, provide:
- paper behavior;
- prototype behavior;
- reason;
- expected impact;
- how it will be evaluated.

## Required source discipline

Every substantive claim must be labeled:
- `[PAPER]` — explicitly stated or shown in the paper;
- `[OFFICIAL CODE]` — verified in the official repository;
- `[PROTOTYPE DECISION]` — a choice for NCM-Lite;
- `[UNRESOLVED]` — not established by available evidence.

Cite paper section/page or repository path and commit where possible. Never invent
hyperparameters, equations, dimensions, or benchmark values. If the paper and code
differ, document the discrepancy.

## Final recommendation

Conclude with:
1. What can be implemented faithfully at small scale?
2. What must be simplified for CPU constraints?
3. What research details remain uncertain?
4. What is the minimum viable experiment that can validate the core idea?
5. A go/no-go recommendation for `spec-v1`.

Do not write implementation code in this phase.
