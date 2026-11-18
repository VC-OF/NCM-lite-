# NCM-Lite Shared Agent Rules

These rules apply to every agent, every task, and every worktree in the NCM-Lite project.

## Core Governance Rules (The 13 Rules)

1. **Two-Stage Research Fidelity**: Strictly preserve the paper's core two-stage architecture: Stage 1 multiscale residual vector-quantized (RVQ) tokenizer trained first; then frozen in Stage 2 while a coarse-to-fine next-concept predictor is trained.
2. **Strict Scope & Folder Ownership**: Edit only the folders and files explicitly assigned to your role. Cross-module changes require stopping and requesting approval from the project lead.
3. **No Silent Replacement of Research Components**: Do not substitute simplified heuristics, standard BPE tokenizers, ordinary left-to-right autoregressive next-token prediction, or continuous latent diffusion for the paper's core algorithmic components.
4. **Explicit Deviation Labeling**: Every difference from the paper must be categorized as `paper-faithful`, `prototype adaptation` (for laptop constraints), or `experimental`.
5. **Accurate Prototype Attribution**: Always refer to this project as an "NCM-inspired prototype"; never claim exact reproduction of Stanford's large-scale compute runs.
6. **Git Worktree Isolation**: Work exclusively within your assigned feature branch / worktree. Never commit directly to `main`. Merging is performed solely by the project lead, one branch at a time.
7. **Contract & Interface Immutability**: Public class/function signatures, tensor shapes, dtypes, and configuration keys in `docs/INTERFACES.md` and `configs/laptop_cpu.yaml` cannot be altered without an approved Interface Change Proposal.
8. **CPU-First & Memory Ceiling**: Initial development is CPU-first. The laptop has ~13.3 GB usable RAM; process RAM ceiling is 8.0 GB, with an upper model size of 10–15M parameters. Account for activations, gradients, optimizer states, and IDE overhead.
9. **Mock & Synthetic Tensors First**: Develop against synthetic/mock tensors and deterministic unit tests. Never block an implementation agent on another agent's unmerged code or full dataset training.
10. **Explicit & Opt-In Data**: No automatic large dataset downloads or hidden network calls. Use streaming or memory-conscious chunked loading.
11. **Type Hints & Shape Documentation**: All public functions and classes must include type annotations and explicit tensor shape/dtype documentation in docstrings.
12. **Rigorous Verification via PyTest**: Run unit and shape tests before handoff. Never claim a test passed without running it and inspecting actual stdout/stderr.
13. **Structured Handoff & Transparency**: Conclude every task with a structured handoff report detailing files modified, public APIs added, exact test commands and outputs, memory/latency observations, and decisions requiring approval.

---

## Detailed Rules & Protocols

### A. Research Fidelity
- Read `AGENTS.md`, your assigned role file in `.agents/agents/` or `docs/agents/`, the approved `docs/research/spec.md`, and `docs/INTERFACES.md` before implementation.
- The tokenizer is trained first; the generator is trained only after the tokenizer is frozen in evaluation mode.
- Do not invent hyperparameters, equations, or benchmark results. Distinguish paper-faithful elements from prototype simplifications.

### B. Scope and Ownership
- Edit only the folders assigned to your role.
- Never edit another agent's files, shared configs, approved specs, or interface documents without an explicit proposal.
- Keep all work in your designated Git branch / worktree.

### C. Engineering Quality
- Every public function/method must specify tensor shapes, dtypes, and expected devices.
- Add unit tests for shape, dtype, device, boundary conditions, and deterministic small synthetic-data checks.
- Handle exceptions cleanly; do not swallow errors or fabricate mock outputs to artificially pass tests.
- Save checkpoints atomically with metadata (step, loss, config, random state).
- Keep dependencies minimal and pinned in `pyproject.toml`.

### D. Compute and Data Safety
- Do not assume CUDA or GPU availability.
- Start with tiny smoke tests before any extended run.
- Do not load full datasets into memory.
- Never commit datasets, checkpoints, caches, or API keys to Git.

### E. Validation and Reporting
- Run the smallest relevant unit tests first, then integration tests.
- Report exact commands, test results, files changed, and known limitations.
- Provide a clear handoff block at the end of each task.

### F. Collaboration Protocol
- If blocked by an interface ambiguity, ask a focused question and work on independent items.
- Submit interface modifications using the Interface Change Proposal template in `docs/INTERFACES.md`.
- Keep commits focused and reviewable.
