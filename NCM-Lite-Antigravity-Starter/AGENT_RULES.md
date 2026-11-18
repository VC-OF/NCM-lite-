# NCM-Lite Shared Agent Rules

These rules apply to every agent, every task, and every worktree.

## A. Research fidelity

1. Read this file, the assigned role file, the approved `docs/research/spec.md`,
   and `docs/INTERFACES.md` before implementation. During Phase 1, the Research
   Architect works from `docs/research/RESEARCH_BRIEF.md` instead.
2. Preserve the research distinction between the tokenizer stage and the next-concept
   modeling stage. The tokenizer is trained first; the generator is trained after the
   tokenizer is frozen.
3. Do not silently replace multiscale residual vector quantization, hierarchical
   discrete concept codes, or coarse-to-fine next-concept prediction with BPE-only
   tokenization, ordinary next-token prediction, continuous latent diffusion, or
   another architecture.
4. Label every deviation from the paper as one of:
   - `paper-faithful`: directly supported by the paper or official implementation;
   - `prototype adaptation`: an explicit simplification for laptop constraints;
   - `experimental`: a proposed change that must be tested.
5. Never claim exact reproduction of Stanford NCM. Use “NCM-inspired prototype”
   unless the research review demonstrates faithful implementation of the relevant
   method and clearly states what is and is not reproduced.

## B. Scope and ownership

6. Edit only the folders and files listed in your role document. If a cross-module
   change is necessary, stop and request approval; do not edit another agent's area.
7. Do not overwrite another agent's work, change public interfaces, rename shared
   symbols, or change configuration keys without an interface-change proposal.
8. Keep changes within your own Git worktree/branch. Never commit directly to `main`.
9. Do not merge branches. The project lead performs reviews and merges one branch at
   a time.
10. Do not modify `AGENT_RULES.md`, role prompts, approved specs, or interface contracts
    unless explicitly assigned.

## C. Engineering quality

11. Every public module/function must have type hints and concise docstrings. Keep
    tensor shapes and dtypes explicit in comments or docstrings where useful.
12. Add unit tests for new behavior. Tests must include shape, dtype, device, edge-case,
    and deterministic/small synthetic-data checks where applicable.
13. Use small synthetic/mock tensors for development. Do not make implementation agents
    wait for the complete dataset or full tokenizer training.
14. Do not download datasets, install packages, run long jobs, or execute destructive
    shell commands without the user's approval in the IDE.
15. Avoid hidden network calls and telemetry in project code. Dataset download scripts
    must be explicit, documented, and opt-in.
16. Use configuration files for model sizes, paths, random seeds, optimizer settings,
    and training limits. Avoid hard-coded machine-specific paths.
17. Save checkpoints atomically and include enough metadata to resume or reproduce a run.
18. Log training stage, step, loss components, learning rate, elapsed time, peak memory
    when available, configuration, and random seed.
19. Use clear exceptions and validation. Do not silently swallow errors or return
    fabricated outputs to make a test pass.
20. Keep dependencies minimal and pinned or bounded in `pyproject.toml`. Do not add
    heavyweight packages without explaining why.

## D. Compute and data safety

21. The initial target is CPU-first with a conservative process memory budget. Do not
    assume CUDA or a discrete GPU.
22. Start with tiny smoke tests. No agent may launch a full training run by default.
23. Do not load the entire dataset into RAM. Use streaming or memory-mapped access when
    appropriate.
24. Do not commit datasets, model checkpoints, caches, credentials, API keys, or personal
    data. Ensure `.gitignore` covers them.
25. Treat 10–15M parameters as a provisional model-size target, not a guarantee that
    training will fit. Account for gradients, optimizer states, activations, dataloader
    workers, and IDE overhead.
26. Any proposed configuration that exceeds the agreed memory budget must be marked
    experimental and must not become the default without the project lead's approval.

## E. Validation and reporting

27. Run the smallest relevant tests first, then the full test suite if practical.
28. Report exact commands run, test results, files changed, known limitations, and any
    deviation from the spec.
29. Never claim a test passed unless it was actually run and its output inspected.
30. Do not fabricate benchmark results, paper claims, citations, or hardware measurements.
31. Before handing off, provide:
    - implementation summary;
    - files changed;
    - public APIs added/changed;
    - tests and exact results;
    - known issues;
    - memory/runtime observations;
    - decisions requiring project-lead approval.

## F. Collaboration protocol

32. If blocked by an unclear interface, ask a focused question and continue only on
    independent work.
33. If two agents need the same file, the project lead assigns a single owner.
34. Submit interface changes as a proposal containing rationale, old/new signatures,
    affected modules, migration plan, and tests.
35. Prefer small, reviewable commits. Do not squash or rebase another agent's branch.
