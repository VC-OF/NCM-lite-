# Antigravity + Git Worktree Workflow

Antigravity UI labels and agent capabilities can change. This workflow deliberately
uses ordinary Git commands and repository Markdown so the project remains portable.

## Phase 0 — Prepare

1. Install Git for Windows and Python 3.11 or 3.12.
2. Install Antigravity from its official distribution and verify current documentation
   for workspace rules, Manager mode, agent concurrency, terminal approvals, and quotas.
3. Create the repository and virtual environment.
4. Copy this starter pack into the repository.
5. Add `.venv/`, datasets, checkpoints, logs, caches, and local secrets to `.gitignore`.
6. Configure Antigravity workspace rules to use `AGENT_RULES.md`, if supported. If the
   product uses another mechanism, follow its current docs and retain this file as
   canonical project policy.

## Phase 1 — Research gate

- Start only `00-research-architect`.
- Ask it to read `AGENT_RULES.md` and `docs/agents/00-research-architect.md`.
- Require a plan before file edits.
- Review its plan and artifacts.
- Ask for corrections until interfaces and deviations are explicit.
- Approve and commit the research specification.
- Tag the commit: `git tag spec-v1`.

## Phase 2 — Parallel implementation

Use four independent worktrees for Agents 1–4. Begin with only 2–3 active agents
because the laptop has approximately 13.3 GB usable RAM and the IDE itself consumes
memory. The agent count is a practical starting limit, not an Antigravity product limit.

Example commands, run from the main repository:

```bash
git worktree add ../NCM-Lite-tokenizer -b feature/tokenizer
git worktree add ../NCM-Lite-model -b feature/next-concept
git worktree add ../NCM-Lite-training -b feature/training
git worktree add ../NCM-Lite-eval -b feature/evaluation
```

Open each worktree as a separate Antigravity workspace if the installed version supports
that workflow. Assign one agent per worktree. If the Manager cannot attach separate
workspaces, use separate IDE windows or sequential tasks; do not let agents edit the
same checkout concurrently.

Every task prompt must begin with:
- Read `AGENT_RULES.md`.
- Read your role file.
- Read approved `docs/research/spec.md` and `docs/INTERFACES.md`.
- Only edit your explicitly owned paths.
- Use mock tensors and small tests.
- Do not merge or edit other agents' folders.

## Phase 3 — Review and merge

The project lead, not the agents, merges branches.

For each branch:
1. Inspect `git status` and `git diff --stat`.
2. Inspect the full diff for out-of-scope edits.
3. Run targeted tests.
4. Merge one branch at a time.
5. Run the full test suite after each merge.
6. If a merge changes a shared interface, stop and reconcile the contract before proceeding.

## Phase 4 — Training gate

- Run CPU-only synthetic smoke tests first.
- Measure peak process memory and runtime.
- Start with fallback model configuration if memory is high.
- Train tokenizer first; validate reconstruction and code utilization.
- Freeze tokenizer; then train the next-concept model.
- Save checkpoints and verify resume.
- Only then opt in to a small real text dataset.

## Phase 5 — Independent review

Ask a reviewer model from a different model family, if available, to inspect:
- paper fidelity;
- code interfaces and tensor shapes;
- data leakage and target leakage;
- training-stage separation;
- tests and reproducibility;
- claims versus measured results.

The reviewer must inspect code and tests, not merely summarize agent reports.

## Git safety

Useful commands:

```bash
git status --short
git diff --stat
git diff
git branch --show-current
git worktree list
pytest -q
```

Do not delete a worktree until its branch is merged or intentionally abandoned and
its uncommitted changes have been checked.
