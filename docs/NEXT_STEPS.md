# Next Steps

This roadmap keeps the project aligned with a public SWE-bench-style evaluation
and training loop. It is ordered by engineering value and resume evidence, not by
feature size.

## P0: Real Public SWE Evaluation Smoke

- Run `import-swe-dataset` against `SWE-bench/SWE-bench_Lite` or a small exported
  JSONL sample with at least 20 held-out issues.
- Configure a Linux/Docker machine with the official SWE-bench harness.
- Generate predictions with `swe-predict` for `base`, `student`, and `teacher`
  roles, then run `run-swe-harness` with an explicit harness command.
- Save the report under `reports/` and summarize resolved rate, invalid patch
  rate, patch apply failures, timeout count, and average latency.

## P1: Model-Backed Agent Runner

- Replace the current single-turn patch prompt with a small repo-context packer:
  issue text, file tree, selected snippets, and prior failure messages.
- Add patch budget controls: max prompt tokens, max output tokens, allowed paths,
  and retry count for invalid unified diffs.
- Record one artifact per model attempt: prompt hash, model ID, raw output, parsed
  patch, patch validation result, and final judge status.
- Keep multi-step shell agents out of scope until the single-turn patch path has
  stable metrics.

## P2: Training Data Expansion

- Add adapters for public successful trajectories such as SWE-smith trajectories
  and SWE-Gym/OpenHands SFT trajectories.
- Export two LLaMA-Factory datasets: `issue_to_patch` and `trajectory_action`.
- Filter examples by license, source, context length, duplicate patch hash, and
  held-out split isolation.
- Track dataset version IDs in every model comparison report.

## P3: Student Training and Serving

- Train a QLoRA student on the Windows CUDA machine using the generated
  LLaMA-Factory config.
- Serve the adapter through vLLM or LLaMA-Factory's OpenAI-style API.
- Compare base vs student vs teacher on the same held-out public SWE tasks.
- Report only measured metrics and keep GPT-family models as teacher APIs, not
  trainable weights.

## P4: Platform Hardening

- Add a Postgres/object-storage path for larger run records and artifacts.
- Add OpenTelemetry spans for data import, model inference, patch parsing, patch
  apply, harness runtime, and scoring.
- Add CI jobs for toy hidden-pytest tasks; keep full SWE-bench harness runs as
  explicit/manual jobs due to Docker and runtime cost.
- Add a reproducibility bundle with commit hash, dataset version, model config,
  command lines, environment, and hardware.

## Resume Evidence Checklist

- Public reference map: `docs/PUBLIC_SWE_REFERENCE_MAP.md`.
- At least one public SWE import artifact.
- At least one model comparison report.
- At least one failed-case analysis with failure categories.
- Clear boundary statement: no internal code, data, protocol, or private metrics.
