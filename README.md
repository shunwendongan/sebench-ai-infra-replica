# SE-Bench AI Infra Replica

This repository is a public-paper reproduction prototype inspired by a resume item about
SE-Bench long-horizon AI agent evaluation, benchmark authoring agents, and 3D spatial
reasoning diagnosis. It is not ByteDance internal code and does not claim to reproduce
private systems. The implementation uses public papers, open-source benchmark patterns,
and synthetic fixtures so it can be inspected and run locally.

## What This Builds

- A benchmark authoring loop that turns a natural-language requirement into typed tasks.
- A dataset builder and SQLite artifact store for benchmark items and run records.
- A Work/Judge evaluation orchestrator with path whitelist checks and a local mock runner.
- A 3D spatial reasoning diagnostic module using centroid displacement and Kabsch SVD.
- FastAPI and Typer entrypoints for service and CLI usage.
- Technical documentation with architecture diagrams, related work, and reproduction notes.

See `docs/TECHNICAL_DESIGN.md` for the original prototype design and
`docs/SE_BENCH_REPRODUCTION_TECH_DESIGN.md` for the expanded v2.5 reproduction
design covering literature priority, mature GitHub project lessons, architecture,
data lineage, isolation, performance optimization, cache-hit strategy, pass-rate
improvement, Mac MLX/MPS local benchmark boundaries, true evaluation-loop acceptance,
P0/P1 toy true-loop measurements, round3 simulated-real-repo validation, round4 patch-submission validation, round5 parallel runner validation, round6 local git + hidden pytest worker sweep validation, round7 hidden pytest/filesystem I/O diagnostics, round8 independent-repeat stability sweep and production worker cap, round9 pytest plugin autoload optimization, round10 default acceleration validation, round11 pytest plugin dependency scan, round12 Torch MPS deployment, hard task timeout, repo-affinity scheduling, round13 shared cache/repo-shard scheduling, auto cache policy, and acceptance criteria. The companion analysis is in
`docs/SE_BENCH_MATURE_PROJECT_OPTIMIZATION_ANALYSIS.md`; benchmark highlights are in
`docs/BENCHMARK_RESULTS.md`; the public SWE reference boundary is in
`docs/PUBLIC_SWE_REFERENCE_MAP.md`; planned follow-up work is in
`docs/NEXT_STEPS.md`; editable Mermaid sources
are under `docs/diagrams/` and mirrored as `docs/*.mmd` aliases.

## Quickstart

```bash
git clone https://github.com/shunwendongan/sebench-ai-infra-replica.git
cd sebench-ai-infra-replica
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/build_git_pytest_benchmark.py --tasks 128 --out examples/git_pytest_benchmark.json
python -m pytest
```

Apple Silicon local benchmark setup:

```bash
python3 -m venv .venv-apple
source .venv-apple/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,apple]"
```

Install PyTorch MPS only when local Apple GPU comparison is needed:

```bash
python -m pip install -e ".[mps]"
```

Run a mock reproduction pipeline:

```bash
sebench reproduce \
  --requirements examples/requirements.json \
  --spatial-scene examples/synthetic_spatial_scene.json \
  --out reports/demo_run.json
```

Build teacher-student training data and LLaMA-Factory exports:

```bash
sebench import-swe-dataset \
  --hf-dataset SWE-bench/SWE-bench_Lite \
  --limit 20 \
  --out artifacts/public_swe/dataset.json \
  --training-out artifacts/public_swe/dataset_version.json

sebench generate-data \
  --requirements examples/requirements.json \
  --count 10 \
  --out artifacts/training/dataset_version.json

sebench export-llamafactory \
  --dataset-version artifacts/public_swe/dataset_version.json \
  --out-dir artifacts/llamafactory/sebench_student_sft
```

After training a student on a CUDA machine and serving it through an
OpenAI-compatible endpoint, compare model roles:

```bash
sebench swe-predict \
  --dataset artifacts/public_swe/dataset.json \
  --role student \
  --out artifacts/public_swe/predictions.jsonl

sebench evaluate-models \
  --dataset examples/git_pytest_benchmark.json \
  --roles base,student,teacher \
  --out-dir reports/model_evals
```

Start the API:

```bash
uvicorn sebench_infra.api:app --reload
```

Run the true-loop toy benchmark on Apple Silicon/MLX:

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/toy_benchmark.json \
  --out artifacts/reports/toy_baseline_report.json \
  --summary-out artifacts/reports/mac_mlx_baseline_summary.md
```

Run the current round2 benchmark with MLX and Torch MPS sanity:

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/toy_benchmark.json \
  --warmup 1 \
  --repeat 5 \
  --mlx-size 512 \
  --mlx-iters 5 \
  --out artifacts/reports/toy_round2_report.json \
  --summary-out artifacts/reports/mac_mlx_round2_summary.md
```

Run the standalone Apple Silicon CPU/MPS/MLX sanity:

```bash
python scripts/benchmark_apple_silicon.py \
  --size 2048 \
  --warmup 1 \
  --iters 5 \
  --out artifacts/reports/apple_silicon_round2_sanity_2048.json
```

Build and run the round3 simulated-real-repo benchmark:

```bash
python scripts/build_realistic_benchmark_manifest.py \
  --out examples/realistic_benchmark.json

python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/realistic_benchmark.json \
  --warmup 1 \
  --repeat 5 \
  --mlx-size 512 \
  --mlx-iters 5 \
  --out artifacts/reports/realistic_round3_fixed_warm_report.json \
  --summary-out artifacts/reports/mac_mlx_realistic_round3_fixed_warm_summary.md
```

Build and run the round4 patch-submission benchmark:

```bash
python scripts/build_patch_benchmark_manifest.py \
  --source examples/realistic_benchmark.json \
  --out examples/patch_benchmark.json

python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/patch_benchmark.json \
  --warmup 1 \
  --repeat 5 \
  --workers 4 \
  --task-timeout-sec 10 \
  --mlx-size 512 \
  --mlx-iters 5 \
  --out artifacts/reports/patch_round4_warm_report.json \
  --summary-out artifacts/reports/mac_mlx_patch_round4_warm_summary.md
```

Build and run the round6 local git + hidden pytest worker sweep:

```bash
python scripts/build_git_pytest_benchmark.py \
  --tasks 128 \
  --out examples/git_pytest_benchmark.json

python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark.json \
  --workers 64 128 256 1024 \
  --warmup 0 \
  --repeat 1 \
  --task-timeout-sec 30 \
  --mlx-size 256 \
  --mlx-iters 1 \
  --prefix git_pytest_round6 \
  --out artifacts/reports/git_pytest_round6_worker_sweep.json \
  --summary-out artifacts/reports/git_pytest_round6_worker_sweep.md
```

Run the round7 hidden pytest + filesystem I/O diagnostics:

```bash
python scripts/build_git_pytest_benchmark.py \
  --tasks 1024 \
  --out examples/git_pytest_benchmark_1024.json

python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --workers 32 64 96 \
  --task-counts 128 \
  --checkout-strategies clone worktree copytree tar \
  --pytest-diagnostics \
  --prefix git_pytest_round7_diagnostics \
  --out artifacts/reports/git_pytest_round7_diagnostics_sweep.json \
  --summary-out artifacts/reports/git_pytest_round7_diagnostics_sweep.md

python scripts/analyze_io_pytest_diagnostics.py \
  --sweep artifacts/reports/git_pytest_round7_diagnostics_sweep.json \
  --out artifacts/reports/git_pytest_round7_diagnostics_analysis.md
```

Run the round8 stability sweep and production-cap validation:

```bash
python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --workers 8 16 32 48 64 96 128 \
  --task-counts 1024 \
  --checkout-strategies worktree \
  --warmup 0 \
  --repeat 3 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --pytest-plugin-policy auto \
  --mlx-size 128 \
  --mlx-iters 1 \
  --pressure-test \
  --command-timeout-sec 1800 \
  --prefix git_pytest_round8_stability \
  --out artifacts/reports/git_pytest_round8_stability_sweep.json \
  --summary-out artifacts/reports/git_pytest_round8_stability_sweep.md

python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --workers 32 64 96 \
  --task-counts 1024 \
  --checkout-strategies clone \
  --warmup 0 \
  --repeat 3 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --mlx-size 128 \
  --mlx-iters 1 \
  --pressure-test \
  --command-timeout-sec 1800 \
  --prefix git_pytest_round8_clone_baseline \
  --out artifacts/reports/git_pytest_round8_clone_baseline_sweep.json \
  --summary-out artifacts/reports/git_pytest_round8_clone_baseline_sweep.md
```

By default, Mac local normal runs cap workers to the current measured stable value
(`8` workers). Hard-timeout production runs can further apply repo-affinity cap so
tasks from the same repo family reuse the same worker cache. `--cache-policy auto`
is the default: stable repo-affinity production resolves to process-local cache,
while explicit `--pressure-test` resolves to shared checkout cache. Use
`--pressure-test` when intentionally testing 64/96/128-worker saturation.
Diagnostics mode caps to `32` workers unless pressure testing.

Run the current default pytest acceleration validation:

```bash
python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --workers 8 \
  --task-counts 1024 \
  --checkout-strategies worktree \
  --warmup 0 \
  --repeat 3 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --mlx-size 128 \
  --mlx-iters 1 \
  --pressure-test \
  --command-timeout-sec 900 \
  --prefix git_pytest_round10_default_accel \
  --out artifacts/reports/git_pytest_round10_default_accel_sweep.json \
  --summary-out artifacts/reports/git_pytest_round10_default_accel_sweep.md
```

Hidden pytest subprocesses use `--pytest-plugin-policy auto` by default. The scanner
keeps `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for generated local fixtures and automatically
opts out for external repositories that declare pytest plugin dependencies. Compatibility
aliases remain available: `--disable-pytest-plugin-autoload` maps to `disabled`, and
`--enable-pytest-plugin-autoload` maps to `enabled`.

Validate the plugin dependency scan:

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/pytest_plugin_policy_benchmark.json \
  --warmup 0 \
  --repeat 3 \
  --workers 2 \
  --task-timeout-sec 10 \
  --pytest-plugin-policy auto \
  --mlx-size 128 \
  --mlx-iters 1 \
  --skip-torch-mps \
  --out artifacts/reports/pytest_plugin_policy_auto_report.json \
  --summary-out artifacts/reports/pytest_plugin_policy_auto_summary.md
```

Run the current hard-timeout + repo-affinity production validation:

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --warmup 0 \
  --repeat 3 \
  --workers 32 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --scheduler-policy fixed \
  --task-distribution repo-shard-worksteal \
  --checkout-strategy worktree \
  --cache-policy auto \
  --pytest-plugin-policy auto \
  --mlx-size 128 \
  --mlx-iters 1 \
  --out artifacts/reports/shared_cache_auto_production_1024_report.json \
  --summary-out artifacts/reports/shared_cache_auto_production_1024_summary.md
```

Run shared cache + repo-shard pressure A/B when deliberately probing high-worker
cache fragmentation:

```bash
python scripts/run_worker_sweep.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --workers 4 8 16 32 \
  --task-counts 128 \
  --checkout-strategies worktree \
  --task-distribution repo-shard-worksteal \
  --cache-policy shared \
  --shared-cache-root artifacts/cache/shared_cache_ab_fast_cleanup \
  --warmup 0 \
  --repeat 3 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --pressure-test \
  --mlx-size 128 \
  --mlx-iters 1 \
  --prefix shared_cache_ab_shared_repo_shard_fast_cleanup \
  --out artifacts/reports/shared_cache_ab_shared_repo_shard_fast_cleanup_128_sweep.json \
  --summary-out artifacts/reports/shared_cache_ab_shared_repo_shard_fast_cleanup_128_sweep.md
```

Run adaptive pressure testing when deliberately probing saturation:

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/git_pytest_benchmark_1024.json \
  --max-tasks 128 \
  --warmup 0 \
  --repeat 3 \
  --workers 32 \
  --task-timeout-sec 30 \
  --pytest-timeout-sec 10 \
  --scheduler-policy adaptive \
  --checkout-strategy worktree \
  --pytest-plugin-policy auto \
  --pressure-test \
  --skip-torch-mps \
  --out artifacts/reports/timeout_scheduler_hard_adaptive_pressure_128_report.json \
  --summary-out artifacts/reports/timeout_scheduler_hard_adaptive_pressure_128_summary.md
```

Compare two benchmark reports:

```bash
python scripts/compare_benchmark_reports.py \
  --before artifacts/reports/toy_baseline_report.json \
  --after artifacts/reports/toy_optimized_report.json \
  --out artifacts/reports/mac_mlx_optimization_delta.md
```

Read the industrial benchmark interpretation:

```bash
sed -n '1,220p' docs/BENCHMARK_RESULTS.md
```

Generated local git fixtures and benchmark reports are intentionally ignored by
Git. Rebuild fixtures with `scripts/build_git_pytest_benchmark.py`; write large
run reports under `artifacts/reports/` for local analysis.

## AI Infra Positioning

The project treats benchmark creation and evaluation as infrastructure: typed contracts,
isolated execution, repeatable datasets, regression checks, observability hooks, and clear
adapter boundaries for LLM inference, vLLM serving, LoRA fine-tuning, and future GPU-scale
evaluation.

## Main References

The documentation cites public references such as SWE-bench, AgentBench, SWE-agent,
3D-LLM, 3D-LLava, ScanNet, LoRA, OpenAI-compatible vLLM serving, Hugging Face PEFT,
and Docker-based execution isolation.
