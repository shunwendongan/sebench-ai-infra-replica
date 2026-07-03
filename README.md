# SE-Bench AI Infra 复现项目

这是一个基于公开论文和开源工程模式搭建的 AI Infra 复现原型，主题覆盖 SE-Bench 风格长程 Agent 评测、benchmark authoring agent、Work/Judge 隔离评测，以及 3D 空间推理诊断。项目不包含字节跳动内部代码、内部数据或私有系统实现，也不声称复现任何未公开系统。

本仓库使用公开论文、开源 benchmark 设计和合成 fixture，目标是让工程链路可以在本地被检查、运行和复盘。

## 项目内容

- 将自然语言需求转换为强类型 benchmark task 的题目构建闭环。
- 用 SQLite artifact store 管理 benchmark item、数据集版本和运行记录。
- 提供带路径白名单校验的 Work/Judge 评测编排器和本地 mock runner。
- 提供 3D 空间推理诊断模块，使用质心位移和 Kabsch SVD 做确定性几何判断。
- 提供 FastAPI 服务入口和 Typer CLI 入口。
- 提供架构图、复现说明、公开参考边界、实验路线和 benchmark 结果文档。

主要文档入口：

- [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)：原型技术设计。
- [docs/SE_BENCH_REPRODUCTION_TECH_DESIGN.md](docs/SE_BENCH_REPRODUCTION_TECH_DESIGN.md)：v2.5 扩展复现设计，覆盖文献优先级、成熟开源项目经验、Work/Judge 隔离、数据链路、性能优化、缓存策略、Mac MLX/MPS 本地 benchmark 边界、真实评测闭环、各轮实测结果和验收标准。
- [docs/SE_BENCH_MATURE_PROJECT_OPTIMIZATION_ANALYSIS.md](docs/SE_BENCH_MATURE_PROJECT_OPTIMIZATION_ANALYSIS.md)：成熟项目经验与性能优化分析。
- [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md)：当前 benchmark 摘要。
- [docs/PUBLIC_SWE_REFERENCE_MAP.md](docs/PUBLIC_SWE_REFERENCE_MAP.md)：公开 SWE 生态参考边界。
- [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)：后续路线。
- [docs/diagrams/](docs/diagrams/)：可编辑 Mermaid 图源；`docs/*.mmd` 是对应镜像文件。

## 快速开始

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

如果移动了本地 checkout 或虚拟环境里的 `sebench` 指向旧路径，重新安装当前项目入口即可：

```bash
python -m pip install -e . --no-deps
python -c "import sebench_infra; print(sebench_infra.__file__)"
sebench --help
sebench evaluate --dataset examples/toy_benchmark.json --out artifacts/reports/smoke_toy_report.json
```

Apple Silicon 本地 benchmark 环境：

```bash
python3 -m venv .venv-apple
source .venv-apple/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,apple]"
```

只有在需要本地 Apple GPU 对比时才安装 PyTorch MPS：

```bash
python -m pip install -e ".[mps]"
```

运行 mock 复现流水线：

```bash
sebench reproduce \
  --requirements examples/requirements.json \
  --spatial-scene examples/synthetic_spatial_scene.json \
  --out reports/demo_run.json
```

构建 teacher-student 训练数据并导出 LLaMA-Factory 数据：

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

学生模型在 CUDA 机器上训练并通过 OpenAI-compatible endpoint 提供服务后，可比较不同模型角色：

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

本机已安装 Codex CLI 时，可以显式选择实验性 Codex agent runner 做小样本执行路径 smoke。该路径用于验证 agent 接入，不作为默认公平 benchmark：

```bash
sebench evaluate \
  --dataset examples/patch_benchmark.json \
  --max-tasks 1 \
  --agent-backend codex_cli \
  --codex-binary codex \
  --codex-timeout-sec 300 \
  --out artifacts/reports/codex_cli_smoke_report.json
```

启动 API 服务：

```bash
uvicorn sebench_infra.api:app --reload
```

## 常用 benchmark 命令

运行 Apple Silicon/MLX toy true-loop benchmark：

```bash
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/toy_benchmark.json \
  --out artifacts/reports/toy_baseline_report.json \
  --summary-out artifacts/reports/mac_mlx_baseline_summary.md
```

运行当前 round2 benchmark，并做 MLX 与 Torch MPS sanity 检查：

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

运行独立 Apple Silicon CPU/MPS/MLX sanity：

```bash
python scripts/benchmark_apple_silicon.py \
  --size 2048 \
  --warmup 1 \
  --iters 5 \
  --out artifacts/reports/apple_silicon_round2_sanity_2048.json
```

构建并运行 round3 仿真真实 repo benchmark：

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

构建并运行 round4 patch-submission benchmark：

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

构建并运行 round6 本地 git + hidden pytest worker sweep：

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

运行 round7 hidden pytest 与文件系统 I/O 诊断：

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

运行 round8 稳定性 sweep 和生产 cap 验证：

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

Mac 本地普通运行默认把 worker 限制为当前实测稳定值 `8`。生产 hard-timeout 路径还会应用 repo-affinity cap，让同一 repo family 的任务复用同一 worker cache。`--cache-policy auto` 是默认策略：稳定 repo-affinity 生产路径会解析为 process-local cache，显式 `--pressure-test` 会解析为 shared checkout cache。只有在刻意测试 64/96/128-worker 饱和时才使用 `--pressure-test`。诊断模式默认限制为 `32` workers，除非显式开启压力测试。

运行当前默认 pytest 加速验证：

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

hidden pytest 子进程默认使用 `--pytest-plugin-policy auto`。扫描器会对本地生成 fixture 保持 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，并在外部仓库声明 pytest 插件依赖时自动退出该加速路径。兼容别名仍可使用：`--disable-pytest-plugin-autoload` 映射到 `disabled`，`--enable-pytest-plugin-autoload` 映射到 `enabled`。

验证插件依赖扫描：

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

运行当前 hard-timeout + repo-affinity 生产验证：

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

刻意探测高 worker cache fragmentation 时，可做 shared cache + repo-shard 压力 A/B：

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

刻意探测饱和点时，可运行 adaptive pressure testing：

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

比较两个 benchmark 报告：

```bash
python scripts/compare_benchmark_reports.py \
  --before artifacts/reports/toy_baseline_report.json \
  --after artifacts/reports/toy_optimized_report.json \
  --out artifacts/reports/mac_mlx_optimization_delta.md
```

查看工业化 benchmark 解读：

```bash
sed -n '1,220p' docs/BENCHMARK_RESULTS.md
```

本地生成的 git fixture 和 benchmark report 会被 Git 忽略。需要复建 fixture 时运行 `scripts/build_git_pytest_benchmark.py`；较大的本地运行报告建议写入 `artifacts/reports/` 用于分析。

## AI Infra 定位

本项目把 benchmark 创建和评测视为基础设施能力：强类型契约、隔离执行、可重复数据集、回归检查、可观测性 hook，以及面向 LLM 推理、vLLM serving、LoRA 微调和未来 GPU 规模评测的清晰 adapter 边界。

## 主要公开参考

文档引用的公开参考包括 SWE-bench、AgentBench、SWE-agent、3D-LLM、3D-LLava、ScanNet、LoRA、OpenAI-compatible vLLM serving、Hugging Face PEFT，以及基于 Docker 的执行隔离。
