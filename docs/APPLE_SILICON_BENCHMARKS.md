# Apple Silicon MPS/MLX 本地基准测试

本项目包含一个可选 Apple Silicon 探针，用于补充 AI Infra 性能分析。它不在默认测试路径中，因为 PyTorch MPS 和 MLX 都是较重且平台相关的依赖。

## 环境

项目本地开发建议使用单独虚拟环境：

```bash
cd sebench-ai-infra-replica
python3 -m venv .venv-apple
source .venv-apple/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,apple]"
```

`apple` extra 只安装 MLX。PyTorch MPS 是可选项；只有明确需要 PyTorch 对比时，才使用 `python -m pip install -e ".[mps]"` 安装。当前迁移后的 `Documents/dev` 环境默认 benchmark 路径不要求 Torch MPS，未安装时会报告为 skipped。

## 运行

```bash
source .venv-apple/bin/activate
python scripts/benchmark_apple_silicon.py \
  --size 1024 \
  --warmup 2 \
  --iters 5 \
  --out reports/apple_silicon_benchmark.json
```

toy true-loop benchmark：

```bash
source .venv-apple/bin/activate
python scripts/run_mac_mlx_benchmark.py \
  --manifest examples/toy_benchmark.json \
  --warmup 1 \
  --repeat 5 \
  --mlx-size 512 \
  --mlx-iters 5 \
  --out artifacts/reports/toy_round2_report.json \
  --summary-out artifacts/reports/mac_mlx_round2_summary.md
```

独立 CPU/MPS/MLX sanity：

```bash
source .venv-apple/bin/activate
python scripts/benchmark_apple_silicon.py \
  --size 2048 \
  --warmup 1 \
  --iters 5 \
  --out artifacts/reports/apple_silicon_round2_sanity_2048.json
```

## 公平性规则

- 所有 backend 使用相同 matrix size、dtype、seed、warmup 和 iteration count。
- 计时停止前要同步 GPU work：MPS 使用 `torch.mps.synchronize()`，MLX 使用 `mx.eval()`。
- 小幅 speedup 变化应视为噪声，除非重复运行能显示稳定差距。
- 不要声称“导入 MLX 就能加速 NumPy”；计算必须真正使用 MLX tensors。
- backend 缺失时报告 skipped，而不是让整个 benchmark 失败。

## 如何使用结果

JSON 输出只能作为 AI Infra 支撑证据，不能作为模型质量指标。它回答的是：本地 Apple GPU kernel 能否加速原型中使用的小矩阵负载；它不衡量 LLM serving throughput，也不衡量 benchmark task quality。

`run_mac_mlx_benchmark.py` 用于获取 benchmark-loop 证据。它的结果衡量本地 toy Work/Judge loop、cache flags、pass rate、Judge timing、MLX sanity timing，以及可选 Torch MPS sanity timing。它仍然不代表 CUDA 或 GPU 集群的工业吞吐。

round2 产生了以下仅限本地参考的结果：

| Workload | Backend | Mean seconds | GFLOP/s mean |
|---|---|---:|---:|
| 512x512 matmul | MLX GPU | 0.001022 | 262.661 |
| 512x512 matmul | Torch MPS | 0.000766 | 350.396 |
| 2048x2048 matmul | NumPy CPU | 0.020298 | 846.392 |
| 2048x2048 matmul | Torch MPS | 0.011284 | 1522.505 |
| 2048x2048 matmul | MLX GPU | 0.011784 | 1457.910 |

512x512 结果主要确认两条 Apple GPU 路径都可调用。2048x2048 运行是更有价值的本地 sanity signal，但最终工业吞吐仍需要 CUDA/server benchmark。
