# Apple Silicon MPS/MLX Benchmarking

This project includes an optional Apple Silicon probe for AI Infra performance notes.
It is intentionally outside the default test path because PyTorch MPS and MLX are heavy,
platform-specific dependencies.

## Environment

For project-local development, prefer a local virtual environment:

```bash
cd sebench-ai-infra-replica
python3 -m venv .venv-apple
source .venv-apple/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,apple]"
```

The `apple` extra installs MLX only. PyTorch MPS is optional and can be installed with
`python -m pip install -e ".[mps]"` if a run explicitly needs PyTorch comparison. In the
current migrated Documents/dev environment, Torch MPS is intentionally not required for
the default benchmark path and is reported as skipped when not installed.

## Run

```bash
source .venv-apple/bin/activate
python scripts/benchmark_apple_silicon.py \
  --size 1024 \
  --warmup 2 \
  --iters 5 \
  --out reports/apple_silicon_benchmark.json
```

True-loop toy benchmark:

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

Standalone CPU/MPS/MLX sanity:

```bash
source .venv-apple/bin/activate
python scripts/benchmark_apple_silicon.py \
  --size 2048 \
  --warmup 1 \
  --iters 5 \
  --out artifacts/reports/apple_silicon_round2_sanity_2048.json
```

## Fairness Rules

- Use the same matrix size, dtype, seed, warmup, and iteration count for all backends.
- Synchronize GPU work before stopping the timer: `torch.mps.synchronize()` for MPS and
  `mx.eval()` for MLX.
- Treat small speedup changes as noise unless repeated runs show stable separation.
- Do not claim that importing MLX accelerates NumPy; the computation must use MLX tensors.
- Report skipped backends instead of failing the whole benchmark.

## How to Use the Result

Use the JSON output as supporting AI Infra evidence, not as a model-quality metric. It
answers whether local Apple GPU kernels can accelerate the small matrix workloads used by
the prototype; it does not measure LLM serving throughput or benchmark task quality.

Use `run_mac_mlx_benchmark.py` for benchmark-loop evidence. Its result measures the local
toy Work/Judge loop, cache flags, pass rate, Judge timing, MLX sanity timing, and optional
Torch MPS sanity timing. It still does not represent CUDA or GPU-cluster industrial
throughput.

Round2 produced these local-only reference points:

| Workload | Backend | Mean seconds | GFLOP/s mean |
|---|---|---:|---:|
| 512x512 matmul | MLX GPU | 0.001022 | 262.661 |
| 512x512 matmul | Torch MPS | 0.000766 | 350.396 |
| 2048x2048 matmul | NumPy CPU | 0.020298 | 846.392 |
| 2048x2048 matmul | Torch MPS | 0.011284 | 1522.505 |
| 2048x2048 matmul | MLX GPU | 0.011784 | 1457.910 |

The 512x512 results mainly confirm that both Apple GPU paths are callable. The 2048x2048
run is a better local sanity signal, but final industrial throughput still requires a
CUDA/server benchmark.
