# Benchmark Results

Date: 2026-07-01

This project uses local synthetic fixtures to validate the engineering loop before
running real SWE-bench style workloads. Mac MLX/MPS results are for local
engineering validation and bottleneck analysis only; they are not CUDA/GPU cluster
throughput claims.

## Current Production-like Local Run

Command family: `run_mac_mlx_benchmark.py` with hard task timeout, repo-affinity
worker cap, `task_distribution=repo-shard-worksteal`, and `cache_policy=auto`.

| Dataset | Requested Workers | Effective Workers | Pass Rate | Timeout | Throughput | Checkout Avg | Pytest Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 generated git+hidden pytest tasks | 32 | 4 | 1.000000 | 0 | 11.799176 tasks/s | 0.046374s | 0.234169s |

## Shared Cache Pressure Test

The shared checkout cache is useful when deliberately stress testing high worker
counts, but it is not the default production path on this Mac fixture.

| Mode | Workers | Pass Rate | Throughput | Checkout Avg | Pytest Avg | Cleanup Avg |
|---|---:|---:|---:|---:|---:|---:|
| process + load | 32 | 1.000000 | 4.565038 tasks/s | 0.278636s | 1.651093s | 0.001293s |
| shared + repo-shard + targeted cleanup | 32 | 1.000000 | 8.036103 tasks/s | 0.053264s | 0.273548s | 0.003259s |

## Interpretation

- The current local production cap is controlled by repo family affinity: four
  generated repo families map to four effective workers.
- The dominant remaining cost is hidden pytest subprocess execution, not MLX/MPS
  compute.
- The next meaningful benchmark step is a real 20-50 task SWE-bench Lite sample on
  Linux x86_64 Docker, then a CUDA/A100 inference path for agent generation cost.

For the full design narrative, see
`docs/SE_BENCH_REPRODUCTION_TECH_DESIGN.md` and
`docs/SE_BENCH_MATURE_PROJECT_OPTIMIZATION_ANALYSIS.md`.
