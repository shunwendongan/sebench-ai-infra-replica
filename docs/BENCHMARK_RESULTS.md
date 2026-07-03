# 基准测试结果

日期：2026-07-01

本项目先使用本地合成 fixture 验证工程闭环，再运行真实 SWE-bench 风格负载。Mac MLX/MPS 结果只用于本地工程验证和瓶颈分析，不代表 CUDA/GPU 集群吞吐结论。

## 当前类生产本地运行

命令族：`run_mac_mlx_benchmark.py`，启用 hard task timeout、repo-affinity worker cap、`task_distribution=repo-shard-worksteal` 和 `cache_policy=auto`。

| 数据集 | 请求 Workers | 实际 Workers | Pass Rate | Timeout | Throughput | Checkout Avg | Pytest Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 generated git+hidden pytest tasks | 32 | 4 | 1.000000 | 0 | 11.799176 tasks/s | 0.046374s | 0.234169s |

## 共享缓存压力测试

shared checkout cache 在刻意压测高 worker 数时有价值，但在这台 Mac fixture 上不是默认生产路径。

| 模式 | Workers | Pass Rate | Throughput | Checkout Avg | Pytest Avg | Cleanup Avg |
|---|---:|---:|---:|---:|---:|---:|
| process + load | 32 | 1.000000 | 4.565038 tasks/s | 0.278636s | 1.651093s | 0.001293s |
| shared + repo-shard + targeted cleanup | 32 | 1.000000 | 8.036103 tasks/s | 0.053264s | 0.273548s | 0.003259s |

## 结果解读

- 当前本地生产 cap 由 repo family affinity 控制：四个 generated repo family 映射为四个实际 worker。
- 剩余主要成本是 hidden pytest 子进程执行，而不是 MLX/MPS 计算。
- 下一步最有价值的 benchmark 是在 Linux x86_64 Docker 上跑 20-50 个 SWE-bench Lite 真实任务样本，再用 CUDA/A100 推理路径衡量 Agent 生成成本。

完整设计叙述见 `docs/SE_BENCH_REPRODUCTION_TECH_DESIGN.md` 和 `docs/SE_BENCH_MATURE_PROJECT_OPTIMIZATION_ANALYSIS.md`。
