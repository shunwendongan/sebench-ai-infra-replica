# 公开论文复现说明

## Reproduction Boundary

本项目只复现公开可描述的工程方法，不使用任何内部代码、内部数据、内部评测题或未公开论文实现。所有数字指标都应来自本地运行报告或公开文献，不在代码中硬编码简历指标。

## Public Baselines to Compare

1. **SWE-bench style**: 用真实或 synthetic 软件工程任务评估 agent 是否能产出可验证提交。
2. **AgentBench/WebArena style**: 把任务环境和 judge 环境隔离，避免 agent 直接看到答案。
3. **SWE-agent style**: 将 agent 操作收敛到显式接口，方便评测和回放。
4. **3D-LLM/3D-LLava style**: 将 3D 场景信息结构化后提供给语言模型。
5. **LoRA/PEFT style**: 用低秩适配器复现实验，不修改大模型主体参数。

## Suggested Experiments

| Experiment | Baseline | Variant | Metric |
|---|---|---|---|
| Authoring quality | mock static task | LLM-generated + schema repair | valid task rate |
| Evaluation isolation | local runner | Docker Work/Judge runner | judge leakage rate, pass rate |
| Spatial reasoning | prompt only | geometry-prefix | direction accuracy, numeric MAE |
| Low-cost adaptation | zero-shot | LoRA rank=16 | pass rate, cost, latency |
| Regression stability | single run | fixed seed repeated runs | score variance |

## Minimal Claims Allowed

- The repository provides a runnable prototype for SE-Bench-like evaluation infrastructure.
- The spatial bridge correctly recovers translation and rotation on synthetic point sets.
- The authoring loop creates schema-valid benchmark tasks with deterministic mock inference.

## Claims Not Allowed Without Extra Evidence

- It reproduces ByteDance internal SE-Bench.
- It matches private benchmark metrics.
- It proves 3D-LLava/Qwen/LLaMA accuracy improvements without real model runs.
- It improves RL reward quality without a training loop and measured ablations.
