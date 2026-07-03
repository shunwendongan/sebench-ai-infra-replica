# 公开 SWE 生态参考边界

本项目是一个公开 SWE-bench 风格的评测与训练闭环。它不复现字节跳动内部系统、私有 SE-Bench 协议、私有数据集或私有指标。

## 主要公开参考

| 公开参考 | 在项目中的作用 | 本地实现 |
|---|---|---|
| [SWE-bench](https://github.com/swe-bench/SWE-bench) | 真实 issue 实例、Docker/harness 评测、prediction 文件约定 | `SWEIssueInstance`、`write_swe_predictions`、可选 `SWEHarnessRunner` |
| [SWE-bench datasets](https://www.swebench.com/SWE-bench/guides/datasets/) | Lite、Verified 等公开 split | `import-swe-dataset --hf-dataset ...` |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | 可规模化合成 SWE 任务 | `ExternalBenchmarkSource` 与 SWE-style import/export |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | SWE agents 的训练环境与轨迹 | `TrainingTaskKind.TRAJECTORY_ACTION` 导出路径 |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | 可用于 SFT-style 训练的公开成功轨迹 | LLaMA-Factory export metadata 与后续 trajectory adapters |
| [OpenHands SFT trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 软件工程 Agent 的公开 action trajectories | `trajectory_action` 训练任务类型 |

## 数据边界

- 评测 prompt 可以包含 issue 文本、仓库名、base commit 和公开 metadata。
- 评测 prompt 不得包含 gold patch、hidden test patch 内容或 oracle answers。
- gold patch 只能出现在离线训练样本或分析 artifact 中，并且必须带明确 source 和 license metadata。

## 本地模块

- `sebench_infra.training_loop.swe`：解析 SWE-bench/Hugging Face 风格数据行，保留 provenance，并将 issue 转换为内部 `TaskSpec`。
- `sebench_infra.training_loop.patch_agent`：单轮真实模型 patch 生成，以及本地 unified-diff 校验。
- `sebench_infra.training_loop.swe_harness`：官方 SWE-bench harness 命令和 prediction JSONL 生成的可选适配器。
- `sebench_infra.training_loop.export`：为 `task_spec_authoring`、`issue_to_patch` 和 `trajectory_action` 样本导出 LLaMA-Factory SFT 数据。

## 简历安全表述

可以使用：

> 基于公开 SWE-bench 风格搭建长程软件工程 Agent 评测与训练闭环，包含强类型任务转换、prediction 导出、Work/Judge 兼容评分，以及 LLaMA-Factory student 数据导出。

避免使用：

> 复现了字节跳动内部 SE-Bench、使用了内部协议，或训练了 GPT-5.5 参数。
