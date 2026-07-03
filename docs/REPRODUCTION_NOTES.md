# 公开论文复现说明

## 复现边界

本项目只复现公开可描述的工程方法，不使用任何内部代码、内部数据、内部评测题或未公开论文实现。所有数字指标都应来自本地运行报告或公开文献，不在代码中硬编码简历指标。

## 可对比的公开 Baseline

1. **SWE-bench style**：用真实或 synthetic 软件工程任务评估 Agent 是否能产出可验证提交。
2. **AgentBench/WebArena style**：隔离任务环境和 Judge 环境，避免 Agent 直接看到答案。
3. **SWE-agent style**：将 Agent 操作收敛到显式接口，方便评测和回放。
4. **3D-LLM/3D-LLava style**：将 3D 场景信息结构化后提供给语言模型。
5. **LoRA/PEFT style**：用低秩适配器复现实验，不修改大模型主体参数。

## 建议实验

| 实验 | Baseline | 变体 | 指标 |
|---|---|---|---|
| 题目生成质量 | mock static task | LLM-generated + schema repair | valid task rate |
| 评测隔离 | local runner | Docker Work/Judge runner | judge leakage rate、pass rate |
| 空间推理 | prompt only | geometry-prefix | direction accuracy、numeric MAE |
| 低成本适配 | zero-shot | LoRA rank=16 | pass rate、cost、latency |
| 回归稳定性 | single run | fixed seed repeated runs | score variance |

## 当前允许的最小结论

- 仓库提供了一个可运行的 SE-Bench-like 评测基础设施原型。
- spatial bridge 能在合成点集上正确恢复平移和旋转。
- authoring loop 能用 deterministic mock inference 生成 schema-valid benchmark tasks。

## 没有额外证据时禁止声称

- 项目复现了字节跳动内部 SE-Bench。
- 项目达到了私有 benchmark 指标。
- 在没有真实模型运行的情况下证明 3D-LLava/Qwen/LLaMA 准确率提升。
- 在没有训练闭环和实测 ablation 的情况下证明 RL reward 质量提升。
