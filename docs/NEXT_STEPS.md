# 后续路线

这份路线图用于让项目持续对齐公开 SWE-bench 风格的评测与训练闭环。优先级按工程价值和可展示证据排序，而不是按功能体量排序。

## P0：真实公开 SWE 评测冒烟

- 用 `import-swe-dataset` 导入 `SWE-bench/SWE-bench_Lite`，或导入一份至少包含 20 个留出 issue 的小型 JSONL 样本。
- 准备一台带 Linux/Docker 的机器，并配置官方 SWE-bench harness。
- 使用 `swe-predict` 分别为 `base`、`student` 和 `teacher` 角色生成 predictions，然后用显式 harness command 运行 `run-swe-harness`。
- 将报告保存到 `reports/`，并汇总 resolved rate、invalid patch rate、patch apply failure、timeout count 和平均延迟。

## P1：接入真实模型的 Agent 运行器

- 将当前单轮 patch prompt 替换为小型 repo-context packer：issue 文本、文件树、选中代码片段和历史失败信息。
- 增加 patch budget 控制：最大 prompt tokens、最大输出 tokens、允许修改路径，以及 invalid unified diff 的重试次数。
- 为每次模型尝试记录一个 artifact：prompt hash、model ID、原始输出、解析后的 patch、patch 校验结果和最终 judge 状态。
- 在单轮 patch 路径具备稳定指标前，暂不扩展到多步 shell agent。

## P2：扩展训练数据

- 增加公开成功轨迹适配器，例如 SWE-smith trajectories 和 SWE-Gym/OpenHands SFT trajectories。
- 导出两类 LLaMA-Factory 数据集：`issue_to_patch` 和 `trajectory_action`。
- 按 license、source、context length、duplicate patch hash 和 held-out split isolation 过滤样本。
- 在每份模型对比报告中记录 dataset version ID。

## P3：学生模型训练与服务

- 在 Windows CUDA 机器上用生成的 LLaMA-Factory 配置训练 QLoRA student。
- 通过 vLLM 或 LLaMA-Factory 的 OpenAI-style API 对外提供 adapter 服务。
- 在同一组 held-out public SWE tasks 上比较 base、student 和 teacher。
- 只报告实测指标；GPT-family 模型只作为 teacher API，不作为可训练权重。

## P4：平台化加固

- 为更大的 run records 和 artifacts 增加 Postgres/object-storage 路径。
- 为数据导入、模型推理、patch 解析、patch apply、harness runtime 和 scoring 增加 OpenTelemetry spans。
- 为 toy hidden-pytest tasks 增加 CI job；完整 SWE-bench harness 运行由于 Docker 和耗时成本，应保留为显式/manual job。
- 增加 reproducibility bundle，记录 commit hash、dataset version、model config、命令行、环境和硬件信息。

## 简历证据清单

- 公开参考边界：`docs/PUBLIC_SWE_REFERENCE_MAP.md`。
- 至少一个 public SWE import artifact。
- 至少一个 model comparison report。
- 至少一份带失败类别的 failed-case analysis。
- 清晰边界声明：不使用内部代码、数据、协议或私有指标。
