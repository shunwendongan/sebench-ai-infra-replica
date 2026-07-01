# SE-Bench 成熟项目经验与性能优化分析

日期：2026-06-30
对象：`SE-Bench 长程 Agent 评测基准工程复现技术设计文档 v1.1`
目标：基于成熟论文与 GitHub 高赞项目，给出降低开销、提高缓存命中率、提高 pass rate/准确率的工程改造建议。

## 1. 分析结论

当前复现主线应聚焦“真实软件工程任务 benchmark 基础设施”，不要先把精力放在重 UI、多 Agent 花哨编排或 3D 扩展任务。最优先的工程路线是：

1. 用 SWE-bench 的 `task instance -> patch prediction -> Docker/Judge` 作为数据与评测协议。
2. 用 mini-swe-agent/SWE-agent 的最小 Agent loop 作为 baseline，先跑通而不是先堆工具。
3. 用 SWE-ReX 的 runtime 解耦思想改造 Work 容器执行层。
4. 用 Aider repo map 降低上下文成本并提升关键文件召回。
5. 用 lm-evaluation-harness/openai/evals 的 registry、config、grader 思路让评测配置可版本化。
6. 用 LangGraph 的 durable workflow 思想做长程任务 checkpoint，但放在 P2，避免早期复杂化。

## 2. 成熟项目可迁移经验

GitHub 星标来自 GitHub API，查询时间为 2026-06-30。

| 项目 | Stars | 核心经验 | 迁移到 SE-Bench 的位置 | 优先级 |
|---|---:|---|---|---|
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | 78774 | 多后端 agent runtime、workspace、agent server、自动化入口 | 后续产品化和远程执行后端 | P2 |
| [microsoft/autogen](https://github.com/microsoft/autogen) | 59373 | 多 Agent 编排、模型客户端、工具工作台 | Authoring Agent/Judge Agent 抽象参考 | P2 |
| [Aider-AI/aider](https://github.com/Aider-AI/aider) | 46860 | repo map、符号图、token budget、diff/edit 流程 | `RepoMapBuilder`、context ranking、patch 生成 | P0 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 36120 | 状态图、checkpoint、可恢复执行、长期状态 | `WorkflowState`、Authoring/Judge pipeline 恢复 | P2 |
| [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) | 19674 | Agent-computer interface、batch benchmark、轨迹浏览 | Agent loop、tool trace、trajectory schema | P0 |
| [openai/evals](https://github.com/openai/evals) | 18795 | eval registry、grader、completion function 协议 | task registry、judge adapter、结果格式 | P1 |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 13121 | YAML config、模型后端解耦、batch/async、指标聚合 | `configs/*.yaml`、backend interface、report | P1 |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 5490 | bash-only、线性 history、极低脚手架成本 | 最小 baseline、toy tasks、trajectory 训练导出 | P0 |
| [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | 5300 | Docker harness、prediction JSON、instance schema、split | task/run schema、Judge harness、leaderboard 数据 | P0 |
| [SWE-agent/SWE-ReX](https://github.com/SWE-agent/SWE-ReX) | 543 | sandboxed shell、local/cloud backend、并行 shell | `ExecutionBackend`、Work runtime、并发执行 | P0 |

注意：AutoGen 官方 README 已提示维护模式，新项目不建议把它作为主依赖；OpenHands 体量大、偏产品化，也不适合作为最小复现起点。Aider、SWE-bench、mini-swe-agent、SWE-ReX 的工程经验更直接。

## 3. 文献阅读优先级

| 级别 | 文献 | 理由 |
|---|---|---|
| P0 | SWE-bench、SWE-agent、WebArena、AgentBench、OSWorld | 决定 task schema、环境协议、Agent loop、Judge 与长程任务设计 |
| P1 | ReAct、Reflexion、Toolformer、LLM Agents Survey | 决定 tool trace、失败反馈、自我修复、Agent 架构分类 |
| P2 | ScanNet、ScanQA、3D-LLM、3D-LLaVA、Kabsch、Umeyama | 只服务 3D 空间推理扩展任务，不影响主线可运行闭环 |

引用强度只能作为阅读优先级辅助。主线最相关的是 SWE-bench/SWE-agent/WebArena/AgentBench/OSWorld，即使 SWE-agent 较新、引用数较低，也比通用高引综述更值得优先读实现。

## 4. 当前文档改造点

原文档已经覆盖背景、架构、流程、数据链路、隔离、复现步骤和验收标准，但有四个工程缺口：

| 缺口 | 影响 | v1.1 改造 |
|---|---|---|
| 缓存设计停留在原则 | 无法衡量命中率，也无法判断缓存污染 | 增加 base image、dependency layer、repo snapshot、repo map、prompt、verdict 的 cache key |
| 性能指标不够闭环 | 只知道耗时，不知道成本与准确率关系 | 增加 `cost_per_pass`、`context_recall@k`、`early_stop_saved_time`、`flaky_rate` |
| 准确率优化没有归因 | pass rate 低时容易直接换模型 | 增加 `task_invalid/context_miss/wrong_edit/tool_failure/test_overfit/judge_flaky/format_error` |
| 成熟项目只列链接 | 评审无法看出借鉴了什么 | 增加项目经验映射表与 P0/P1/P2 迁移优先级 |

## 5. 命中率提升方案

缓存分四层做，不要把所有产物扔进同一个 artifact store。

| 层 | 目标 | cache key | 期望命中 |
|---|---|---|---:|
| 构建层 | 避免每题构建环境 | `base_image_key`、`dependency_layer_key` | base > 90%，dependency > 70% |
| 数据层 | 避免重复 checkout/index | `repo_snapshot_key`、`repo_map_key` | snapshot > 80%，repo map > 85% |
| LLM 层 | 避免重复静态 prompt/token | `model + params + messages_hash + tool_schema_hash` | deterministic prompt > 50% |
| Judge 层 | 避免重复验证同 patch | `task_id + patch_hash + judge_version + hidden_tests_hash` | 视重复 run 而定 |

缓存污染风险控制：

- 所有 Judge cache 必须绑定 `hidden_tests_hash`。
- 所有 LLM cache 必须绑定模型、采样参数和工具 schema。
- 所有 repo map cache 必须绑定 parser version 和 ignore policy。
- 命中结果要写入 run record，方便复盘命中率和异常 pass。

## 6. 准确率/pass rate 提升方案

准确率不是一个单点模型问题，应拆成四条链路：

| 链路 | 指标 | 优化动作 |
|---|---|---|
| 题目质量 | `valid_task_rate`、gold patch pass、negative patch fail | Build/Oracle/Negative/Leakage 四道门禁 |
| 上下文召回 | `context_recall@k`、关键文件是否进入 context | Aider-style repo map、stack trace retrieval、import graph ranking |
| Agent 执行 | `tool_error_count`、重复失败命令、edit/test 循环次数 | mini baseline、shell action 标准化、早停与 retry 上限 |
| Judge 稳定 | `flaky_rate`、verdict replay 差异 | 规则优先、隐藏测试复跑、LLM Judge 降级为兜底 |

推荐每 20-50 题做一次失败审计：抽样失败 trajectory，标注失败类型，再决定改模型、改 prompt、改 repo map、改工具还是修题。不要只看 aggregate pass rate。

## 7. 降低开销方案

| 开销 | 主要来源 | 降低方式 | 验收 |
|---|---|---|---|
| 构建时间 | Docker build、依赖安装 | 预构建 base image、依赖层缓存、受控 mirror | cold start 下降，setup failure < 5% |
| checkout/index | repo clone、静态分析 | repo snapshot tarball、repo map cache | checkout p95 下降 |
| LLM token | 超大上下文、长 trajectory | repo map Top-K、trajectory summary、static prompt cache | input tokens/task 下降 |
| 失败重试 | 无效循环、重复测试 | failure-aware early stop、重复命令去重 | `early_stop_saved_time` 上升 |
| Judge 成本 | 重复测试、LLM Judge | verdict cache、规则/测试优先 | LLM Judge ratio < 10% |
| 存储 | full trajectory/log | 热冷分层、zstd 压缩、摘要索引 | 热存储增长可控 |

## 8. 推荐工程改造路线

### P0：两周内应完成

1. 固化 `TaskSpec`、`RunRecord`、`DatasetManifest` schema。
2. 实现 Work/Judge 双容器或至少双目录隔离，提交路径走白名单。
3. 接入 mini-swe-agent 风格 baseline，先跑至少 3 个 toy repo task；当前 round2 已扩展到 7 个 toy task。
4. 按 SWE-bench prediction JSON 设计 patch adapter。
5. 每个 run 写入 fingerprint：`dataset_hash + image_digest + base_commit + agent_config_hash + judge_version`。

### P1：一个月内应完成

1. 增加 repo map/context builder，记录 `context_recall@k`。
2. 增加 image、dependency、snapshot、repo map、prompt、judge verdict 缓存。
3. 增加 failure taxonomy 和自动报告。
4. 增加 judge cascade：schema -> static patch -> hidden tests -> LLM judge。
5. 跑 SWE-bench Lite 小样本，生成按 repo/难度/失败类型分桶的报告。

### P2：稳定后扩展

1. 接入 LangGraph-style checkpoint，支持长程任务恢复。
2. 接入 SWE-ReX/OpenHands-style remote backend，支持多机并发。
3. 增加 Authoring Agent 自动修题 loop。
4. 引入 3D 空间推理扩展任务族，使用 Kabsch/Umeyama solver 做确定性 Judge。

## 9. 验收指标

| 目标 | 指标 |
|---|---|
| 最小闭环 | 至少 3 个 toy task 完成 authoring、work、judge、record、report；当前 round2 已完成 7 个 |
| 隔离可信 | Work 无法读取 hidden tests、oracle、judge prompt |
| 可复现 | 固定 seed/image/base commit/judge version 后复跑结果一致 |
| 命中率 | base image > 90%，snapshot > 80%，repo map > 85% |
| 准确率 | pass rate 按任务族提升，且失败类型可解释 |
| 成本 | `cost_per_pass` 不高于上一个稳定版本，或能解释成本换取的准确率收益 |
| Judge 稳定 | flaky rate < 2%，LLM Judge ratio < 10% |

## 10. 参考来源

- [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench)
- [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent)
- [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
- [SWE-agent/SWE-ReX](https://github.com/SWE-agent/SWE-ReX)
- [Aider repository map docs](https://aider.chat/docs/repomap.html)
- [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands)
- [microsoft/autogen](https://github.com/microsoft/autogen)
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- [openai/evals](https://github.com/openai/evals)
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [THUDM/AgentBench](https://github.com/THUDM/AgentBench)
- [web-arena-x/webarena](https://github.com/web-arena-x/webarena)
- [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
