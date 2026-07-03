# 公开 SWE-Bench 风格 Teacher-Student 闭环

## 定位

本项目把 GPT-family frontier models 作为 teacher API providers，而不是可训练权重。可训练 student 是一个用 LLaMA-Factory 微调的开源模型，随后通过 OpenAI-compatible endpoint 提供服务，并由同一套公开 SWE-bench 风格 Work/Judge 闭环评测。

对外结论需要保持窄口径：这是一个受公开论文启发的 SWE 评测与训练反馈闭环。它不使用字节跳动内部数据、私有 benchmark tasks 或私有 metrics。参考边界见 `docs/PUBLIC_SWE_REFERENCE_MAP.md`。

## 本地流水线

1. 导入公开 SWE-style tasks：

```bash
sebench import-swe-dataset \
  --hf-dataset SWE-bench/SWE-bench_Lite \
  --limit 20 \
  --out artifacts/public_swe/dataset.json \
  --training-out artifacts/public_swe/dataset_version.json
```

2. bootstrap 阶段生成 teacher-authored synthetic data：

```bash
sebench generate-data \
  --requirements examples/requirements.json \
  --count 500 \
  --out artifacts/training/dataset_version.json
```

3. 导出 accepted examples 供 LLaMA-Factory 使用：

```bash
sebench export-llamafactory \
  --dataset-version artifacts/training/dataset_version.json \
  --out-dir artifacts/llamafactory/sebench_student_sft
```

4. 将导出目录移动到 Windows CUDA 机器，安装 LLaMA-Factory，然后运行：

```powershell
.\train_windows_cuda.ps1
```

5. 用 vLLM 或 LLaMA-Factory 的 OpenAI-style API 服务 student checkpoint，然后配置本项目：

```bash
export SEBENCH_STUDENT_PROVIDER=openai_compatible
export SEBENCH_STUDENT_OPENAI_BASE_URL=http://localhost:8000/v1
export SEBENCH_STUDENT_OPENAI_API_KEY=dummy
export SEBENCH_STUDENT_MODEL=sebench-student-lora
```

6. 生成 SWE-bench-style predictions：

```bash
sebench swe-predict \
  --dataset artifacts/public_swe/dataset.json \
  --role student \
  --out artifacts/public_swe/predictions.jsonl
```

7. 运行或暂存官方 SWE-bench harness 评测：

```bash
sebench run-swe-harness \
  --predictions artifacts/public_swe/predictions.jsonl \
  --dataset-name SWE-bench/SWE-bench_Lite \
  --split test \
  --out reports/swebench_harness_report.json
```

8. 在 held-out benchmark 上比较配置好的模型角色：

```bash
sebench evaluate-models \
  --dataset examples/git_pytest_benchmark.json \
  --roles base,student,teacher \
  --out-dir reports/model_evals
```

## 当前证据占位

本节只能填写已经实测的本地或云端运行结果。

| Run | Dataset | Model Role | Model ID | GPU | Pass Rate | Valid Task Rate | Latency Avg | Cost Est. |
|---|---|---|---|---|---:|---:|---:|---:|
| pending | pending | student | pending | pending | pending | pending | pending | pending |

## 保护边界

- 不要声称训练了 GPT-5.5 参数。
- 不要包含朋友、公司或内部来源的数据、任务、prompt、指标或姓名。
- 每个 dataset version 都要绑定 prompt version、teacher model ID、seed、split 和 validation status。
- generated hidden judges 在 deterministic local validation 确认可执行且不会通过 prompt 泄漏答案前，只能视为草稿。
- gold SWE patches 不得进入 evaluation prompts；只可在带 source 和 license metadata 的离线训练导出中使用。
