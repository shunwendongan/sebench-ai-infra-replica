# 训练与服务 Recipe

这份 recipe 是可选路线。仓库默认使用 mock inference 就能运行。

teacher-student 反馈闭环见 `docs/STUDENT_LOOP_REPORT.md`。GPT-family frontier models 在这里作为 teacher API providers，用于数据生成和修复；可训练 student 是一个通过 LLaMA-Factory 微调、再通过 OpenAI-compatible endpoint 提供服务的开源模型。

## 数据

- Agent benchmark tasks：先使用 Hugging Face 上 `princeton-nlp/SWE-bench` 的小样本，或使用 `examples/golden_tasks.json` 中的 synthetic tasks。
- Spatial tasks：先使用合成 3D transforms；如果有可用授权数据，再替换为 ScanNet-derived 或 ScanQA-style 样本。
- Split：生成任务默认使用 80/10/10 train/validation/test；固定 seed，并报告精确 dataset version。

## 模型

默认开放模型候选：

- `Qwen/Qwen2.5-7B-Instruct`：容易接入 OpenAI-compatible serving 的 instruction model family。
- `meta-llama/Llama-3.2-3B-Instruct`：更小的 instruction model，但受 license/access 要求约束。

## LoRA 默认配置

代码暴露了 `LoRARecipe`：

```python
LoRARecipe(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    rank=16,
    alpha=32,
    dropout=0.05,
    learning_rate=2e-4,
    epochs=3,
    max_seq_length=4096,
)
```

这些值只能作为起点。任何真实实验都必须报告 hardware、global batch size、gradient accumulation、seed、optimizer、precision，以及 dataset commit/hash。

## vLLM 服务

启动 OpenAI-compatible endpoint，然后配置：

```bash
export SEBENCH_LLM_PROVIDER=openai_compatible
export SEBENCH_OPENAI_BASE_URL=http://localhost:8000/v1
export SEBENCH_OPENAI_API_KEY=dummy
```

当前 CLI 默认使用 `MockLLMClient`。运行真实推理时，可在 `cli.py` 或小型实验脚本中切换到 `OpenAICompatibleClient`。
