# Training and Serving Recipe

This recipe is intentionally optional. The default repo runs with mock inference.

## Data

- Agent benchmark tasks: start with a tiny sample from `princeton-nlp/SWE-bench` on Hugging Face or synthetic tasks from `examples/golden_tasks.json`.
- Spatial tasks: start with synthetic 3D transforms, then replace with licensed ScanNet-derived or ScanQA-style samples if available.
- Split: 80/10/10 train/validation/test for generated tasks; keep a fixed seed and report exact dataset version.

## Model

Default open model candidates:

- `Qwen/Qwen2.5-7B-Instruct`: accessible instruction model family for OpenAI-compatible serving.
- `meta-llama/Llama-3.2-3B-Instruct`: smaller instruction model, subject to license/access requirements.

## LoRA Defaults

The code exposes `LoRARecipe`:

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

Use these only as a starting point. Report hardware, global batch size, gradient accumulation, seed, optimizer, precision, and dataset commit/hash for any real experiment.

## vLLM Serving

Run an OpenAI-compatible endpoint, then configure:

```bash
export SEBENCH_LLM_PROVIDER=openai_compatible
export SEBENCH_OPENAI_BASE_URL=http://localhost:8000/v1
export SEBENCH_OPENAI_API_KEY=dummy
```

The current CLI uses `MockLLMClient` by default. Swap in `OpenAICompatibleClient` inside `cli.py` or a small experiment script when running real inference.
