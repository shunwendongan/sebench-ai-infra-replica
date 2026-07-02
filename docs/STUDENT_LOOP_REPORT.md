# Public SWE-Bench-Style Teacher-Student Loop

## Positioning

This project treats GPT-family frontier models as teacher API providers, not as
trainable weights. The trainable student is an open-source model fine-tuned with
LLaMA-Factory, then served through an OpenAI-compatible endpoint and evaluated by
the same public SWE-bench-style Work/Judge loop.

The public claim is intentionally narrow: this is a public-paper-inspired SWE
evaluation and training feedback loop. It does not use ByteDance internal data,
private benchmark tasks, or private metrics. See
`docs/PUBLIC_SWE_REFERENCE_MAP.md` for the reference boundary.

## Local Pipeline

1. Import public SWE-style tasks:

```bash
sebench import-swe-dataset \
  --hf-dataset SWE-bench/SWE-bench_Lite \
  --limit 20 \
  --out artifacts/public_swe/dataset.json \
  --training-out artifacts/public_swe/dataset_version.json
```

2. Generate teacher-authored synthetic data when bootstrapping:

```bash
sebench generate-data \
  --requirements examples/requirements.json \
  --count 500 \
  --out artifacts/training/dataset_version.json
```

3. Export accepted examples for LLaMA-Factory:

```bash
sebench export-llamafactory \
  --dataset-version artifacts/training/dataset_version.json \
  --out-dir artifacts/llamafactory/sebench_student_sft
```

4. Move the exported directory to the Windows CUDA machine, install
LLaMA-Factory, then run:

```powershell
.\train_windows_cuda.ps1
```

5. Serve the student checkpoint with vLLM or LLaMA-Factory's OpenAI-style API,
then configure this project:

```bash
export SEBENCH_STUDENT_PROVIDER=openai_compatible
export SEBENCH_STUDENT_OPENAI_BASE_URL=http://localhost:8000/v1
export SEBENCH_STUDENT_OPENAI_API_KEY=dummy
export SEBENCH_STUDENT_MODEL=sebench-student-lora
```

6. Generate SWE-bench-style predictions:

```bash
sebench swe-predict \
  --dataset artifacts/public_swe/dataset.json \
  --role student \
  --out artifacts/public_swe/predictions.jsonl
```

7. Run or stage official SWE-bench harness evaluation:

```bash
sebench run-swe-harness \
  --predictions artifacts/public_swe/predictions.jsonl \
  --dataset-name SWE-bench/SWE-bench_Lite \
  --split test \
  --out reports/swebench_harness_report.json
```

8. Compare configured roles on a held-out benchmark:

```bash
sebench evaluate-models \
  --dataset examples/git_pytest_benchmark.json \
  --roles base,student,teacher \
  --out-dir reports/model_evals
```

## Current Evidence Slots

Fill this section only with measured local or cloud runs.

| Run | Dataset | Model Role | Model ID | GPU | Pass Rate | Valid Task Rate | Latency Avg | Cost Est. |
|---|---|---|---|---|---:|---:|---:|---:|
| pending | pending | student | pending | pending | pending | pending | pending | pending |

## Guardrails

- Do not claim that GPT-5.5 parameters were trained.
- Do not include friend/company internal data, tasks, prompts, metrics, or names.
- Keep every dataset version tied to prompt version, teacher model ID, seed, split,
  and validation status.
- Treat generated hidden judges as drafts until deterministic local validation
  confirms they are executable and do not leak answers through the prompt.
- Keep gold SWE patches out of evaluation prompts; use them only for offline
  training exports with source and license metadata.
