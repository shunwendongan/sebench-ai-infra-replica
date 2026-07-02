from __future__ import annotations

import json
from pathlib import Path

from sebench_infra.training_loop.models import (
    DatasetVersion,
    LlamaFactoryExportManifest,
    TrainingSplit,
)


class LlamaFactoryExporter:
    """Export accepted SFT examples and a conservative QLoRA recipe."""

    def export(
        self,
        dataset: DatasetVersion,
        out_dir: Path,
        *,
        dataset_name: str = "sebench_student_sft",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        template: str = "qwen",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        epochs: float = 3.0,
        cutoff_len: int = 4096,
        seed: int | None = None,
    ) -> LlamaFactoryExportManifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = dataset.seed if seed is None else seed

        train_file = out_dir / f"{dataset_name}_train.json"
        validation_file = out_dir / f"{dataset_name}_validation.json"
        test_file = out_dir / f"{dataset_name}_test.json"
        dataset_info_file = out_dir / "dataset_info.json"
        train_config_file = out_dir / "train_qlora.yaml"
        windows_script_file = out_dir / "train_windows_cuda.ps1"

        split_payloads = {
            TrainingSplit.TRAIN: self._examples_for_split(dataset, TrainingSplit.TRAIN),
            TrainingSplit.VALIDATION: self._examples_for_split(dataset, TrainingSplit.VALIDATION),
            TrainingSplit.TEST: self._examples_for_split(dataset, TrainingSplit.TEST),
        }
        self._write_json(train_file, split_payloads[TrainingSplit.TRAIN])
        self._write_json(validation_file, split_payloads[TrainingSplit.VALIDATION])
        self._write_json(test_file, split_payloads[TrainingSplit.TEST])
        self._write_json(
            dataset_info_file,
            {
                f"{dataset_name}_train": {
                    "file_name": train_file.name,
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                },
                f"{dataset_name}_validation": {
                    "file_name": validation_file.name,
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                },
                f"{dataset_name}_test": {
                    "file_name": test_file.name,
                    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
                },
            },
        )
        train_config_file.write_text(
            self._training_yaml(
                dataset_name=f"{dataset_name}_train",
                base_model=base_model,
                template=template,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                cutoff_len=cutoff_len,
                seed=seed,
            ),
            encoding="utf-8",
        )
        windows_script_file.write_text(_WINDOWS_SCRIPT, encoding="utf-8")

        manifest = LlamaFactoryExportManifest(
            dataset_version_id=dataset.dataset_version_id,
            dataset_name=dataset_name,
            output_dir=str(out_dir),
            train_file=str(train_file),
            validation_file=str(validation_file),
            test_file=str(test_file),
            dataset_info_file=str(dataset_info_file),
            train_config_file=str(train_config_file),
            windows_script_file=str(windows_script_file),
            train_examples=len(split_payloads[TrainingSplit.TRAIN]),
            validation_examples=len(split_payloads[TrainingSplit.VALIDATION]),
            test_examples=len(split_payloads[TrainingSplit.TEST]),
            base_model=base_model,
            metadata={
                "template": template,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "cutoff_len": cutoff_len,
                "seed": seed,
                "training_task_counts": self._training_task_counts(dataset),
            },
        )
        (out_dir / "export_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def _examples_for_split(
        self,
        dataset: DatasetVersion,
        split: TrainingSplit,
    ) -> list[dict[str, str]]:
        return [
            {
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
            }
            for example in dataset.examples
            if example.is_trainable and example.split == split
        ]

    def _training_task_counts(self, dataset: DatasetVersion) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in dataset.examples:
            if not example.is_trainable:
                continue
            key = example.training_task.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _training_yaml(
        self,
        *,
        dataset_name: str,
        base_model: str,
        template: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        learning_rate: float,
        epochs: float,
        cutoff_len: int,
        seed: int,
    ) -> str:
        return f"""model_name_or_path: {base_model}
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_dropout: {lora_dropout}
dataset: {dataset_name}
dataset_dir: .
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 8
output_dir: saves/sebench_student_lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
seed: {seed}
report_to: none
"""


_WINDOWS_SCRIPT = """$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command llamafactory-cli -ErrorAction SilentlyContinue)) {
  Write-Error "llamafactory-cli was not found. Install LLaMA-Factory in this environment first."
}

llamafactory-cli train .\\train_qlora.yaml
"""
