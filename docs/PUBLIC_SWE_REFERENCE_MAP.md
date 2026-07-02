# Public SWE Ecosystem Reference Map

This project is a public SWE-bench-style evaluation and training loop. It does
not reproduce ByteDance internal systems, private SE-Bench protocols, private
datasets, or private metrics.

## Primary Public References

| Public reference | Project role | Local implementation |
|---|---|---|
| [SWE-bench](https://github.com/swe-bench/SWE-bench) | Real issue instances, Docker/harness evaluation, prediction file convention | `SWEIssueInstance`, `write_swe_predictions`, optional `SWEHarnessRunner` |
| [SWE-bench datasets](https://www.swebench.com/SWE-bench/guides/datasets/) | Public splits such as Lite and Verified | `import-swe-dataset --hf-dataset ...` |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Synthetic scalable SWE task generation | `ExternalBenchmarkSource` plus SWE-style import/export |
| [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) | Training environments and trajectories for SWE agents | `TrainingTaskKind.TRAJECTORY_ACTION` export path |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | Public successful trajectories for SFT-style training | LLaMA-Factory export metadata and future trajectory adapters |
| [OpenHands SFT trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | Public action trajectories for software engineering agents | `trajectory_action` training task type |

## Data Boundary

- Evaluation prompts may include issue text, repository name, base commit, and
  public metadata.
- Evaluation prompts must not include gold patches, hidden test patch contents,
  or oracle answers.
- Gold patches are allowed only in offline training examples or analysis artifacts
  with explicit source and license metadata.

## Local Modules

- `sebench_infra.training_loop.swe`: parses SWE-bench/Hugging Face-style rows,
  preserves provenance, and converts issues to internal `TaskSpec`.
- `sebench_infra.training_loop.patch_agent`: single-turn model-backed patch
  generation and local unified-diff validation.
- `sebench_infra.training_loop.swe_harness`: optional adapter for official
  SWE-bench harness commands and prediction JSONL generation.
- `sebench_infra.training_loop.export`: LLaMA-Factory SFT export for
  `task_spec_authoring`, `issue_to_patch`, and `trajectory_action` examples.

## Resume-Safe Claim

Use:

> Built a public SWE-bench-style long-horizon software engineering agent
> evaluation and training loop with typed task conversion, prediction export,
> Work/Judge-compatible scoring, and LLaMA-Factory student data export.

Avoid:

> Reproduced ByteDance internal SE-Bench, used internal protocols, or trained
> GPT-5.5 parameters.
