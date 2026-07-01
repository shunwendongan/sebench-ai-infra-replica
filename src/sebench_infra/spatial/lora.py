from dataclasses import dataclass


@dataclass(frozen=True)
class LoRARecipe:
    """PEFT-compatible LoRA recipe placeholder for public reproduction runs."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    max_seq_length: int = 4096

    def to_peft_kwargs(self) -> dict:
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
