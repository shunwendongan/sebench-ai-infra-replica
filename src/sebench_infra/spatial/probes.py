from dataclasses import dataclass, field


@dataclass
class AttentionProbeRecord:
    layer: int
    token: str
    score: float


@dataclass
class AttentionProbe:
    """A lightweight interface for future model attention hook integration."""

    records: list[AttentionProbeRecord] = field(default_factory=list)

    def add(self, layer: int, token: str, score: float) -> None:
        self.records.append(AttentionProbeRecord(layer=layer, token=token, score=score))

    def summarize(self) -> dict:
        if not self.records:
            return {"available": False, "max_score": None, "layers": []}
        return {
            "available": True,
            "max_score": max(record.score for record in self.records),
            "layers": sorted({record.layer for record in self.records}),
        }
