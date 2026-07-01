import json
from pathlib import Path

import numpy as np

from sebench_infra.spatial.geometry import GeometryBridge


class SpatialDiagnosisEngine:
    """Diagnose whether geometry facts are available before LLM reasoning."""

    def __init__(self, bridge: GeometryBridge | None = None) -> None:
        self.bridge = bridge or GeometryBridge()

    def diagnose_scene_file(self, path: Path | str) -> dict:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        source = np.asarray(payload["source_points"], dtype=float)
        target = np.asarray(payload["target_points"], dtype=float)
        facts = self.bridge.compute(source, target)
        return {
            "scene_id": payload.get("scene_id", "synthetic-scene"),
            "facts": facts.__dict__,
            "llm_prefix": facts.to_prefix(),
            "diagnosis": {
                "geometry_bridge_ready": True,
                "requires_real_multimodal_model": False,
                "note": "Synthetic fixtures emulate ScanNet-style object transformations.",
            },
        }
