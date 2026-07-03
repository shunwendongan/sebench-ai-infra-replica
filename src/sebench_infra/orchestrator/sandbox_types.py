from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SandboxResult:
    artifacts: dict[str, str]
    metadata: dict[str, Any]
