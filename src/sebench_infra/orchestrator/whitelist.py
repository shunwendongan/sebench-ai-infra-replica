from pathlib import PurePosixPath


class PathWhitelist:
    """Validate that produced artifacts stay inside allowed relative prefixes."""

    def __init__(self, allowed_prefixes: list[str]) -> None:
        self.allowed_prefixes = [self._normalize(prefix) for prefix in allowed_prefixes]

    def is_allowed(self, path: str) -> bool:
        normalized = self._normalize(path)
        if normalized.startswith("../") or normalized == "..":
            return False
        return any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in self.allowed_prefixes
        )

    def filter(self, paths: list[str]) -> list[str]:
        return [path for path in paths if self.is_allowed(path)]

    @staticmethod
    def _normalize(path: str) -> str:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            return "../blocked"
        normalized = str(pure)
        if path.endswith("/") and not normalized.endswith("/"):
            normalized += "/"
        return normalized
