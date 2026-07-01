from uuid import uuid4


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid4().hex[:12]}"
