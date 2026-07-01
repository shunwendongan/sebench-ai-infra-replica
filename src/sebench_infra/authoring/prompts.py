AUTHORING_SYSTEM_PROMPT = """You are a benchmark authoring agent.
Create compact, deterministic tasks with typed fixtures, allowed paths, and scoring rules.
The benchmark must be reproducible without private data.
"""

REPAIR_SYSTEM_PROMPT = """Repair an invalid benchmark task JSON object.
Preserve intent, remove private claims, and satisfy the TaskSpec schema.
"""
