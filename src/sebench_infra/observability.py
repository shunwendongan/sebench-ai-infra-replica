import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Histogram

RUN_COUNTER = Counter("sebench_runs_total", "Total benchmark runs", ["status"])
RUN_LATENCY = Histogram("sebench_run_latency_seconds", "Benchmark run latency")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    logging.getLogger("sebench").info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


@contextmanager
def observe_run() -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    except Exception:
        RUN_COUNTER.labels(status="failed").inc()
        raise
    else:
        RUN_COUNTER.labels(status="ok").inc()
    finally:
        RUN_LATENCY.observe(time.perf_counter() - start)
