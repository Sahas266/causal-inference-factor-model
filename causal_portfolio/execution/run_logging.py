"""Per-run text logging shared by every execution entrypoint."""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from causal_portfolio.execution.audit import LOG_DIR


_ACTIVE_LOG: ContextVar[Path | None] = ContextVar("cpcm_execution_log", default=None)


@contextmanager
def execution_run_log(name: str, log_dir: Path | None = None) -> Iterator[Path]:
    """Capture all model and execution logs for one run in a unique file."""
    run_logger = logging.getLogger("cpcm.execution.run")
    active_path = _ACTIVE_LOG.get()
    if active_path is not None:
        try:
            yield active_path
        except BaseException:
            run_logger.exception("execution run failed: %s", name)
            raise
        return

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if not safe_name:
        raise ValueError("execution log name must contain a letter or number")
    directory = log_dir or LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"execution-{safe_name}-{stamp}-{os.getpid()}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)

    # ponytail: process-wide handler assumes serialized outer runs; add
    # serialization or context-aware filtering before allowing concurrency.
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(min(previous_level, logging.INFO))
    root.addHandler(handler)
    token = _ACTIVE_LOG.set(path)
    try:
        run_logger.info("execution run started: %s log=%s", name, path)
        yield path
    except BaseException:
        run_logger.exception("execution run failed: %s", name)
        raise
    else:
        run_logger.info("execution run completed: %s", name)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()
        _ACTIVE_LOG.reset(token)
