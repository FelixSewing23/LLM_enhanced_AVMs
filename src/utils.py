"""Shared utilities: logging, JSON/JSONL I/O, path helpers, timing."""

import functools
import json
import logging
import time
from pathlib import Path


# ── Logger factory ─────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Directory helpers ──────────────────────────────────────────────────────────
def ensure_dirs(*paths: Path) -> None:
    """Create directories (and parents) if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ── JSON helpers ───────────────────────────────────────────────────────────────
def save_json(data, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Timing decorator ───────────────────────────────────────────────────────────
def timer(func):
    """Log wall-clock time for any function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} finished in {time.time() - t0:.1f}s")
        return result
    return wrapper
