"""Read-only bridge to 중범's repo: import `fasi.*` and keep its relative paths working.

Nothing under ~/바탕화면/jungbeom is written to. `Pi05Steer` resolves `scripts/eval_vla_pi05.py`,
`outputs/vla/paligemma_tokenizer/hf` and `LIBERO_STATS` relative to the process cwd (pi05_steer.py:16,20,28),
so the cwd has to be that repo root while the policy is built.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

JB = Path.home() / "바탕화면" / "jungbeom"
HERE = Path(__file__).resolve().parent


def enter() -> Path:
    """chdir into the jungbeom repo and put it on sys.path. Returns the previous cwd."""
    if not JB.is_dir():
        raise SystemExit(f"jungbeom repo not found: {JB}")
    prev = Path.cwd()
    if str(JB) not in sys.path:
        sys.path.insert(0, str(JB))
    os.chdir(JB)
    return prev
