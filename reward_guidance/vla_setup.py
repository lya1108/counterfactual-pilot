"""Build Pi05Steer against a checkpoint in OUR tree.

2026-10-12: `outputs/vla/pi05_libero_finetuned` and `outputs/vla/paligemma_tokenizer/hf` are no longer on this
machine -- the jungbeom repo keeps them git-excluded and they were lost with the previous disk. Both are
restored here instead of there, because nothing under ~/바탕화면/jungbeom is written to:

  ~/yuna/ckpt/pi05_libero_finetuned   re-downloaded from lerobot/pi05_libero_finetuned (7.47 GB)
  ~/yuna/ckpt/paligemma_tokenizer/hf  rebuilt by rebuild_tokenizer.py from the public sentencepiece model

`pi05_steer.LIBERO_STATS` is a module-level relative path read inside `Pi05Steer.__init__`, so it is rebound
on the module object at runtime. The file on disk is untouched.

The Hub repo has moved since this project's data was collected (the stale cache ref names revision 8e174154,
the Hub now serves 7152f13e), so verify_ckpt.py checks the restored weights against the stored rollouts
before anything is run on top of them.
"""
from __future__ import annotations

from pathlib import Path

CKPT = Path.home() / "yuna/ckpt/pi05_libero_finetuned"
TOKENIZER = Path.home() / "yuna/ckpt/paligemma_tokenizer/hf"
STATS = CKPT / "policy_preprocessor_step_2_normalizer_processor.safetensors"


def make_vla(batch: int = 4, ckpt: str | Path | None = None):
    """Pi05Steer on the restored checkpoint. Call only after jb.enter()."""
    from fasi.vla import pi05_steer

    for p in (CKPT, TOKENIZER, STATS):
        if not Path(p).exists():
            raise SystemExit(f"없음: {p}\n  체크포인트: huggingface_hub.snapshot_download"
                             f"('lerobot/pi05_libero_finetuned', local_dir='{CKPT}')\n"
                             f"  토크나이저: python reward_guidance/rebuild_tokenizer.py")
    pi05_steer.LIBERO_STATS = str(STATS)
    vla = pi05_steer.Pi05Steer(str(ckpt or CKPT), tokenizer=str(TOKENIZER), batch=batch)
    return vla
