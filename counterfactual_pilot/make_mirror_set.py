#!/usr/bin/env python3
"""Build a horizontally mirrored copy of a frame set, to test whether left/right is grounded.

On the r2 frames Gemma answered MOVE_LEFT on 10/10 calls for `lateral_offset_y_pos` but MOVE_RIGHT
on only 3/10 for `lateral_offset_y_neg`. A model that simply leans toward LEFT produces exactly that
pattern, and so does a model that genuinely reads the offset — the two are not separable from those
numbers alone.

Mirroring separates them. Flipping the image swaps which side of the cube the gripper is on, so:

    grounded  -> the answer flips  (MOVE_LEFT <-> MOVE_RIGHT)
    biased    -> the answer stays  (MOVE_LEFT either way)

Vertical and gripper failures are the internal control: their correction has no left/right component,
so mirroring should leave them alone. If MOVE_DOWN also starts moving around, the mirrored images are
simply harder to read and the lateral result means nothing either.

The mirrored frames do not correspond to any real simulator state, so no oracle can be run on them.
This is a consistency probe, not a second oracle.

    python make_mirror_set.py
    python gemma_action_choice.py --task action --robustness --images-dir images_r2_mirror \
        --run-id R2_mirror
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

FLIP = {"MOVE_LEFT": "MOVE_RIGHT", "MOVE_RIGHT": "MOVE_LEFT"}


def mirror_label(label: str | None) -> str | None:
    return FLIP.get(label, label)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(here / "images_r2"))
    ap.add_argument("--dst", default=str(here / "images_r2_mirror"))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (src / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    out = []
    for r in rows:
        img = Image.open(src / r["image_file"]).transpose(Image.FLIP_LEFT_RIGHT)
        name = r["image_file"].replace(".png", "_mirror.png")
        img.save(dst / name)

        m = dict(r)
        m["image_file"] = name
        m["image_path"] = str(dst / name)
        m["mirrored"] = True
        m["source_image"] = r["image_file"]
        # what a grounded answer should become: the left/right component swaps, nothing else does
        m["reference_correction"] = mirror_label(r["reference_correction"])
        m["original_action_discrete"] = mirror_label(r["original_action_discrete"])
        a = list(r["original_action"])
        a[1] = -a[1]  # world y is image-right; mirroring negates it
        m["original_action"] = a
        out.append(m)

    with (dst / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in out:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    flipped = sum(1 for r, m in zip(rows, out) if r["reference_correction"] != m["reference_correction"])
    print(f"{len(out)} mirrored frames -> {dst}")
    print(f"reference_correction swapped on {flipped} of them (the lateral ones); "
          f"{len(out) - flipped} unchanged (vertical / gripper controls)")


if __name__ == "__main__":
    main()
