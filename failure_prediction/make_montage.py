#!/usr/bin/env python3
"""Render what a condition actually showed the model, as one labelled strip per record.

The numbers in the result tables say which observation setting won; they do not show what the model
was looking at. This writes that out: one PNG per record, frames left to right in time order, each
labelled with its offset from the lead point and its absolute timestep, and a header carrying the
truth plus what the recorded runs answered.

Labels are ASCII on purpose -- PIL's default font has no Korean glyphs and renders them as boxes.

    python make_montage.py --setting fixed3 --faults close_too_high --leads 1
    python make_montage.py --setting add_t20 --faults close_too_high,gripper_closed_early --per-fault 2
    python make_montage.py --setting fixed3 --only-disagree      # current_only 와 답이 갈린 것만
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from adaptive_probe import OFFSETS, obs_files


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return {r["image_file"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def strip(images_dir: Path, m: dict, setting: str, header: str) -> Image.Image | None:
    f = {"timestep": m["timestep"], "chunk_timesteps": m["chunk_timesteps"],
         "chunk_files": m["chunk_files"]}
    files, offs = obs_files(f, setting)
    if files is None:
        return None
    ims = [Image.open(images_dir / x).convert("RGB") for x in files]
    w, h = ims[0].size
    pad, hdr, bar = 8, 32, 22
    body = Image.new("RGB", (len(ims) * w + (len(ims) + 1) * pad, h + 2 * pad + hdr), (255, 255, 255))
    d = ImageDraw.Draw(body)
    for i, (im, o) in enumerate(zip(ims, offs)):
        x = pad + i * (w + pad)
        body.paste(im, (x, pad + hdr))
        d.text((x + 3, pad + 2), f"FRAME {i + 1}", fill=(0, 0, 0))
        d.text((x + 3, pad + 16), f"offset {o:+d}   t={m['timestep'] + o}", fill=(90, 90, 90))
    top = Image.new("RGB", (body.width, bar), (30, 30, 30))
    ImageDraw.Draw(top).text((4, 5), header[: body.width // 6], fill=(255, 255, 120))
    out = Image.new("RGB", (body.width, body.height + bar), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(body, (0, bar))
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default=str(here / "images_leadtime_adaptive"))
    ap.add_argument("--out-dir", default=str(here / "montage"))
    ap.add_argument("--setting", default="fixed3", choices=sorted(OFFSETS))
    ap.add_argument("--faults", default=None, help="comma-separated failure_type (nominal 포함 가능)")
    ap.add_argument("--leads", default=None)
    ap.add_argument("--per-fault", type=int, default=1, help="결함·lead 조합마다 몇 개까지")
    ap.add_argument("--only-disagree", action="store_true",
                    help="이 설정과 current_only 의 답이 갈린 기록만")
    ap.add_argument("--oracle", default="results/adaptive_oracle_G1.jsonl")
    ap.add_argument("--fixed", default="results/adaptive_fixed_ARMC.jsonl")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    meta = {json.loads(l)["image_file"]: json.loads(l)
            for l in (images_dir / "metadata.jsonl").read_text().splitlines() if l.strip()}
    O, F = load(here / args.oracle), load(here / args.fixed)

    want_f = {x.strip() for x in args.faults.split(",")} if args.faults else None
    want_l = {int(x) for x in args.leads.split(",")} if args.leads else None
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def answer(fn: str) -> tuple[str, str]:
        if args.setting == "fixed3":
            a = F.get(fn, {}).get("ans_fixed3", "-")
        else:
            a = O.get(fn, {}).get(f"ans_{args.setting}", "-")
        return a, O.get(fn, {}).get("ans_current_only", "-")

    made = 0
    seen: dict[tuple, int] = {}
    for fn, m in sorted(meta.items()):
        if want_f and m["failure_type"] not in want_f:
            continue
        if want_l and m["lead_time"] not in want_l:
            continue
        a, c1 = answer(fn)
        if args.only_disagree and (a == c1 or "-" in (a, c1)):
            continue
        key = (m["failure_type"], m["lead_time"])
        if seen.get(key, 0) >= args.per_fault:
            continue
        mark = "OK" if a == m["label"] else ("XX" if a in ("SUCCESS", "FAILURE") else "--")
        header = (f"[{mark}] {m['failure_type']} | lead t-{m['lead_time']} | truth {m['label']} | "
                  f"{args.setting} -> {a} | current_only -> {c1}")
        im = strip(images_dir, m, args.setting, header)
        if im is None:
            print(f"  skip {fn}: {args.setting} 프레임이 에피소드 시작에 걸려 없음")
            continue
        p = out / f"{args.setting}_{mark}_{fn}"
        im.save(p)
        seen[key] = seen.get(key, 0) + 1
        made += 1
        print(f"  {mark}  {m['failure_type']:22s} t-{m['lead_time']:<3d} -> {p.name}")
    print(f"\n{made}개 저장 -> {out}")


if __name__ == "__main__":
    main()
