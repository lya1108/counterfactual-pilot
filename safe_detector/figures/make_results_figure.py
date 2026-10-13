"""Results figure for the slides: prefix-8 failure prediction from frozen pi0.5 latents.

Three panels, left to right:
  1. Data collection   frozen fine-tuned pi0.5 drives LIBERO; a read-only tap keeps one latent z_t per VLA call
  2. Prefix-8 setup    full rollouts differ in length (successes stop when solved, failures run to the timeout), so
                       every episode is cut to its first 8 calls before a separately trained detector reads it
  3. Main results      episode AUROC, time-only baseline vs. VLA latent, unseen rollouts and unseen tasks

Every number is read from results/ at build time and checked against the values quoted on the slide, so the
figure cannot silently drift from the run it describes:
  scores_lstm_concat_{init,task}_prefix8.json   auroc_episode   0.823 / 0.790
  scores_time_{init,task}_prefix8.json          auroc_episode   0.500 / 0.500
  data/*/episodes.jsonl                         800 episodes, 330 success / 470 failure, 4 suites

Rollout frames are read (never written) from ~/바탕화면/jungbeom/docs/libero_report/img/libero_spatial_t9_S.jpg,
a 10-frame strip sampled every 20 env steps; frames 0-3 are env steps 0, 20, 40, 60, i.e. the first step of calls
t1, t3, t5, t7, so each frame sits above the pair of calls it starts.

    python safe_detector/figures/make_results_figure.py     # -> results_prefix8.svg, results_prefix8.png (3840x2160)
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRIP = Path.home() / "바탕화면" / "jungbeom" / "docs" / "libero_report" / "img" / "libero_spatial_t9_S.jpg"
INSTRUCTION = "“pick up the black bowl … and place it on the plate”"
K = 8  # prefix length in VLA calls
EXEC = 10  # env steps executed per VLA call

W, H = 1920, 1080
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"
SERIF = "'Times New Roman', 'Liberation Serif', Times, serif"
INK, GREY, LIGHT = "#1A1A1A", "#555555", "#BDBDBD"
BLUE, BLUE_BG, BLUE_CHIP, BLUE_INK = "#2F5D9B", "#EEF3FA", "#DCE6F4", "#1F3F6E"
ORANGE, ORANGE_BG = "#C8701C", "#FDF1E4"
ENV, ENV_BG = "#8C8C8C", "#F6F6F6"
GREEN, GREEN_BG = "#2E7D32", "#E8F3E9"
RED, RED_BG = "#C62828", "#FBEAEA"
BASE = "#8C8C8C"  # time-only bar: deliberately grey (a reference, not a competing series)

out: list[str] = []


# ==================================================================== numbers, read and checked
def load_numbers() -> dict:
    res = ROOT / "results"
    ep = lambda tag: json.loads((res / f"scores_{tag}.json").read_text())
    n = {}
    for split in ("init", "task"):
        lat, tim = ep(f"lstm_concat_{split}_prefix8"), ep(f"time_{split}_prefix8")
        assert lat["args"]["max_calls"] == K and tim["args"]["max_calls"] == K
        n[split] = dict(latent=lat["auroc_episode"], time=tim["auroc_episode"], n_test=len(lat["episodes"]),
                        n_fail=sum(not e["success"] for e in lat["episodes"]), groups=len(lat["test_groups"]))
    rows = [json.loads(l) for p in sorted((ROOT / "data").glob("*/episodes.jsonl")) for l in p.read_text().splitlines()]
    n["episodes"], n["success"] = len(rows), sum(r["success"] for r in rows)
    n["suites"] = len({r["suite"] for r in rows})
    n["kept"] = sum(r["n_calls"] >= K for r in rows)
    assert {r["exec"] for r in rows} == {EXEC}
    # the values the slide text quotes; fail loudly if the results on disk say otherwise
    assert round(n["init"]["latent"], 3) == 0.823 and round(n["task"]["latent"], 3) == 0.790
    assert round(n["init"]["time"], 3) == 0.500 and round(n["task"]["time"], 3) == 0.500
    assert (n["episodes"], n["success"], n["suites"]) == (800, 330, 4)
    return n


# ==================================================================== svg primitives
def add(s: str):
    out.append(s)


def text(x, y, s, size=22, anchor="start", weight="normal", fill=INK, style="normal", family=SANS, extra=""):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


def _w(s: str, size: float, italic: bool) -> float:
    """Rough advance width for Times-like glyphs; good enough to place subscripts."""
    w = 0.0
    for c in s:
        if c.isupper():
            w += 0.62 if italic else 0.70
        elif c in "il:t":
            w += 0.28
        elif c == "z":
            w += 0.39
        else:
            w += 0.5
    return w * size


def math(x, y, base, sub=None, size=28, anchor="start", fill=INK):
    """base_{sub}, base italic, digit subscripts upright (as LaTeX sets them)."""
    s2 = size * 0.68
    wb = _w(base, size, True) + 0.02 * size
    total = wb + _w(sub or "", s2, True)
    x0 = x - total if anchor == "end" else x - total / 2 if anchor == "middle" else x
    parts = [f'<tspan font-style="italic">{base}</tspan>']
    if sub:
        st = "normal" if sub[0].isdigit() else "italic"
        parts.append(f'<tspan x="{x0 + wb:.1f}" dy="{0.30 * size:.1f}" font-size="{s2:.1f}" font-style="{st}">{sub}</tspan>')
    add(f'<text x="{x0:.1f}" y="{y}" font-family="{SERIF}" font-size="{size}" fill="{fill}">{"".join(parts)}</text>')


def rect(x, y, w, h, fill, stroke, sw=2.5, rx=14, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, color=INK, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')


def arrow(pts, color=INK, sw=3, dash=None, head="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{a},{b}" for a, b in pts)
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{d}'
        f' marker-end="url(#{head})"/>')


def tag(x, y, label, color):
    w = 22 + len(label) * 14
    add(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="17" fill="#FFFFFF" stroke="{color}" stroke-width="2"/>')
    text(x + w / 2, y + 24, label, size=18, anchor="middle", weight="bold", fill=color, extra='letter-spacing="1.5"')


def pill(cx, cy, label, color, bg, w=132, h=38, size=19):
    rect(cx - w / 2, cy - h / 2, w, h, bg, color, sw=2, rx=h / 2)
    text(cx, cy + size * 0.36, label, size=size, anchor="middle", weight="bold", fill=color, extra='letter-spacing="1"')


def chip(cx, cy, w, h, sub, size=24):
    rect(cx - w / 2, cy - h / 2, w, h, BLUE_CHIP, BLUE, sw=2, rx=7)
    math(cx, cy + size * 0.22, "z", sub=sub, size=size, anchor="middle", fill=BLUE_INK)


def heading(x, y, num, title):
    add(f'<circle cx="{x + 18}" cy="{y - 11}" r="18" fill="{INK}"/>')
    text(x + 18, y - 3, num, size=22, anchor="middle", weight="bold", fill="#FFFFFF")
    text(x + 48, y, title, size=32, weight="bold")


def frame_uris(idx) -> list[str]:
    strip = Image.open(STRIP).convert("RGB")
    side = strip.height
    assert strip.width // side == 10
    uris = []
    for i in idx:
        buf = io.BytesIO()
        strip.crop((i * side, 0, (i + 1) * side, side)).save(buf, "JPEG", quality=92)
        uris.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return uris


# ==================================================================== panels
def panel_collection(n):
    heading(40, 80, "1", "Data Collection")
    L, R = 76, 596  # box span

    # observation + instruction
    rect(L, 112, R - L, 92, ENV_BG, ENV, sw=2.5)
    text(L + 24, 150, "Observation + Task Instruction", size=24, weight="bold")
    text(L + 24, 184, INSTRUCTION, size=20, fill=GREY, style="italic")
    arrow([(336, 204), (336, 240)], sw=3)

    # frozen policy
    rect(L, 244, R - L, 88, BLUE_BG, BLUE, sw=3)
    text(L + 24, 299, "Fine-tuned π0.5", size=32, weight="bold")
    tag(R - 140, 271, "FROZEN", BLUE)
    arrow([(336, 332), (336, 370)], sw=3)
    text(352, 360, "Robot action", size=22)

    # rollout
    top, bot = 374, 760
    rect(L, top, R - L, bot - top, "#FFFFFF", ENV, sw=2.5)
    text(L + 20, top + 38, "LIBERO Rollout", size=26, weight="bold")
    text(R - 20, top + 38, f"1 VLA call = {EXEC} env steps", size=18, anchor="end", fill=GREY)
    fw, gap, fx0, fy = 110, 13.3, L + 20, top + 58
    for k, uri in enumerate(frame_uris((0, 1, 2, 3))):
        x = fx0 + k * (fw + gap)
        add(f'<image x="{x:.1f}" y="{fy}" width="{fw}" height="{fw}" href="{uri}" preserveAspectRatio="xMidYMid slice"/>')
        add(f'<rect x="{x:.1f}" y="{fy}" width="{fw}" height="{fw}" fill="none" stroke="{ENV}" stroke-width="1.5"/>')
    # t_i -> z_i, two calls under each frame
    ty, cy = fy + fw + 34, fy + fw + 92
    for i in range(K):
        k, half = divmod(i, 2)
        cx = fx0 + k * (fw + gap) + (28 if half == 0 else fw - 28)
        math(cx, ty, "t", sub=str(i + 1), size=24, anchor="middle")
        arrow([(cx, ty + 9), (cx, cy - 24)], color=BLUE, sw=2.5, head="ahb")
        chip(cx, cy, 46, 40, str(i + 1), size=22)
    add(f'<text x="{L + 20}" y="{cy + 52}" font-family="{SANS}" font-size="19" fill="{BLUE_INK}">'
        f'<tspan font-family="{SERIF}" font-style="italic" font-size="22">z</tspan><tspan font-family="{SERIF}" '
        f'font-style="italic" font-size="15" dy="6">t</tspan><tspan dy="-6"> : π0.5 hidden state at VLA call </tspan>'
        f'<tspan font-family="{SERIF}" font-style="italic" font-size="22">t</tspan><tspan> (read-only)</tspan></text>')
    # outcome
    oy = bot - 40
    text(L + 20, oy + 7, "Episode ends in", size=21, fill=INK)
    pill(L + 270, oy, "SUCCESS", GREEN, GREEN_BG)
    text(L + 352, oy + 7, "or", size=21, fill=GREY, anchor="middle")
    pill(L + 436, oy, "FAILURE", RED, RED_BG)

    # read-only latent tap: from pi0.5, down the right margin, into the latent row
    add(f'<circle cx="{R}" cy="288" r="6" fill="{BLUE}"/>')
    arrow([(R, 288), (614, 288), (614, cy), (R - 4, cy)], color=BLUE, sw=2.5, dash="8 6", head="ahb")
    # control loop: environment feeds the next observation back
    arrow([(L, 420), (56, 420), (56, 158), (L - 2, 158)], color=ENV, sw=2.5, head="ahg")

    # dataset
    sy = 838
    text(L, sy - 46, "Dataset", size=22, weight="bold", fill=GREY)
    for cx, big, lab in ((L + 70, f"{n['episodes']}", "episodes"),
                         (L + 262, f"{n['success']} / {n['episodes'] - n['success']}", "success / failure"),
                         (L + 452, f"{n['suites']}", "LIBERO suites")):
        text(cx, sy, big, size=34, anchor="middle", weight="bold")
        text(cx, sy + 30, lab, size=19, anchor="middle", fill=GREY)
    text(L, sy + 82, f"Prefix-8 analysis: {n['kept']} episodes with ≥ {K} calls", size=18, fill=GREY)
    text(L, sy + 106, f"({n['episodes'] - n['kept']} successes shorter than {K} calls excluded)", size=18, fill=GREY)


def panel_prefix(n):
    heading(650, 80, "2", "Prefix-8 Failure Prediction")
    lab_x = 670
    step = 34
    X = lambda c: 806 + (c - 1) * step  # x of VLA call c
    cut = X(K) + step / 2 + 3
    r_s, r_f = 222, 304  # row centres
    band_top, band_bot = 168, 346

    # used / not-used regions
    rect(X(1) - 22, band_top, cut - X(1) + 22, band_bot - band_top, BLUE_BG, BLUE, sw=2.5, rx=10, dash="9 6")
    text((X(1) - 22 + cut) / 2, band_top - 14, "Used", size=22, anchor="middle", weight="bold", fill=BLUE)
    text((cut + 1268) / 2, band_top - 14, "Not used", size=22, anchor="middle", weight="bold", fill=ENV)
    line(cut, band_top - 4, cut, band_bot + 4, color=BLUE, sw=3)

    for y, label, color, end_c, end_lab, broken in ((r_s, "SUCCESS", GREEN, 10, "Success", False),
                                                    (r_f, "FAILURE", RED, None, "Timeout", True)):
        text(lab_x, y + 7, label, size=20, weight="bold", fill=color)
        for c in range(1, K + 1):
            chip(X(c), y, 30, 38, str(c), size=20)
        # tail beyond the prefix: grey, to the episode's own end
        tail = [X(K + 1), X(K + 2)] if broken else [X(c) for c in range(K + 1, end_c + 1)]
        line(cut + 2, y, tail[-1], y, color=LIGHT, sw=3)
        for x in tail:
            add(f'<circle cx="{x}" cy="{y}" r="6" fill="{LIGHT}"/>')
        if broken:  # ... // ... t_T: failures always run to the suite timeout
            bx = X(K + 2) + 22
            add(f'<path d="M {bx - 4} {y + 12} L {bx + 4} {y - 12} M {bx + 6} {y + 12} L {bx + 14} {y - 12}" '
                f'stroke="{ENV}" stroke-width="2.5" fill="none"/>')
            ex = bx + 40
            line(bx + 18, y, ex, y, color=LIGHT, sw=3)
            add(f'<circle cx="{ex}" cy="{y}" r="8" fill="{RED}"/>')
            math(ex, y - 18, "t", sub="T", size=22, anchor="middle", fill=GREY)
        else:
            ex = tail[-1]
            add(f'<circle cx="{ex}" cy="{y}" r="8" fill="{GREEN}"/>')
        text(ex + 16, y + 7, end_lab, size=20, weight="bold", fill=color)

    # what the cut buys
    text(lab_x, band_bot + 44, "Full rollouts differ in length → length alone would separate outcomes",
         size=18, fill=GREY, style="italic")
    rect(lab_x, band_bot + 70, 600, 76, BLUE_BG, BLUE, sw=0, rx=10)
    text(970, band_bot + 102, "Same observation horizon", size=24, anchor="middle", weight="bold", fill=BLUE_INK)
    text(970, band_bot + 132, f"first {K} VLA calls ({K * EXEC} env steps) for every episode", size=20,
         anchor="middle", fill=BLUE_INK)

    # separate detector on the prefix latents
    y0 = band_bot + 176
    arrow([(905, y0 + 4), (905, y0 + 72)], color=BLUE, sw=3, head="ahb")
    add(f'<text x="925" y="{y0 + 42}" font-family="{SERIF}" font-size="26" fill="{BLUE_INK}">'
        f'<tspan font-style="italic">z</tspan><tspan font-size="18" dy="7">1</tspan><tspan dy="-7"> … </tspan>'
        f'<tspan font-style="italic">z</tspan><tspan font-size="18" dy="7">{K}</tspan></text>')
    dt = y0 + 76
    rect(780, dt, 250, 104, ORANGE_BG, ORANGE, sw=3)
    text(905, dt + 46, "Failure Detector", size=28, anchor="middle", weight="bold")
    text(905, dt + 80, "separate model (LSTM)", size=19, anchor="middle", fill=GREY)
    tag(1046, dt + 35, "TRAINED", ORANGE)
    arrow([(905, dt + 104), (905, dt + 168)], color=ORANGE, sw=3, head="aho")
    text(905, dt + 202, "Failure Probability", size=26, anchor="middle", weight="bold")
    text(905, dt + 232, "of the eventual episode outcome", size=19, anchor="middle", fill=GREY)
    # frozen vs trained, said once in words
    text(970, 948, "π0.5 stays frozen: only the detector is trained", size=19, anchor="middle", fill=GREY,
         style="italic")


def panel_results(n):
    heading(1330, 80, "3", "Main Results")
    lo, hi = 0.4, 0.9  # axis floor below chance so the 0.500 baseline bar stays visible
    y_lo, y_hi = 720, 290
    Y = lambda v: y_lo - (v - lo) / (hi - lo) * (y_lo - y_hi)
    ax = 1418
    right = 1880
    # grid + ticks
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        line(ax, Y(v), right, Y(v), color="#E6E6E6", sw=1.5)
        text(ax - 10, Y(v) + 7, f"{v:.1f}", size=20, anchor="end", fill=GREY)
    line(ax, y_lo, right, y_lo, color=INK, sw=2)
    line(ax, y_lo, ax, y_hi - 6, color=INK, sw=2)
    yc = (y_lo + y_hi) / 2
    text(1352, yc, "Episode AUROC", size=22, anchor="middle", extra=f'transform="rotate(-90 1352 {yc})"')
    # chance
    line(ax, Y(0.5), right, Y(0.5), color=INK, sw=2, dash="8 6")

    bw = 74
    for gx, split, title, sub in ((1534, "init", "Unseen Rollout", "held-out initial states"),
                                  (1764, "task", "Unseen Task", f"{n['task']['groups']} held-out tasks")):
        d = n[split]
        text(gx, 214, title, size=24, anchor="middle", weight="bold")
        text(gx, 242, sub, size=18, anchor="middle", fill=GREY)
        for bx, v, color, lab in ((gx - 52, d["time"], BASE, "Time-only"), (gx + 52, d["latent"], BLUE, "VLA latent")):
            top = Y(v)
            add(f'<path d="M {bx - bw / 2} {y_lo} L {bx - bw / 2} {top + 4} Q {bx - bw / 2} {top} {bx - bw / 2 + 4} {top} '
                f'L {bx + bw / 2 - 4} {top} Q {bx + bw / 2} {top} {bx + bw / 2} {top + 4} L {bx + bw / 2} {y_lo} Z" '
                f'fill="{color}"/>')
            text(bx, top - 12, f"{v:.3f}", size=24, anchor="middle", weight="bold")
            text(bx, y_lo + 30, lab, size=19, anchor="middle")
        text(gx, y_lo + 60, f"n = {d['n_test']} test episodes", size=18, anchor="middle", fill=GREY)
    # chance legend, under the axis so it never sits on a bar
    line(ax + 4, y_lo + 96, ax + 48, y_lo + 96, color=INK, sw=2, dash="8 6")
    text(ax + 58, y_lo + 103, "Random level (AUROC = 0.5)", size=19, fill=GREY)

    text(1349, 880, "Latent > Time-only", size=28, weight="bold")
    text(1349, 914, "Failure-related information exists in", size=20, fill=GREY)
    text(1349, 940, "early VLA representations", size=20, fill=GREY)


def build(n) -> str:
    marker = lambda mid, c: (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.2" '
                             f'markerHeight="4.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Frozen fine-tuned pi0.5 rolls out LIBERO tasks while its latents are read at every VLA call; every '
        f'episode is cut to its first 8 calls and a separately trained detector predicts the eventual outcome. Episode '
        f'AUROC: time-only 0.500 in both settings, VLA latent {n["init"]["latent"]:.3f} on unseen rollouts and '
        f'{n["task"]["latent"]:.3f} on unseen tasks.">')
    add("<defs>" + marker("ah", INK) + marker("ahb", BLUE) + marker("aho", ORANGE) + marker("ahg", ENV) + "</defs>")
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    for x in (630, 1310):
        line(x, 44, x, 960, color="#D6D6D6", sw=2)

    panel_collection(n)
    panel_prefix(n)
    panel_results(n)

    line(40, 982, 1880, 982, color="#D6D6D6", sw=2)
    add(f'<text x="960" y="1036" font-family="{SANS}" font-size="30" fill="{INK}" text-anchor="middle">'
        f'Early VLA latent representations predict <tspan font-weight="bold">eventual rollout failure</tspan> '
        f'beyond episode-time information.</text>')
    add("</svg>")
    return "\n".join(out)


def main():
    n = load_numbers()
    svg = build(n)
    (HERE / "results_prefix8.svg").write_text(svg)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        page.write_text(f"<!doctype html><html><body style='margin:0;background:#fff'>{svg}</body></html>")
        subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={W},{H}", "--force-device-scale-factor=2",
                        f"--screenshot={HERE / 'results_prefix8.png'}", page.as_uri()],
                       check=True, capture_output=True)
    print(HERE / "results_prefix8.svg", HERE / "results_prefix8.png")


if __name__ == "__main__":
    main()
