"""Pipeline figure for the slides: frozen pi0.5 drives the LIBERO rollout while a separate detector reads its latents.

Structure follows the code, not a sketch:
  latent_hook.py   LatentTap wraps PaliGemmaWithExpertModel.forward (read-only) and keeps the prefix hidden states
                   (PaliGemma, 2048-d, image + text tokens) and the action-expert hidden states (1024-d, K=10 flow steps)
  collect_rollouts one latent per VLA call; one call = 50-step chunk, first 10 executed
  train_detector   linear probe or causal LSTM (hidden 128), sigmoid failure score, pi0.5 weights untouched
  eval_leadtime    threshold from a false-alarm budget on successful episodes; lead time = alert -> episode end

The probability curve is SCHEMATIC (labelled as such in the figure). Rollout frames are read (never written) from
~/바탕화면/jungbeom/docs/libero_report/img/libero_spatial_t4_F.jpg, a 10-frame strip of a failed LIBERO-Spatial episode.

    python safe_detector/figures/make_pipeline_figure.py     # -> pipeline.svg, pipeline.png (3840x2160)
"""
from __future__ import annotations

import base64
import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STRIP = Path.home() / "바탕화면" / "jungbeom" / "docs" / "libero_report" / "img" / "libero_spatial_t4_F.jpg"
FRAMES = (0, 2, 4, 9)  # indices into the 10-frame strip (20 env steps apart)

W, H = 1920, 1080
SANS = "Arial, 'Liberation Sans', Helvetica, sans-serif"
SERIF = "'Times New Roman', 'Liberation Serif', Times, serif"
INK, GREY = "#1A1A1A", "#555555"
BLUE, BLUE_BG, BLUE_CHIP, BLUE_INK = "#2F5D9B", "#EEF3FA", "#DCE6F4", "#1F3F6E"
ORANGE, ORANGE_BG = "#C8701C", "#FDF1E4"
ENV, ENV_BG = "#8C8C8C", "#F6F6F6"
RED = "#C62828"

out: list[str] = []


def add(s: str):
    out.append(s)


def text(x, y, s, size=22, anchor="start", weight="normal", fill=INK, style="normal", family=SANS, extra=""):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


def _w(s: str, size: float, italic: bool) -> float:
    """Rough advance width for Times-like glyphs; good enough to place sub/superscripts."""
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


def math(x, y, base, sub=None, sup=None, size=28, anchor="start", fill=INK):
    """base_{sub}^{sup}, sub and sup stacked; base and sub italic, sup upright (VLM / ACT are names)."""
    s2 = size * 0.68
    wb = _w(base, size, True) + 0.02 * size
    total = wb + max(_w(sub or "", s2, True), _w(sup or "", s2, False))
    x0 = x - total if anchor == "end" else x - total / 2 if anchor == "middle" else x
    parts = [f'<tspan font-style="italic">{base}</tspan>']
    dy = 0.0
    if sub:
        st = "normal" if sub[0].isdigit() else "italic"  # digits upright, as LaTeX sets them
        parts.append(f'<tspan x="{x0 + wb:.1f}" dy="{0.30 * size:.1f}" font-size="{s2:.1f}" font-style="{st}">{sub}</tspan>')
        dy = 0.30 * size
    if sup:
        parts.append(f'<tspan x="{x0 + wb:.1f}" dy="{-dy - 0.36 * size:.1f}" font-size="{s2:.1f}">{sup}</tspan>')
    add(f'<text x="{x0:.1f}" y="{y}" font-family="{SERIF}" font-size="{size}" fill="{fill}">{"".join(parts)}</text>')
    return x0 + total


def rect(x, y, w, h, fill, stroke, sw=2.5, rx=14, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(pts, color=INK, sw=3, dash=None, head="ah", both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{a},{b}" for a, b in pts)
    start = f' marker-start="url(#{head}s)"' if both else ""
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{d}'
        f' marker-end="url(#{head})"{start}/>')


def tag(x, y, label, color, bg):
    w = 22 + len(label) * 14
    add(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="17" fill="{bg}" stroke="{color}" stroke-width="2"/>')
    text(x + w / 2, y + 24, label, size=18, anchor="middle", weight="bold", fill=color, extra='letter-spacing="1.5"')


def chip(cx, cy, w, h, base, sub, size=28):
    rect(cx - w / 2, cy - h / 2, w, h, BLUE_CHIP, BLUE, sw=2, rx=8)
    math(cx, cy + size * 0.22, base, sub=sub, size=size, anchor="middle", fill=BLUE_INK)


def frame_uris() -> list[str]:
    strip = Image.open(STRIP).convert("RGB")
    n = strip.width // strip.height
    side = strip.height
    uris = []
    for i in FRAMES:
        assert i < n
        buf = io.BytesIO()
        strip.crop((i * side, 0, (i + 1) * side, side)).save(buf, "JPEG", quality=92)
        uris.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return uris


def build() -> str:
    marker = lambda mid, c: (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.2" markerHeight="4.2" '
                             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="Frozen fine-tuned pi0.5 controls the LIBERO rollout; a separately trained failure detector reads '
        f'pi0.5 internal latents at every VLA call and raises a failure alert before the rollout fails.">')
    add("<defs>" + marker("ah", INK) + marker("ahs", INK) + marker("ahb", BLUE) + marker("aho", ORANGE)
        + marker("ahr", RED) + marker("ahrs", RED) + "</defs>")
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ---------------------------------------------------------------- lane divider
    add(f'<line x1="40" y1="590" x2="1880" y2="590" stroke="#B5B5B5" stroke-width="2" stroke-dasharray="10 8"/>')
    text(1860, 578, "Robot control loop", size=20, anchor="end", fill=GREY, weight="bold")
    text(1860, 616, "Failure monitoring", size=20, anchor="end", fill=GREY, weight="bold")

    # ---------------------------------------------------------------- 1. pi0.5 (frozen)
    rect(60, 80, 680, 460, BLUE_BG, BLUE, sw=3)
    text(90, 130, "Fine-tuned π0.5", size=34, weight="bold")
    tag(604, 101, "FROZEN", BLUE, "#FFFFFF")
    for x0, l1, l2, s1, s2 in ((90, "Vision-Language", "Representation", "PaliGemma · 2048-d", "image + text tokens"),
                               (450, "Action Expert", "Representation", "Gemma expert · 1024-d", "flow matching, K = 10")):
        rect(x0, 230, 260, 180, "#FFFFFF", BLUE, sw=2.5, rx=10)
        text(x0 + 130, 280, l1, size=26, anchor="middle", weight="bold")
        text(x0 + 130, 312, l2, size=26, anchor="middle", weight="bold")
        text(x0 + 130, 352, s1, size=20, anchor="middle", fill=GREY)
        text(x0 + 130, 380, s2, size=20, anchor="middle", fill=GREY)
    arrow([(350, 320), (448, 320)], sw=2.5)
    text(400, 306, "KV cache", size=18, anchor="middle", fill=GREY)

    # ---------------------------------------------------------------- control loop: o_t in, a_t out
    arrow([(920, 190), (220, 190), (220, 228)], sw=3)
    text(830, 178, "observation", size=22, anchor="middle")
    math(830, 222, "o", sub="t", size=28, anchor="middle")
    arrow([(710, 380), (918, 380)], sw=3)
    text(820, 368, "action chunk", size=22, anchor="middle")
    math(820, 412, "a", sub="t", size=28, anchor="middle")

    # ---------------------------------------------------------------- 2. LIBERO rollout
    rect(920, 80, 940, 460, ENV_BG, ENV, sw=3)
    text(950, 130, "LIBERO Rollout", size=34, weight="bold")
    text(1830, 130, "LIBERO-Spatial, failed episode", size=20, anchor="end", fill=GREY)
    fx = (950, 1150, 1350, 1630)
    for i, (x, uri) in enumerate(zip(fx, frame_uris())):
        add(f'<image x="{x}" y="175" width="180" height="180" href="{uri}" preserveAspectRatio="xMidYMid slice"/>')
        add(f'<rect x="{x}" y="175" width="180" height="180" fill="none" stroke="{ENV}" stroke-width="1.5"/>')
        sub = "T" if i == 3 else str(i + 1)
        math(x + 90, 392, "t", sub=sub, size=28, anchor="middle")
        chip(x + 90, 432, 76, 40, "z", sub, size=24)
    text(1580, 278, "…", size=40, anchor="middle", fill=GREY)
    text(1580, 440, "…", size=32, anchor="middle", fill=GREY)
    arrow([(950, 482), (1828, 482)], color=GREY, sw=2.5)
    text(950, 516, "1 VLA call = 10 env steps; latent extracted at every call", size=20, fill=GREY)
    text(1830, 516, "time", size=20, anchor="end", fill=GREY, style="italic")

    # ---------------------------------------------------------------- taps out of pi0.5
    for x, sup in ((220, "VLM"), (580, "ACT")):
        add(f'<circle cx="{x}" cy="410" r="7" fill="{BLUE}"/>')
        arrow([(x, 417), (x, 648)], color=BLUE, sw=3, head="ahb")
        math(x - 14, 478, "z", sub="t", sup=sup, size=30, anchor="end", fill=BLUE_INK)
    text(400, 470, "read-only", size=18, anchor="middle", fill=BLUE_INK, style="italic")
    text(400, 492, "forward tap", size=18, anchor="middle", fill=BLUE_INK, style="italic")

    # ---------------------------------------------------------------- 3. latent features
    rect(60, 650, 680, 390, "#FFFFFF", BLUE, sw=3)
    text(90, 700, "VLA Latent Features", size=30, weight="bold")
    text(90, 732, "π0.5 internal hidden states, one per VLA call", size=20, fill=GREY)
    for cx, sub in ((150, "1"), (280, "2"), (410, "3"), (640, "t")):
        chip(cx, 802, 100, 64, "z", sub, size=30)
    text(525, 812, "…", size=36, anchor="middle", fill=GREY)
    y = 900
    for sup, desc in (("VLM", "mean over image + text tokens"), ("ACT", "action-expert tokens, last flow step")):
        math(96, y, "z", sub="t", sup=sup, size=28, fill=BLUE_INK)
        text(166, y, ": " + desc, size=20, fill=INK)
        y += 46
    text(96, y + 4, "Detector input: either latent, or both concatenated", size=20, fill=GREY, style="italic")

    # ---------------------------------------------------------------- 4. detector (trained, separate)
    arrow([(740, 802), (838, 802)], sw=3)
    math(790, 786, "z", sub="1:t", size=26, anchor="middle")
    tag(840, 662, "TRAINED", ORANGE, "#FFFFFF")
    rect(840, 712, 270, 180, ORANGE_BG, ORANGE, sw=3)
    text(975, 768, "Failure", size=30, anchor="middle", weight="bold")
    text(975, 804, "Detector", size=30, anchor="middle", weight="bold")
    text(975, 844, "causal LSTM", size=20, anchor="middle", fill=GREY)
    text(975, 870, "or linear probe", size=20, anchor="middle", fill=GREY)
    arrow([(1110, 802), (1208, 802)], color=ORANGE, sw=3, head="aho")
    math(1160, 786, "p", sub="t", size=28, anchor="middle", fill=ORANGE)

    # ---------------------------------------------------------------- 5. failure prediction plot (schematic)
    rect(1210, 650, 650, 390, "#FFFFFF", "#9A9A9A", sw=2)
    text(1240, 700, "Failure Prediction", size=30, weight="bold")
    text(1830, 700, "schematic", size=18, anchor="end", fill=GREY, style="italic")
    ox, oy, xw, yh = 1330, 960, 470, 200  # plot origin, x span, y span (p = 1 at oy - yh)
    X = lambda u: ox + u * xw
    Y = lambda p: oy - p * yh
    thr, ua = 0.6, 0.68
    arrow([(ox, oy), (ox + xw + 26, oy)], sw=2.5)
    arrow([(ox, oy), (ox, Y(1) - 22)], sw=2.5)
    text(ox - 12, oy + 7, "0", size=20, anchor="end", fill=GREY)
    add(f'<line x1="{ox - 6}" y1="{Y(1)}" x2="{ox}" y2="{Y(1)}" stroke="{INK}" stroke-width="2.5"/>')
    text(ox - 12, Y(1) + 7, "1", size=20, anchor="end", fill=GREY)
    text(ox + xw / 2, 1006, "Rollout step (VLA call)", size=22, anchor="middle")
    text(1272, (oy + Y(1)) / 2, "Failure probability", size=22, anchor="middle",
         extra=f'transform="rotate(-90 1272 {(oy + Y(1)) / 2})"')
    # threshold
    add(f'<line x1="{ox}" y1="{Y(thr)}" x2="{X(1)}" y2="{Y(thr)}" stroke="#333333" stroke-width="2.5" stroke-dasharray="10 8"/>')
    text(ox + 12, Y(thr) + 26, "Alert threshold", size=20, fill="#333333")
    # rollout ends in failure
    add(f'<line x1="{X(1)}" y1="{oy}" x2="{X(1)}" y2="{Y(1) - 4}" stroke="{INK}" stroke-width="2.5"/>')
    text(X(1) + 6, Y(1) - 14, "Rollout fails", size=20, anchor="end", weight="bold")
    # curve: low early, rising, crossing the threshold exactly at ua
    pts = [(0, .06), (.1, .07), (.2, .08), (.3, .10), (.4, .14), (.5, .22), (.6, .40), (ua, thr), (.78, .74),
           (.9, .83), (1.0, .88)]
    P = [(X(u), Y(p)) for u, p in pts]
    d = f"M {P[0][0]:.1f} {P[0][1]:.1f}"
    for i in range(len(P) - 1):  # Catmull-Rom -> cubic Bezier
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f" C {c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
    add(f'<path d="{d}" fill="none" stroke="{ORANGE}" stroke-width="4" stroke-linecap="round"/>')
    # alert
    ax_, ay_ = X(ua), Y(thr)
    add(f'<line x1="{ax_}" y1="{ay_}" x2="{ax_}" y2="{oy}" stroke="{RED}" stroke-width="2" stroke-dasharray="6 6"/>')
    add(f'<circle cx="{ax_}" cy="{ay_}" r="10" fill="{RED}" stroke="#FFFFFF" stroke-width="3"/>')
    text(ax_ - 16, ay_ - 18, "Failure Alert", size=22, anchor="end", weight="bold", fill=RED)
    arrow([(ax_ + 4, 932), (X(1) - 4, 932)], color=RED, sw=2.5, head="ahr", both=True)
    text((ax_ + X(1)) / 2, 920, "lead time", size=20, anchor="middle", fill=RED, weight="bold")

    add("</svg>")
    return "\n".join(out)


def main():
    svg = build()
    (HERE / "pipeline.svg").write_text(svg)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        page.write_text(f"<!doctype html><html><body style='margin:0;background:#fff'>{svg}</body></html>")
        subprocess.run(["google-chrome", "--headless", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={W},{H}", "--force-device-scale-factor=2",
                        f"--screenshot={HERE / 'pipeline.png'}", page.as_uri()],
                       check=True, capture_output=True)
    print(HERE / "pipeline.svg", HERE / "pipeline.png")


if __name__ == "__main__":
    main()
