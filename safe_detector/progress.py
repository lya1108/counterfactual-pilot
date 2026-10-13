"""Read-only progress report for a running collect_rollouts sweep.

Reads results/collect.log and run_collect.sh. Touches nothing else: no writes, no signals, no imports from the
running process. Safe to run as often as you like while collection is in flight.

    python safe_detector/progress.py
    watch -n 10 "python ~/yuna_ws/safe_detector/progress.py"
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "results" / "collect.log"
PLAN = HERE / "run_collect.sh"

# robosuite step caps per suite (fasi/envs/libero_env.py:14) -- used only to scale the ETA for suites not started yet
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}

EP = re.compile(r"^task (\d+) init (\d+) success=(True|False) t=(\d+) calls=(\d+) \(([\d.]+)s\)")
SUITE = re.compile(r"^#{4,}\s*(\S+)\s*#{4,}")


def parse_ids(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def read_plan():
    """Suites and the task/init grid, taken from run_collect.sh so the two can never drift apart."""
    txt = PLAN.read_text() if PLAN.exists() else ""
    m = re.search(r"for suite in ([^\n;]+); do", txt)
    suites = m.group(1).split() if m else []
    t = re.search(r"--tasks\s+(\S+)", txt)
    i = re.search(r"--inits\s+(\S+)", txt)
    tasks = parse_ids(t.group(1)) if t else []
    inits = parse_ids(i.group(1)) if i else []
    return suites, tasks, inits


def read_log():
    rows, suite, last_suite = [], None, None
    for line in LOG.read_text(errors="ignore").splitlines():
        ms = SUITE.match(line)
        if ms:
            suite = ms.group(1)
            if suite != "ALL":
                last_suite = suite
            continue
        me = EP.match(line)
        if me:
            rows.append(dict(suite=suite, task=int(me.group(1)), init=int(me.group(2)),
                             success=me.group(3) == "True", t=int(me.group(4)),
                             calls=int(me.group(5)), sec=float(me.group(6))))
    done = "ALL DONE" in LOG.read_text(errors="ignore")
    return rows, last_suite, done


def running_pid():
    try:
        out = subprocess.run(["pgrep", "-f", "run_collect.sh"], capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def elapsed_seconds(pid):
    if pid:
        try:
            s = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            if s:
                return float(s), "process uptime"
        except Exception:
            pass
    st = LOG.stat()
    return st.st_mtime - st.st_ctime, "log first-to-last write"


def hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def bar(frac: float, width: int = 32) -> str:
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def main():
    if not LOG.exists():
        raise SystemExit(f"no log yet: {LOG}")
    suites, tasks, inits = read_plan()
    per_suite = len(tasks) * len(inits)
    planned = len(suites) * per_suite
    rows, cur_suite, finished = read_log()
    pid = running_pid()
    el, el_src = elapsed_seconds(pid)

    n = len(rows)
    ok = sum(r["success"] for r in rows)
    bad = n - ok
    roll = sum(r["sec"] for r in rows)

    print(f"SAFE rollout collection   {'DONE' if finished else ('running pid ' + str(pid)) if pid else 'NOT RUNNING'}")
    print()
    frac = n / planned if planned else 0.0
    print(f"  episodes    {n} / {planned}   {bar(frac)} {frac:6.1%}")
    print(f"  outcome     success {ok}   failure {bad}   success rate {ok / n:.1%}" if n else "  outcome     --")
    print(f"  elapsed     {hms(el)}   ({el_src})")
    if n:
        print(f"  per episode {el / n:5.1f}s wall   {roll / n:5.1f}s rollout   "
              f"(overhead {max(0, el - roll) / max(1, n):.1f}s/ep: model load, env build)")

    # per-suite table
    print()
    print(f"  {'suite':16s} {'done':>9s} {'S':>4s} {'F':>4s} {'mean s':>7s}")
    by = {}
    for r in rows:
        by.setdefault(r["suite"], []).append(r)
    for s in suites:
        g = by.get(s, [])
        if g:
            print(f"  {s:16s} {len(g):4d}/{per_suite:<4d} {sum(x['success'] for x in g):4d} "
                  f"{sum(not x['success'] for x in g):4d} {sum(x['sec'] for x in g) / len(g):7.1f}")
        else:
            print(f"  {s:16s} {'-':>4s}/{per_suite:<4d} {'-':>4s} {'-':>4s} {'-':>7s}")

    # where it is right now
    print()
    if finished:
        print("  current     all suites finished")
    elif rows and cur_suite:
        last = rows[-1]
        nxt_init = last["init"] + 1
        nxt_task = last["task"]
        if nxt_init > max(inits):
            nxt_init, nxt_task = min(inits), last["task"] + 1
        if nxt_task > max(tasks):
            print(f"  current     {cur_suite} finished, next suite starting")
        else:
            print(f"  current     {cur_suite}  task {nxt_task}  init {nxt_init}   "
                  f"(last done: task {last['task']} init {last['init']} "
                  f"{'S' if last['success'] else 'F'})")
    else:
        print(f"  current     {cur_suite or '?'}  loading model / building env")

    # ETA: remaining episodes of started suites at that suite's measured rate; suites not started yet are scaled
    # by their step cap, because an episode that times out is as long as the cap allows.
    if n and not finished:
        ref_cap = sum(MAX_STEPS.get(r["suite"], 300) for r in rows) / n
        base = roll / n  # seconds of rollout per episode at the mix seen so far
        oh = max(0.0, el - roll) / max(1, len(by))  # per-suite overhead (model load dominates)
        eta = 0.0
        for s in suites:
            g = by.get(s, [])
            left = per_suite - len(g)
            if left <= 0:
                continue
            rate = (sum(x["sec"] for x in g) / len(g)) if g else base * MAX_STEPS.get(s, 300) / ref_cap
            eta += left * rate + (0.0 if g else oh)
        print(f"  ETA         {hms(eta)} remaining   (finish around {time.strftime('%H:%M', time.localtime(time.time() + eta))})")
        print(f"              rough: unstarted suites scaled by their step cap "
              f"({', '.join(f'{s}={MAX_STEPS.get(s, 300)}' for s in suites if s not in by) or 'none left'})")


if __name__ == "__main__":
    main()
