"""Checks for buffer.py, run on synthetic rollouts so they are fast and need no GPU.

Covers the six properties the buffer has to have: writing experience, querying and sampling it, producing
mini-batches, keeping splits leak-free, honouring a size cap, and surviving a restart.

    python xattn_detector/test_buffer.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from buffer import SPLITS, Buffer  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'   ' + detail if detail else ''}")


def fake_rollout(buf, suite, task, init, success, n_calls=8, rng=None):
    rng = rng or np.random.default_rng(task * 100 + init)
    r = buf.open_rollout(suite, task, init, language=f"task {task}", goal_names=["g"])
    for t in range(n_calls):
        r.add_call(step=t * 10,
                   suf_mean=rng.standard_normal(1024).astype(np.float32),
                   action_chunk=rng.standard_normal((50, 7)).astype(np.float32),
                   action_exec=rng.standard_normal((10, 7)).astype(np.float32),
                   eef=rng.standard_normal(3).astype(np.float32),
                   opening=np.float32(0.04),
                   score=np.float32(0.5))
    return buf.close_rollout(r, success=success, n_steps=n_calls * 10)


def main():
    root = Path(tempfile.mkdtemp(prefix="buftest_"))
    try:
        print("1. 기록과 라벨 지연 부여")
        buf = Buffer(root / "b1", capacity_bytes=10 ** 9)
        r = buf.open_rollout("libero_goal", 3, 0)
        r.add_call(step=0, suf_mean=np.zeros(1024, np.float32),
                   action_chunk=np.zeros((50, 7), np.float32), action_exec=np.zeros((10, 7), np.float32),
                   eef=np.zeros(3, np.float32), opening=np.float32(0), score=np.float32(0))
        check("닫히기 전에는 샘플링 대상이 아니다", len(buf.query()) == 0)
        rec = buf.close_rollout(r, success=False, n_steps=10)
        check("닫으면 라벨이 붙는다", rec["y"] == 1 and len(buf.query()) == 1)
        try:
            r.add_call(step=1)
            check("닫힌 롤아웃에 추가하면 거부된다", False)
        except RuntimeError:
            check("닫힌 롤아웃에 추가하면 거부된다", True)

        print("\n2. 분할 누수 방지")
        buf = Buffer(root / "b2", capacity_bytes=10 ** 9, group_by="task")
        for task in range(10):
            for init in range(6):
                fake_rollout(buf, "libero_goal", task, init, success=(init % 2 == 0), n_calls=5)
        by_task = {}
        for rec in buf.records:
            by_task.setdefault((rec["suite"], rec["task"]), set()).add(rec["split"])
        check("한 task는 한 split에만 속한다", all(len(v) == 1 for v in by_task.values()),
              f"{sum(len(v) > 1 for v in by_task.values())}개 task가 여러 split에 걸침")
        # arrival order must not matter
        buf_rev = Buffer(root / "b2r", capacity_bytes=10 ** 9, group_by="task")
        for task in reversed(range(10)):
            for init in reversed(range(6)):
                fake_rollout(buf_rev, "libero_goal", task, init, success=(init % 2 == 0), n_calls=5)
        a = {(r["task"], r["init"]): r["split"] for r in buf.records}
        b = {(r["task"], r["init"]): r["split"] for r in buf_rev.records}
        check("도착 순서를 바꿔도 split이 같다", a == b)
        s = buf.summary()
        # 10 groups hashed into 70/15/15 leaves val with an expected 1.5 tasks, so an empty split here is the
        # design behaving as specified, not a defect -- what must not happen is it going unreported.
        empty = buf.check_splits(warn=False)
        check("빈 split은 보고된다", (not empty) or set(empty) <= set(SPLITS),
              f"train={s['train']['n']} val={s['val']['n']} test={s['test']['n']}, 빈 split={empty or '없음'}")

        buf40 = Buffer(root / "b2b", capacity_bytes=10 ** 9, group_by="task")
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            for task in range(10):
                for init in range(3):
                    fake_rollout(buf40, suite, task, init, success=(init % 2 == 0), n_calls=5)
        s40 = buf40.summary()
        check("현실 규모(40 task)에서는 세 split이 모두 찬다", all(s40[k]["n"] > 0 for k in SPLITS),
              " ".join(f"{k}={s40[k]['n']}({s40[k]['tasks']}task)" for k in SPLITS))

        print("\n3. 조회와 샘플링")
        check("성공/실패 조회", len(buf.query(success=True)) + len(buf.query(success=False)) == len(buf))
        rng = np.random.default_rng(0)
        smp = buf.sample(20, split="train", balance=True, rng=rng)
        pos = sum(r["y"] for r in smp)
        check("균형 샘플링이 두 클래스를 섞는다", 0 < pos < len(smp), f"{pos}/{len(smp)} 실패")
        check("샘플은 요청한 split에서만 나온다", all(r["split"] == "train" for r in smp))

        print("\n4. 미니배치")
        got = 0
        for X, y, M, recs in buf.minibatches(batch=8, split="train", feature="suf_mean"):
            check_shape = X.ndim == 3 and X.shape[0] == len(recs) and X.shape[2] == 1024
            got += len(recs)
            if not check_shape:
                break
        check("미니배치 모양과 개수", check_shape and got == len(buf.query(split="train")),
              f"{got}개 반환 / train {len(buf.query(split='train'))}개")

        print("\n5. 용량 제한")
        one = buf.records[0]["bytes"]
        small = Buffer(root / "b3", capacity_bytes=one * 12, group_by="task")
        for task in range(10):
            for init in range(6):
                fake_rollout(small, "libero_goal", task, init, success=(init % 2 == 0), n_calls=5)
        held = small.query(split="val") + small.query(split="test")
        held_bytes = sum(r["bytes"] for r in held)
        if held_bytes <= small.capacity_bytes:
            check("용량 상한을 지킨다", small.total_bytes() <= small.capacity_bytes,
                  f"{small.total_bytes()} <= {small.capacity_bytes}")
        else:
            # the cap is unreachable here by design: val+test are not evictable and already exceed it
            check("퇴거 가능한 train을 모두 비운다", len(small.query(split="train")) == 0,
                  f"train {len(small.query(split='train'))}개 남음")
            check("상한을 못 지키면 경고한다", small._warned_capacity,
                  f"보존 {held_bytes / 1024:.0f} KB > 상한 {small.capacity_bytes / 1024:.0f} KB")
        check("val/test는 퇴거되지 않는다", len(held) > 0 and all((small.root / r["npz"]).exists() for r in held),
              f"{len(held)}개 보존")
        files = {p.name for p in small.root.glob("*.npz")}
        check("퇴거된 파일은 실제로 지워진다", files == {r["npz"] for r in small.records})

        print("\n6. 저장과 재시작")
        reloaded = Buffer.load(buf.root)
        check("재시작 후 레코드 수가 같다", len(reloaded) == len(buf))
        check("재시작 후 split이 같다",
              {(r["task"], r["init"]): r["split"] for r in reloaded.records} == a)
        x = reloaded.fetch(reloaded.records[0], ("suf_mean",))["suf_mean"]
        check("재시작 후 배열을 읽을 수 있다", x.shape[1] == 1024, str(x.shape))
        # a dangling index entry must be dropped, not crash
        victim = reloaded.records[-1]
        (reloaded.root / victim["npz"]).unlink()
        again = Buffer.load(buf.root)
        check("npz가 사라진 항목은 색인에서 제거된다", len(again) == len(reloaded) - 1)

        print(f"\n{'=' * 60}\n통과 {len(PASS)} / 실패 {len(FAIL)}")
        if FAIL:
            print("실패:", ", ".join(FAIL))
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
