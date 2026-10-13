"""Append-only rollout buffer for online failure-detector training.

A rollout's label only exists once it has ended, so experience is written in two stages: `open_rollout` /
`add_call` while the robot is moving, then `close_rollout(success=...)` which stamps the label onto everything
that rollout recorded. Nothing is trainable until it has been closed.

What a call stores, and why the default is small: the offline comparison found that a linear probe on the
mean-pooled action-expert vector matched every richer model, on seen and unseen tasks alike (0.724 vs 0.724 for
cross-attention on unseen tasks). Token-level and VLM tensors cost 50x and 385x more per call, so they are
opt-in rather than the default.

    suf_mean   (1024,)      always      2 KB/call
    suf_tokens (50, 1024)   opt-in    100 KB/call
    pre_img/pre_lang        opt-in    770 KB/call
    action chunk, executed actions, eef, gripper, goal predicates, score   always, negligible

Because of that spread the capacity limit is in BYTES, not episodes -- the same "10000 rollouts" means 1 GB or
1 TB depending on what is switched on. Eviction drops whole rollouts, oldest first, and never evicts a rollout
assigned to validation or test.

Leakage: every rollout gets a split the moment it is opened, from a hash of its grouping key (task id, or task
and init) -- not from when it arrived. A rollout therefore cannot migrate between splits as the buffer grows,
re-shuffles, or is reloaded from disk, and the same task cannot appear on both sides of a comparison.

On disk: one npz per rollout plus an append-only index.jsonl, so an interrupted run loses at most the rollout
that was in flight. `Buffer.load` rebuilds the index and re-checks every npz it lists.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
FIELDS_ALWAYS = ("suf_mean", "action_chunk", "action_exec", "eef", "opening", "score", "step")
FIELDS_OPT = ("suf_tokens", "pre_img", "pre_lang", "lang_mask")


def _split_of(key: str, fracs=(0.7, 0.15, 0.15), salt: str = "safe-v1") -> str:
    """Deterministic split from the grouping key alone, so it never depends on arrival order."""
    h = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2 ** 64
    if u < fracs[0]:
        return "train"
    return "val" if u < fracs[0] + fracs[1] else "test"


class Rollout:
    """One episode being recorded. Calls accumulate here; the label arrives only at the end."""

    def __init__(self, rollout_id: str, suite: str, task: int, init: int, language: str,
                 goal_names: list[str], split: str):
        self.id = rollout_id
        self.meta = dict(rollout_id=rollout_id, suite=suite, task=task, init=init, language=language,
                         goal_names=goal_names, split=split, opened=time.time())
        self.calls: list[dict] = []
        self.closed = False

    def add_call(self, **kw):
        if self.closed:
            raise RuntimeError(f"rollout {self.id} is closed")
        self.calls.append(kw)

    def pack(self, keep: tuple[str, ...]):
        out = {}
        for f in keep:
            vals = [c[f] for c in self.calls if f in c]
            if len(vals) != len(self.calls):
                continue
            a = np.asarray(vals)
            out[f] = a.astype(np.float16) if a.dtype == np.float32 and f != "eef" else a
        return out


class Buffer:
    """Rollout store with deterministic splits, byte-capped eviction and crash-safe persistence."""

    def __init__(self, root: str | Path, keep_tokens: bool = False, keep_prefix: bool = False,
                 capacity_bytes: int = 8 * 1024 ** 3, fracs=(0.7, 0.15, 0.15), salt: str = "safe-v1",
                 group_by: str = "task"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.keep = FIELDS_ALWAYS + tuple(
            f for f in FIELDS_OPT
            if (keep_tokens and f == "suf_tokens") or (keep_prefix and f in ("pre_img", "pre_lang", "lang_mask")))
        self.capacity_bytes = int(capacity_bytes)
        self.fracs, self.salt, self.group_by = fracs, salt, group_by
        self.records: list[dict] = []
        self._open: dict[str, Rollout] = {}
        self._warned_capacity = False
        (self.root / "manifest.json").write_text(json.dumps(
            dict(keep=list(self.keep), capacity_bytes=self.capacity_bytes, fracs=list(fracs),
                 salt=salt, group_by=group_by), indent=2))

    # ------------------------------------------------------------------ split
    def group_key(self, suite: str, task: int, init: int) -> str:
        return f"{suite}/t{task}" if self.group_by == "task" else f"{suite}/t{task}/i{init}"

    def split_for(self, suite: str, task: int, init: int) -> str:
        return _split_of(self.group_key(suite, task, init), self.fracs, self.salt)

    # ------------------------------------------------------------------ write
    def open_rollout(self, suite: str, task: int, init: int, language: str = "",
                     goal_names: list[str] | None = None, rollout_id: str | None = None) -> Rollout:
        rid = rollout_id or f"{suite}_t{task}_i{init}_{int(time.time() * 1000) % 10 ** 9}"
        r = Rollout(rid, suite, task, init, language, goal_names or [], self.split_for(suite, task, init))
        self._open[rid] = r
        return r

    def close_rollout(self, r: Rollout, success: bool, n_steps: int, failure_onset_call: int | None = None):
        """Stamp the outcome onto the rollout and persist it. Only closed rollouts are ever sampled."""
        r.closed = True
        self._open.pop(r.id, None)
        arrays = r.pack(self.keep)
        path = self.root / f"{r.id}.npz"
        np.savez_compressed(path, **arrays)
        rec = dict(**r.meta, success=bool(success), y=int(not success), n_calls=len(r.calls),
                   n_steps=int(n_steps), failure_onset_call=failure_onset_call,
                   closed=time.time(), bytes=path.stat().st_size, npz=path.name)
        with open(self.index_path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.records.append(rec)
        self.evict()
        return rec

    # ------------------------------------------------------------------ capacity
    def total_bytes(self) -> int:
        return sum(r["bytes"] for r in self.records)

    def evict(self) -> int:
        """Drop training rollouts, oldest first, until under capacity. Never evicts val or test.

        Held-out data is not evictable, so the cap CANNOT be met once val+test alone exceed it. That state is
        reported rather than passed over quietly: silently overrunning a disk budget is worse than saying so,
        and silently dropping evaluation data would invalidate every comparison made afterwards.
        """
        dropped = 0
        while self.total_bytes() > self.capacity_bytes:
            cand = [r for r in self.records if r["split"] == "train"]
            if not cand:
                if not self._warned_capacity:
                    hum = lambda n: f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else (
                        f"{n / 1024 ** 2:.1f} MB" if n >= 1024 ** 2 else f"{n / 1024:.0f} KB")
                    print(f"[Buffer] 용량 {hum(self.capacity_bytes)} 를 지킬 수 없습니다 — "
                          f"퇴거 불가한 val/test 만으로 {hum(self.total_bytes())}. train 은 비었습니다. "
                          f"capacity_bytes 를 올리거나 fracs 를 조정하세요.")
                    self._warned_capacity = True
                break
            victim = min(cand, key=lambda r: r["closed"])
            (self.root / victim["npz"]).unlink(missing_ok=True)
            self.records.remove(victim)
            dropped += 1
        if dropped:
            self._rewrite_index()
        return dropped

    def check_splits(self, warn: bool = True) -> list[str]:
        """Splits that are empty. Hashing a small number of groups can leave one with nothing in it."""
        empty = [s for s in SPLITS if not self.query(split=s)]
        if empty and warn:
            n_groups = len({self.group_key(r["suite"], r["task"], r["init"]) for r in self.records})
            print(f"[Buffer] 비어 있는 split: {', '.join(empty)}  (그룹 {n_groups}개를 "
                  f"group_by='{self.group_by}' 로 해시 분할). 그룹 수가 적으면 생길 수 있습니다 — "
                  f"group_by='task_init' 로 바꾸거나 그룹을 더 모으세요.")
        return empty

    def _rewrite_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.records) + "\n")
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------ read
    @classmethod
    def load(cls, root: str | Path) -> "Buffer":
        root = Path(root)
        man = json.loads((root / "manifest.json").read_text())
        b = cls(root, capacity_bytes=man["capacity_bytes"], fracs=tuple(man["fracs"]),
                salt=man["salt"], group_by=man["group_by"])
        b.keep = tuple(man["keep"])
        recs, missing = [], 0
        if b.index_path.exists():
            for line in b.index_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if (root / r["npz"]).exists():
                    recs.append(r)
                else:
                    missing += 1
        b.records = recs
        if missing:
            print(f"[Buffer] index listed {missing} rollouts whose npz is gone; dropped them")
            b._rewrite_index()
        return b

    def query(self, split: str | None = None, success: bool | None = None, suite: str | None = None,
              task: int | None = None, min_calls: int = 0) -> list[dict]:
        out = self.records
        if split is not None:
            out = [r for r in out if r["split"] == split]
        if success is not None:
            out = [r for r in out if r["success"] == success]
        if suite is not None:
            out = [r for r in out if r["suite"] == suite]
        if task is not None:
            out = [r for r in out if r["task"] == task]
        return [r for r in out if r["n_calls"] >= min_calls]

    def fetch(self, rec: dict, fields: tuple[str, ...] | None = None) -> dict:
        z = np.load(self.root / rec["npz"])
        want = fields or tuple(z.files)
        return {f: z[f] for f in want if f in z.files}

    def sample(self, n: int, split: str = "train", balance: bool = True, rng=None,
               min_calls: int = 1) -> list[dict]:
        """n rollout records. `balance` draws half from each class, which matters early on when one is rare."""
        rng = rng or np.random.default_rng()
        pool = self.query(split=split, min_calls=min_calls)
        if not pool:
            return []
        if not balance:
            idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
            return [pool[i] for i in idx]
        pos = [r for r in pool if r["y"] == 1]
        neg = [r for r in pool if r["y"] == 0]
        out = []
        for grp, k in ((pos, n // 2), (neg, n - n // 2)):
            if not grp:
                continue
            idx = rng.choice(len(grp), min(k, len(grp)), replace=False)
            out += [grp[i] for i in idx]
        rng.shuffle(out)
        return out

    def minibatches(self, batch: int, split: str = "train", feature: str = "suf_mean",
                    epochs: int = 1, rng=None, min_calls: int = 1):
        """Yield (X (B, T, D) float32, y (B,), records). Episodes are padded to the batch's longest."""
        rng = rng or np.random.default_rng()
        pool = self.query(split=split, min_calls=min_calls)
        for _ in range(epochs):
            order = rng.permutation(len(pool))
            for s in range(0, len(pool), batch):
                recs = [pool[i] for i in order[s:s + batch]]
                xs = [self.fetch(r, (feature,))[feature].astype(np.float32) for r in recs]
                T = max(x.shape[0] for x in xs)
                X = np.zeros((len(xs), T, *xs[0].shape[1:]), np.float32)
                M = np.zeros((len(xs), T), bool)
                for i, x in enumerate(xs):
                    X[i, : x.shape[0]] = x
                    M[i, : x.shape[0]] = True
                yield X, np.array([r["y"] for r in recs], np.float32), M, recs

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        out = {"n": len(self.records), "bytes": self.total_bytes(), "capacity": self.capacity_bytes}
        for s in SPLITS:
            g = self.query(split=s)
            out[s] = dict(n=len(g), fail=sum(r["y"] for r in g), succ=sum(1 - r["y"] for r in g),
                          tasks=len({(r["suite"], r["task"]) for r in g}))
        return out

    def __len__(self):
        return len(self.records)
