"""Atomic JSON persistence: versioned read-only snapshots and sampling tasks.

A *version* is an immutable snapshot of the game data plus the analysis plan
(guarantees / budget) recorded when the user commits. Re-committing after edits
creates a new version; older versions and every allocation computed against
them remain readable forever.

Sampling tasks belong to one version. Their state stores the count of already
consumed permutations (a cursor into the deterministic SplitMix64 stream), so
resuming a task continues the stream instead of restarting it — permutations
are never double-counted. A fingerprint mismatch between the stored task and
the current game data is rejected loudly.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Dict, List, Optional

from .shapley import SamplingState


class StoreError(RuntimeError):
    pass


def game_fingerprint(game_dict: dict) -> str:
    """Semantic hash of game inputs; stable across round-trip serialization.

    Only fields that affect computations participate: player order and, per
    coalition, its member set, exact value or bounds, and shared cost.
    UI-only metadata (notes, lock flags, default keys) is ignored.
    """
    from .games import Game
    game = Game.from_dict(game_dict)
    canonical_players = list(game.players)
    canonical_entries = []
    for mask in sorted(game.entries):
        entry = game.entries[mask]
        lo, hi = game.worth_bounds(mask)
        canonical_entries.append([
            sorted(game.players[i] for i in range(len(game.players))
                   if mask & (1 << i)),
            None if entry.value is None else float(entry.value),
            None if lo == float("-inf") else float(lo + entry.cost),
            None if hi == float("inf") else float(hi + entry.cost),
            float(entry.cost),
        ])
    payload = json.dumps(
        {"players": canonical_players, "entries": canonical_entries},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = {
            "versions": [],          # immutable snapshots, newest last
            "sampling_tasks": {},    # task_id -> state envelope
            "active_version_id": None,
        }
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)

    # ------------------------------------------------------------- primitives
    def _atomic_write(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".store-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2,
                          allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---------------------------------------------------------------- versions
    def create_version(self, game_dict: dict, plan: dict,
                       parent_id: Optional[str] = None) -> dict:
        # Round-trip through Game so internal +/-infinity bounds are rendered
        # back to JSON null before the strict (allow_nan=False) atomic write.
        from .games import Game
        game_dict = Game.from_dict(game_dict).to_dict()
        vid = "v" + str(int(time.time() * 1000)) + "-" + game_fingerprint(game_dict)[:6]
        # guarantee uniqueness even for same-millisecond re-commits
        existing = {v["id"] for v in self.data["versions"]}
        suffix = 1
        unique_vid = vid
        while unique_vid in existing:
            unique_vid = f"{vid}-{suffix}"
            suffix += 1
        vid = unique_vid
        version = {
            "id": vid,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game": game_dict,
            "fingerprint": game_fingerprint(game_dict),
            "plan": plan,          # guarantees / budget at commit time
            "parent_id": parent_id,
            "allocations": [],     # analyses pinned to this immutable version
            "read_only": False,    # flips to True when the next version lands
        }
        for old in self.data["versions"]:
            old["read_only"] = True
        self.data["versions"].append(version)
        self.data["active_version_id"] = vid
        self._atomic_write()
        return version

    def list_versions(self) -> List[dict]:
        summary = []
        for v in self.data["versions"]:
            summary.append({
                "id": v["id"],
                "created_at": v["created_at"],
                "name": v["game"].get("name", ""),
                "players": list(v["game"].get("players", [])),
                "read_only": v["read_only"],
                "parent_id": v["parent_id"],
                "allocation_count": len(v["allocations"]),
            })
        return summary

    def get_version(self, version_id: str) -> dict:
        for v in self.data["versions"]:
            if v["id"] == version_id:
                return v
        raise StoreError(f"版本不存在: {version_id}")

    def active_version(self) -> Optional[dict]:
        vid = self.data.get("active_version_id")
        if vid is None:
            return None
        return self.get_version(vid)

    def add_allocation(self, version_id: str, record: dict) -> dict:
        version = self.get_version(version_id)
        record = dict(record)
        record["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        version["allocations"].append(record)
        self._atomic_write()
        return record

    # --------------------------------------------------------- sampling tasks
    def create_sampling_task(self, version_id: str, fingerprint: str,
                             seed: Optional[int] = None,
                             batch_size: int = 200) -> dict:
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big")
        task_id = "task-" + str(len(self.data["sampling_tasks"]) + 1) + "-" + version_id[:8]
        envelope = {
            "id": task_id,
            "version_id": version_id,
            "fingerprint": fingerprint,
            "state": SamplingState(seed=seed, batch_size=batch_size).to_dict(),
            "last_summary": None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.data["sampling_tasks"][task_id] = envelope
        self._atomic_write()
        return envelope

    def get_task(self, task_id: str) -> dict:
        task = self.data["sampling_tasks"].get(task_id)
        if task is None:
            raise StoreError(f"抽样任务不存在: {task_id}")
        return task

    def list_tasks(self, version_id: Optional[str] = None) -> List[dict]:
        out = list(self.data["sampling_tasks"].values())
        if version_id is not None:
            out = [t for t in out if t["version_id"] == version_id]
        return out

    def save_task(self, task_id: str, state: SamplingState, summary: dict) -> None:
        task = self.get_task(task_id)
        task["state"] = state.to_dict()
        task["last_summary"] = summary
        self._atomic_write()

    def verify_task(self, task_id: str, fingerprint: str) -> dict:
        task = self.get_task(task_id)
        if task["fingerprint"] != fingerprint:
            raise StoreError(
                "抽样任务绑定的游戏数据已变化（指纹不一致），拒绝恢复以避免排列累计错误。"
                "请基于当前版本新建抽样任务。")
        return task
