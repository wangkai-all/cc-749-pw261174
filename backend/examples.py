"""Built-in fictional examples, all synthetic — no real settlements.

Each example bundles the game data plus independently computed expected
results used by the test suite for cross-checking (efficiency conservation,
symmetry / dummy axioms, core emptiness).
"""
from __future__ import annotations

from typing import Dict, List


def _members(n: int, mask: int) -> List[str]:
    return [f"P{i + 1}" for i in range(n) if mask & (1 << i)]


def _point_rows(n: int, worths: Dict[int, float], costs: Dict[int, float] | None = None,
                notes: Dict[int, str] | None = None) -> List[dict]:
    costs = costs or {}
    notes = notes or {}
    rows = []
    for mask in range(1 << n):
        if mask == 0:
            rows.append({"members": [], "value": 0.0})
            continue
        rows.append({
            "members": _members(n, mask),
            "value": worths[mask],
            "cost": costs.get(mask, 0.0),
            "note": notes.get(mask, ""),
        })
    return rows


def example_symmetry() -> dict:
    """Three symmetric players: equal Shapley shares, non-empty core."""
    n = 3
    worths = {0: 0.0}
    for mask in range(1, 1 << n):
        size = bin(mask).count("1")
        worths[mask] = {1: 10.0, 2: 26.0, 3: 48.0}[size]
    return {
        "id": "symmetry",
        "title": "对称参与方（3 人，非空核）",
        "description": "价值只取决于联盟人数：单干 10、双人 26、全体 48。"
                       "三人完全对称，Shapley 各得 16，核心非空。",
        "game": {"name": "对称三人", "players": ["P1", "P2", "P3"],
                 "entries": _point_rows(n, worths)},
        "expected": {"shapley": [16.0, 16.0, 16.0], "core_nonempty": True,
                     "epsilon_le": 0.0},
    }


def example_dummy() -> dict:
    """Additive game: P3 is a dummy (w(S+3)=w(S)+w({3})), Shapley = singleton."""
    n = 3
    standalone = {0: 0.0, 1: 12.0, 2: 8.0, 4: 3.0}
    worths = {}
    for mask in range(1 << n):
        worths[mask] = sum(standalone[1 << i] for i in range(n) if mask & (1 << i))
    notes = {1 << 2: "P3 是虚设参与方：加入任何联盟只带来自己的 3。"}
    return {
        "id": "dummy",
        "title": "虚设参与方（加法博弈）",
        "description": "联盟价值恒为单干价值之和。P3 为虚设参与方，"
                       "其 Shapley 值恰好等于单干价值 3；核心为唯一分配 (12, 8, 3)。",
        "game": {"name": "加法三人", "players": ["P1", "P2", "P3"],
                 "entries": _point_rows(n, worths, notes=notes)},
        "expected": {"shapley": [12.0, 8.0, 3.0], "core_nonempty": True,
                     "epsilon_le": 0.0, "dummy_player": 2},
    }


def example_empty_core() -> dict:
    """Majority game: w(S)=1 if |S|>=2. Classic empty-core example."""
    n = 3
    worths = {}
    for mask in range(1 << n):
        worths[mask] = 1.0 if bin(mask).count("1") >= 2 else 0.0
    return {
        "id": "empty_core",
        "title": "空核：多数投票博弈",
        "description": "任意两人及以上联盟可实现 1，单干为 0。"
                       "效率要求三人合计 1，但任意两人都要求合计至少 1，"
                       "三组不等式互不相容——核心为空；最小核 epsilon* = 1/3。",
        "game": {"name": "多数投票", "players": ["P1", "P2", "P3"],
                 "entries": _point_rows(n, worths)},
        "expected": {"shapley": [1 / 3, 1 / 3, 1 / 3], "core_nonempty": False,
                     "epsilon": 1 / 3},
    }


def example_costs() -> dict:
    """Non-empty core with shared costs deducted from coalition revenue."""
    n = 3
    revenue = {0: 0.0}
    for mask in range(1, 1 << n):
        size = bin(mask).count("1")
        revenue[mask] = {1: 5.0, 2: 18.0, 3: 30.0}[size]
    costs = {mask: 2.0 for mask in range(1, 1 << n)
             if bin(mask).count("1") >= 2}
    worths = {m: revenue[m] - costs.get(m, 0.0) for m in revenue}
    return {
        "id": "costs",
        "title": "共享成本（凸博弈，非空核）",
        "description": "两人及以上联盟发生 2 的共享成本，净价值 w(S)=收入-成本。"
                       "净博弈为凸博弈，Shapley 分配 (28/3, 28/3, 28/3) 落在核心内。",
        "game": {"name": "成本分摊", "players": ["P1", "P2", "P3"],
                 "entries": _point_rows(n, revenue, costs=costs)},
        "expected": {"shapley": [28 / 3, 28 / 3, 28 / 3], "core_nonempty": True,
                     "epsilon_le": 0.0},
    }


def example_bounded() -> dict:
    """Missing coalition value with explicit bounds — never defaulted to zero."""
    n = 3
    rows = [
        {"members": [], "value": 0.0},
        {"members": ["P1"], "value": 4.0},
        {"members": ["P2"], "value": 3.0},
        {"members": ["P3"], "value": 2.0},
        {"members": ["P1", "P2"], "lo": 8.0, "hi": 12.0,
         "note": "该联盟收益不确定，给上下界而非填 0。"},
        {"members": ["P1", "P3"], "value": 9.0},
        {"members": ["P2", "P3"], "value": 8.0},
        {"members": ["P1", "P2", "P3"], "lo": 18.0, "hi": 20.0},
    ]
    return {
        "id": "bounded",
        "title": "缺失价值 + 上下界传播",
        "description": "联盟 {P1,P2} 与大联盟只有区间价值。Shapley 与核心分析"
                       "均按区间传播，任何一端无界都会显式标注而不是补零。",
        "game": {"name": "区间博弈", "players": ["P1", "P2", "P3"],
                 "entries": rows},
        "expected": {"shapley_lo": [20 / 3, 17 / 3, 13 / 3],
                     "shapley_hi": [8.0, 7.0, 19 / 3]},
    }


def example_large() -> dict:
    """14-player near-convex synthetic game: exact infeasible, sampling used."""
    n = 14
    import random
    rng = random.Random(20261124)
    base = [rng.uniform(1.0, 5.0) for _ in range(n)]
    synergy = {}
    for mask in range(1, 1 << n):
        size = bin(mask).count("1")
        if size == 1:
            synergy[mask] = 0.0
        else:
            synergy[mask] = rng.uniform(0.0, 0.6) * size
    worths = {0: 0.0}
    for mask in range(1, 1 << n):
        worths[mask] = (sum(base[i] for i in range(n) if mask & (1 << i))
                        + synergy[mask])
    return {
        "id": "large",
        "title": "大规模 14 人（排列抽样）",
        "description": "2^14 个联盟无法用精确枚举（精确上限 12 人），使用可复现排列抽样"
                       "估计 Shapley，批均值给出 95% 置信区间；抽样任务中断后可从断点恢复且不重复计数。",
        "game": {"name": "14 人合成", "players": [f"P{i + 1}" for i in range(n)],
                 "entries": _point_rows(n, worths)},
        "expected": {"n": n, "shapley_near": None},
    }


ALL_EXAMPLES = [
    example_symmetry(),
    example_dummy(),
    example_empty_core(),
    example_costs(),
    example_bounded(),
    example_large(),
]


def get_example(example_id: str) -> dict | None:
    for ex in ALL_EXAMPLES:
        if ex["id"] == example_id:
            return ex
    return None
