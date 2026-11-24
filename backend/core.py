"""Core stability: feasibility, least core via constraint generation.

For a point game with worths w(S), an imputation x satisfies
    sum_i x_i = w(N),   x_i >= w({i})                      (individual rationality)
and the core additionally requires
    sum_{i in S} x_i >= w(S)      for every coalition S.

The least core relaxes every coalition constraint by a common epsilon:
    sum_{i in S} x_i >= w(S) - epsilon   for all S
and minimizes epsilon.  epsilon* <= 0 means the core is non-empty
(=0: tight core; <0: slack); epsilon* > 0 means the core is empty and epsilon*
is the smallest uniform subsidy needed — never reported as "already fair".

All LPs go through the standard-library simplex solver.  Constraint
generation starts from the singleton + grand-coalition constraints and only
adds violated rows, so large games pay for known coalitions rather than 2^n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .games import Game, INF, TOL, coalition_label, popcount
from .linalg_simplex import solve_lp

CG_GAP_TOL = 1e-8
MAX_CG_ITERS = 1000


@dataclass
class LeastCoreResult:
    status: str                       # "optimal" | "infeasible" | ...
    epsilon: Optional[float]
    allocation: Optional[List[float]]
    binding: List[dict]               # coalitions tight at the optimum
    iterations: int
    coalition_count: int              # how many coalition constraints exist
    used_known_only: bool             # True => large game, unknown S uncertified

    def to_dict(self, players: List[str]) -> dict:
        return {
            "status": self.status,
            "epsilon": self.epsilon,
            "allocation": self.allocation,
            "binding": [
                {"mask": b["mask"],
                 "members": [players[i] for i in range(len(players))
                             if b["mask"] & (1 << i)],
                 "worth": b["worth"],
                 "allocated": b["allocated"],
                 "deficit": b["deficit"]}
                for b in self.binding
            ],
            "iterations": self.iterations,
            "coalition_count": self.coalition_count,
            "used_known_only": self.used_known_only,
            "core_nonempty": (self.epsilon is not None and self.epsilon <= CG_GAP_TOL),
        }


def _worst_deficit(allocation: List[float], worths: Dict[int, float],
                   n: int, candidate_masks: List[int]
                   ) -> Tuple[float, List[int]]:
    """Return (max_S [w(S) - x(S)], tight masks)."""
    sums: Dict[int, float] = {}
    full = (1 << n) - 1
    # prefix sums over subsets via DP: xsum[mask]
    xsum = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        xsum[mask] = xsum[mask ^ low] + allocation[low.bit_length() - 1]
    worst = -INF
    tight: List[int] = []
    for s in candidate_masks:
        if s == 0 or s == full:
            continue
        d = worths[s] - xsum[s]
        if d > worst + CG_GAP_TOL:
            worst = d
            tight = [s]
        elif abs(d - worst) <= CG_GAP_TOL:
            tight.append(s)
    return worst, tight


def least_core(worths: Dict[int, float], n: int,
               candidate_masks: Optional[List[int]] = None) -> LeastCoreResult:
    """Compute the least core of an explicit (point) worth function.

    For small games every coalition is enumerated, so the returned epsilon* is
    the true optimum (negative when the core has slack).  For large games only
    known coalitions enter the LP and the result is explicitly labelled
    `used_known_only` — an uncertified remainder is never hidden.
    """
    full = (1 << n)
    full_mask = full - 1
    if full_mask not in worths:
        return LeastCoreResult("infeasible", None, None, [], 0, 0, False)

    if candidate_masks is None:
        candidate_masks = [m for m in range(full) if m != 0]
        used_known_only = False
    else:
        candidate_masks = sorted(set(candidate_masks) | {full_mask})
        used_known_only = True
    coalition_count = len([m for m in candidate_masks if m != 0 and m != full_mask])

    # LP variables: x_0, ..., x_{n-1}, epsilon (free; solver handles signs).
    # Small games: one full LP over every coalition constraint.
    if not used_known_only:
        constraints = [([1.0] * n + [0.0], "==", worths[full_mask])]
        for s in candidate_masks:
            if s == full_mask:
                continue
            row = [1.0 if s & (1 << i) else 0.0 for i in range(n)] + [1.0]
            constraints.append((row, ">=", worths[s]))
        res = solve_lp([0.0] * n + [1.0], constraints)  # minimize epsilon
        if res["status"] != "optimal":
            return LeastCoreResult(res["status"], None, None, [],
                                   1, coalition_count, used_known_only)
        allocation = res["x"][:n]
        epsilon = res["x"][n]
        binding = []
        for s in candidate_masks:
            if s == full_mask:
                continue
            xs = sum(allocation[i] for i in range(n) if s & (1 << i))
            deficit = worths[s] - xs
            if abs(deficit - epsilon) <= 1e-6:
                binding.append({"mask": s, "worth": worths[s],
                                "allocated": xs, "deficit": deficit})
        return LeastCoreResult("optimal", epsilon, allocation, binding,
                               1, coalition_count, used_known_only)

    # Large games: constraint generation over the known coalitions.
    active: List[int] = [full_mask] + [1 << i for i in range(n)]
    active = sorted(set(active))
    allocation = None
    epsilon = None
    iterations = 0
    for iterations in range(1, MAX_CG_ITERS + 1):
        constraints = [([1.0] * n + [0.0], "==", worths[full_mask])]
        for s in active:
            if s == full_mask:
                continue
            row = [1.0 if s & (1 << i) else 0.0 for i in range(n)] + [1.0]
            constraints.append((row, ">=", worths[s]))
        res = solve_lp([0.0] * n + [1.0], constraints)
        if res["status"] != "optimal":
            return LeastCoreResult(res["status"], None, None, [],
                                   iterations, coalition_count, used_known_only)
        allocation = res["x"][:n]
        epsilon = res["x"][n]

        worst, tight = _worst_deficit(allocation, worths, n, candidate_masks)
        if worst <= epsilon + CG_GAP_TOL:
            binding = []
            for s in active:
                if s == full_mask:
                    continue
                xs = sum(allocation[i] for i in range(n) if s & (1 << i))
                deficit = worths[s] - xs
                if abs(deficit - epsilon) <= 1e-6:
                    binding.append({
                        "mask": s, "worth": worths[s],
                        "allocated": xs, "deficit": deficit,
                    })
            return LeastCoreResult("optimal", epsilon, allocation, binding,
                                   iterations, coalition_count, used_known_only)
        for s in tight:
            if s not in active:
                active.append(s)
    raise RuntimeError("约束生成超过最大迭代次数。")


def core_diagnosis(game: Game) -> dict:
    """Full stability report. Bounded games get lo/mid/hi scenario runs."""
    n = len(game.players)
    full_mask = game.full_mask()
    known = game.known_masks()
    point_worths = game.point_worths()

    if game.is_point_game():
        if n <= 12:
            result = least_core(point_worths, n)
            report = result.to_dict(game.players)
            report["scenario"] = "point"
        else:
            result = least_core(point_worths, n, candidate_masks=known)
            report = result.to_dict(game.players)
            report["scenario"] = "point-known-only"
            report["warning"] = (
                f"参与方较多，仅校验了 {len(known)} 个已录入联盟；"
                "未录入联盟的稳定性无法保证。")
        report["unstable_coalitions"] = _named_unstable(
            _unstable_ranking(point_worths, result.allocation, n), game.players)
        return report

    # ----- bounded / partially specified game: three scenario games
    lo_worths: Dict[int, float] = {}
    hi_worths: Dict[int, float] = {}
    mid_worths: Dict[int, float] = {}
    unbounded_side = []
    for mask in range(1 << n):
        lo, hi = game.worth_bounds(mask)
        pw = game.point_worth(mask)
        if pw is not None:
            lo_worths[mask] = hi_worths[mask] = mid_worths[mask] = pw
        else:
            if lo == -INF or hi == INF:
                unbounded_side.append(coalition_label(mask, game.players))
                continue
            lo_worths[mask], hi_worths[mask] = lo, hi
            mid_worths[mask] = 0.5 * (lo + hi)
    if full_mask not in lo_worths:
        return {"scenario": "bounded", "status": "incomplete",
                "message": "大联盟价值缺少完整上下界，无法进行核心分析。",
                "unbounded_coalitions": unbounded_side}

    # Only coalitions present in every scenario can enter the LP; unbounded
    # ones are reported separately, never silently dropped from candidates.
    candidates = sorted((set(lo_worths) | {full_mask}) & set(hi_worths)
                        & set(mid_worths) & (set(known) | {full_mask}))
    res_lo = least_core(lo_worths, n, candidate_masks=candidates)
    res_hi = least_core(hi_worths, n, candidate_masks=candidates)
    res_mid = least_core(mid_worths, n, candidate_masks=candidates)

    eps_lo, eps_hi = res_lo.epsilon, res_hi.epsilon
    if eps_lo is None or eps_hi is None:
        verdict = "无法判定"
    elif eps_hi <= CG_GAP_TOL:
        verdict = "非空（所有界情形下核心均非空）"
    elif eps_lo > CG_GAP_TOL:
        verdict = "为空（所有界情形下核心均为空）"
    else:
        verdict = "取决于未知价值的实现（界内存在空核与非空核两种可能）"
    if unbounded_side:
        verdict += "；存在无界联盟，结论仅覆盖有完整界的联盟。"
    return {
        "scenario": "bounded",
        "verdict": verdict,
        "epsilon_interval": [eps_lo, eps_hi],
        "epsilon_midpoint": res_mid.epsilon,
        "allocation_midpoint": res_mid.allocation,
        "lo": res_lo.to_dict(game.players),
        "hi": res_hi.to_dict(game.players),
        "mid": res_mid.to_dict(game.players),
        "unbounded_coalitions": unbounded_side,
        "used_known_only": True,
        "unstable_coalitions": _named_unstable(
            _unstable_ranking(mid_worths, res_mid.allocation, n,
                              candidate_masks=candidates), game.players),
    }


def _named_unstable(rows: List[dict], players: List[str]) -> List[dict]:
    """Normalize unstable-coalition rows to player-name member lists so the
    point and bounded paths share one wire format."""
    out = []
    for r in rows:
        r = dict(r)
        r["members"] = [players[i] for i in r["members"]]
        out.append(r)
    return out


def _unstable_ranking(worths: Dict[int, float], allocation: Optional[List[float]],
                      n: int, candidate_masks: Optional[List[int]] = None) -> List[dict]:
    """Coalitions ranked by how much their worth exceeds what they are paid."""
    if allocation is None:
        return []
    if candidate_masks is None:
        candidate_masks = [m for m in range(1, 1 << n)]
    full_mask = (1 << n) - 1
    xsum = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        xsum[mask] = xsum[mask ^ low] + allocation[low.bit_length() - 1]
    rows = []
    for s in candidate_masks:
        if s == 0 or s == full_mask or s not in worths:
            continue
        deficit = worths[s] - xsum[s]
        rows.append({
            "mask": s,
            "members": [i for i in range(n) if s & (1 << i)],
            "worth": worths[s],
            "allocated": xsum[s],
            "deficit": deficit,
            "blocking": deficit > CG_GAP_TOL,
        })
    rows.sort(key=lambda r: -r["deficit"])
    return rows
