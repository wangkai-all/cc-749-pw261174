"""Constrained allocation: closest feasible vector to the Shapley contribution.

Given a target vector phi (exact Shapley, sample estimate, or interval
midpoint), per-player minimum guarantees g_i and a total budget B:

    minimize  sum_i (x_i - phi_i)^2
    s.t.      sum_i x_i = B
              x_i >= g_i                (guarantees; NaN/None => no floor)

The KKT solution is water-filling:
    x_i = max(g_i, phi_i - lambda)
with lambda chosen so the budget binds.  Bisection finds lambda exactly.

When sum g_i > B the constraint set is empty.  We never pretend otherwise:
the returned scheme minimizes total guarantee shortfall
    minimize sum_i max(g_i - x_i, 0)
(while keeping the budget), which for squared-distance tie-breaking yields
x_i = max(g_i - delta, phi_i - lambda), and the report exposes every
unavoidable gap so the UI can show which guarantees cannot be honored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .games import INF, TOL

BISECT_TOL = 1e-12
BISECT_MAX_ITERS = 200


def _floored_sum(lam: float, phi: List[float], floors: List[float]) -> float:
    return sum(max(g, p - lam) for g, p in zip(floors, phi))


@dataclass
class FeasiblePlan:
    feasible: bool
    allocation: List[float]
    target: List[float]
    floors: List[float]
    budget: float
    distance_sq: float
    guarantee_gaps: List[float]          # max(g_i - x_i, 0)
    total_shortfall: float               # sum of gaps (0 when feasible)
    lambda_value: float
    note: str = ""

    def to_dict(self, players: List[str]) -> dict:
        return {
            "feasible": self.feasible,
            "allocation": self.allocation,
            "target": self.target,
            # -INF sentinel means "no floor"; render as JSON null rather than
            # a non-standard Infinity token that strict persistence rejects.
            "floors": [None if g == -INF else g for g in self.floors],
            "budget": self.budget,
            "distance_sq": self.distance_sq,
            "distance_l2": math.sqrt(self.distance_sq),
            "guarantee_gaps": dict(zip(players, self.guarantee_gaps)),
            "total_shortfall": self.total_shortfall,
            "lambda": self.lambda_value,
            "note": self.note,
        }


def closest_feasible(phi: List[float], guarantees: List[Optional[float]],
                     budget: float) -> FeasiblePlan:
    n = len(phi)
    floors = [-INF if g is None else float(g) for g in guarantees]
    finite_floors = [g for g in floors if g > -INF]
    sum_floors = sum(finite_floors)

    # ------------------------------------------------ feasible water-filling
    if sum_floors <= budget + TOL:
        # bracket lambda: at lam -> -inf, sum -> +inf-ish (floors bind);
        # at lam -> +inf, sum -> sum floors. Find lo with sum(lo) >= B.
        lo, hi = -1.0, 1.0
        while _floored_sum(lo, phi, floors) < budget:
            lo *= 2.0
            if lo < -1e12:
                break
        while _floored_sum(hi, phi, floors) > budget:
            hi *= 2.0
            if hi > 1e12:
                break
        for _ in range(BISECT_MAX_ITERS):
            mid = 0.5 * (lo + hi)
            if _floored_sum(mid, phi, floors) > budget:
                lo = mid
            else:
                hi = mid
            if hi - lo < BISECT_TOL * max(1.0, abs(lo)):
                break
        lam = 0.5 * (lo + hi)
        x = [max(g, p - lam) for g, p in zip(floors, phi)]
        # repair rounding so the budget binds exactly
        drift = budget - sum(x)
        if abs(drift) > 0.0:
            # push drift into a non-floored player if one exists
            for i in range(n):
                if x[i] > floors[i] + 1e-9:
                    x[i] += drift
                    break
            else:
                x[0] += drift
        gaps = [max(g - xi, 0.0) if g > -INF else 0.0 for g, xi in zip(floors, x)]
        dist_sq = sum((xi - pi) ** 2 for xi, pi in zip(x, phi))
        return FeasiblePlan(True, x, list(phi), floors, budget, dist_sq,
                            gaps, sum(gaps), lam)

    # --------------------------------------------- infeasible: min violation
    # Largest common haircut delta so that sum max(g_i - delta) <= B, then
    # water-fill residual distance with the hair-cut floors.
    lo_d, hi_d = 0.0, max(finite_floors) - min(min(phi), 0.0) + abs(budget) + 1.0
    while sum(max(g - hi_d, 0.0) for g in floors) > budget:
        hi_d *= 2.0
        if hi_d > 1e15:
            break
    for _ in range(BISECT_MAX_ITERS):
        mid = 0.5 * (lo_d + hi_d)
        if sum(max(g - mid, 0.0) for g in floors) > budget:
            lo_d = mid
        else:
            hi_d = mid
        if hi_d - lo_d < BISECT_TOL * max(1.0, abs(hi_d)):
            break
    delta = 0.5 * (lo_d + hi_d)
    relaxed_floors = [max(g - delta, -INF) if g > -INF else -INF for g in floors]
    lo, hi = -1.0, 1.0
    while _floored_sum(lo, phi, relaxed_floors) < budget:
        lo *= 2.0
    for _ in range(BISECT_MAX_ITERS):
        mid = 0.5 * (lo + hi)
        if _floored_sum(mid, phi, relaxed_floors) > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < BISECT_TOL * max(1.0, abs(lo)):
            break
    lam = 0.5 * (lo + hi)
    x = [max(g, p - lam) for g, p in zip(relaxed_floors, phi)]
    drift = budget - sum(x)
    for i in range(n):
        if x[i] > relaxed_floors[i] + 1e-9:
            x[i] += drift
            break
    else:
        x[0] += drift
    gaps = [max(g - xi, 0.0) if g > -INF else 0.0 for g, xi in zip(floors, x)]
    dist_sq = sum((xi - pi) ** 2 for xi, pi in zip(x, phi))
    note = (f"保障总额 {sum_floors:.6g} 超过预算 {budget:.6g}，"
            f"不存在同时满足全部保障与总额约束的方案；以下方案最小化保障违约"
            f"（统一折让 {delta:.6g}），共 {sum(gaps):.6g} 的保障无法兑现。")
    return FeasiblePlan(False, x, list(phi), floors, budget, dist_sq,
                        gaps, sum(gaps), lam, note)


def plan_interval(phi_lo: List[float], phi_hi: List[float],
                  guarantees: List[Optional[float]], budget: float) -> dict:
    """Project both interval endpoint scenarios; the projection is monotone
    in the target, so these bracket every closest allocation as bounds vary."""
    res_lo = closest_feasible(phi_lo, guarantees, budget)
    res_hi = closest_feasible(phi_hi, guarantees, budget)
    return {
        "feasible": res_lo.feasible and res_hi.feasible,
        "allocation_under_low_targets": res_lo.allocation,
        "allocation_under_high_targets": res_hi.allocation,
        "note": res_hi.note or res_lo.note,
        "total_shortfall": max(res_lo.total_shortfall, res_hi.total_shortfall),
    }
