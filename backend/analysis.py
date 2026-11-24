"""Analysis orchestration: one entry point per user-facing computation."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .allocation import closest_feasible, plan_interval
from .core import core_diagnosis, least_core
from .games import Game, INF
from .shapley import (EXACT_LIMIT, SamplingState, advance_sampling,
                      efficiency_gap, exact_shapley, midpoint_worths,
                      sampling_blocker, shapley_bounds, summarize_sampling)
from .storage import Store, game_fingerprint


def _clean_number(x):
    """JSON-safe numbers: infinities become None (UI renders 无界)."""
    if x is None:
        return None
    if x == INF or x == -INF:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return round(x, 10)
    return x


def _clean_vector(vec):
    return [_clean_number(v) for v in vec]


def run_shapley(game: Game) -> dict:
    """Exact point Shapley when possible; otherwise rigorous intervals."""
    n = len(game.players)
    errors = game.validate()
    if errors:
        return {"ok": False, "errors": errors}
    point = exact_shapley(game) if n <= EXACT_LIMIT else None
    lo, hi = shapley_bounds(game)
    grand_lo, grand_hi = game.grand_worth_bounds()
    result = {
        "ok": True,
        "n": n,
        "mode": "exact" if point is not None else ("interval" if n <= EXACT_LIMIT else "sampling-required"),
        "exact": _clean_vector(point) if point is not None else None,
        "bounds_lo": _clean_vector(lo),
        "bounds_hi": _clean_vector(hi),
        "grand_bounds": [_clean_number(grand_lo), _clean_number(grand_hi)],
        "sampling_blocker": sampling_blocker(game),
    }
    if point is not None:
        gap = efficiency_gap(point, game.point_worth(game.full_mask()))
        result["efficiency_gap"] = _clean_number(gap)
    # efficiency check on intervals
    if all(v > -INF for v in lo) and grand_lo > -INF:
        result["interval_efficiency"] = {
            "sum_lo": _clean_number(sum(lo)),
            "sum_hi": _clean_number(sum(hi)),
            "grand_lo": _clean_number(grand_lo),
            "grand_hi": _clean_number(grand_hi),
            "note": "独立界的外包络通常比大联盟界更宽；点值情形二者严格相等。",
        }
    return result


def run_stability(game: Game) -> dict:
    errors = game.validate()
    if errors:
        return {"ok": False, "errors": errors}
    report = core_diagnosis(game)
    report["ok"] = True
    return report


def run_guaranteed_plan(game: Game, guarantees: List[Optional[float]],
                        budget: Optional[float], target: Optional[List[float]] = None) -> dict:
    """Closest feasible allocation to a Shapley target under floors + budget."""
    errors = game.validate()
    if errors:
        return {"ok": False, "errors": errors}
    # normalize guarantees: pad missing players with "no floor"
    guarantees = list(guarantees) + [None] * max(0, len(game.players) - len(guarantees))
    guarantees = guarantees[:len(game.players)]
    grand_lo, grand_hi = game.grand_worth_bounds()
    if budget is None:
        if grand_lo == grand_hi:
            budget = grand_lo
        elif grand_lo > -INF and grand_hi < INF:
            budget = 0.5 * (grand_lo + grand_hi)
        else:
            return {"ok": False, "errors": ["大联盟价值无界，必须显式提供总额预算。"]}

    point = exact_shapley(game)
    if target is not None:
        phi = [float(x) for x in target]
        target_kind = "custom"
    elif point is not None:
        phi = point
        target_kind = "exact-shapley"
    else:
        mid = midpoint_worths(game)
        if mid is None:
            return {"ok": False,
                    "errors": ["存在无界联盟，无法确定中点目标；请先补全上下界。"]}
        phi = _shapley_from_midpoint(game, mid)
        target_kind = "midpoint-shapley"

    plan = closest_feasible(phi, guarantees, float(budget))
    payload = plan.to_dict(game.players)
    payload["ok"] = True
    payload["target_kind"] = target_kind
    # efficiency conservation on the produced feasible allocation
    payload["budget_residual"] = _clean_number(sum(plan.allocation) - float(budget))

    if point is None:
        lo, hi = shapley_bounds(game)
        if all(v > -INF for v in lo) and all(v < INF for v in hi):
            payload["bounded_plan"] = plan_interval(lo, hi, guarantees, float(budget))
    return payload


def _shapley_from_midpoint(game: Game, worths: Dict[int, float]) -> List[float]:
    from .shapley import _shapley_from_worths
    return _shapley_from_worths(worths, len(game.players))


def resume_or_advance(store: Store, task_id: str, game: Game,
                      additional_permutations: int) -> dict:
    """Advance a sampling task by exactly N new, never-before-counted draws."""
    fingerprint = game_fingerprint(game.to_dict())
    task = store.verify_task(task_id, fingerprint)
    blocker = sampling_blocker(game)
    if blocker:
        return {"ok": False, "errors": [blocker]}
    state = SamplingState.from_dict(task["state"])
    before = state.next_index
    summary = advance_sampling(state, game, additional_permutations)
    if state.next_index != before + additional_permutations:
        raise RuntimeError("排列计数断点不一致，拒绝写入。")
    store.save_task(task_id, state, summary)
    return {"ok": True, "task_id": task_id,
            "consumed_before": before,
            "consumed_after": state.next_index,
            "new_permutations": state.next_index - before,
            "summary": summary}


def task_report(task: dict, n: int) -> dict:
    state = SamplingState.from_dict(task["state"])
    summary = summarize_sampling(state, n)
    return {"id": task["id"], "version_id": task["version_id"],
            "fingerprint": task["fingerprint"],
            "created_at": task["created_at"], **summary}
