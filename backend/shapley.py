"""Shapley values: exact enumeration, interval propagation, permutation sampling.

Exact path (small games, n <= EXACT_LIMIT):
    phi_i = sum_{S subseteq N\\{i}} |S|!(n-|S|-1)!/n! * (w(S+i) - w(S))

Interval path: a coalition with only bounds [lo, hi] is never replaced by
zero.  Each marginal is itself an interval
    [ w_lo(S+i) - w_hi(S),  w_hi(S+i) - w_lo(S) ]
and the weighted intervals are summed.  Any unbounded side propagates to an
infinite Shapley bound, which the UI renders explicitly as "无界".

Sampling path (large point / fully-bounded games): random permutations, each
player's marginal contribution in a permutation; batch means give a
reproducible 95% confidence interval.  A deterministic SplitMix64 stream keyed
by a consumed-permutation counter makes resumed tasks produce exactly the same
permutations they would have produced in one run — permutations are never
double-counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .games import Game, INF, TOL, popcount

EXACT_LIMIT = 12  # 2^12 = 4096 coalitions, enumeration stays instant

# Two-sided 95% Student-t critical values for batch-means degrees of freedom.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z_CRIT_95 = 1.96


# --------------------------------------------------------------------- PRNG
class SplitMix64:
    """Deterministic 64-bit PRNG (Steele/Lea). Independent of Python's hash seed."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def randrange(self, n: int) -> int:
        """Uniform integer in [0, n) via Lemire's unbiased bounded method."""
        threshold = (-n) % n
        while True:
            x = self.next_u64()
            m = x * n
            low = m & self.MASK
            if low >= threshold:
                return m >> 64

    def shuffle(self, items: List[int]) -> List[int]:
        for i in range(len(items) - 1, 0, -1):
            j = self.randrange(i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    def rewind(self, n_players: int, permutations_consumed: int) -> None:
        """Reposition the stream after `permutations_consumed` full shuffles.

        Each permutation consumes exactly n_players randrange() draws, so the
        resume cursor is measured in permutations (never in u64 words, which
        would desynchronize the stream). Replaying draws is deterministic.
        """
        scratch = list(range(n_players))
        for _ in range(permutations_consumed):
            self.shuffle(scratch)


# ------------------------------------------------------------- exact Shapley
def exact_shapley(game: Game) -> Optional[List[float]]:
    """Point Shapley vector, or None if any required coalition value is unknown."""
    n = len(game.players)
    worths: Dict[int, float] = {}
    for mask in range(1 << n):
        w = game.point_worth(mask)
        if w is None:
            return None
        worths[mask] = w
    return _shapley_from_worths(worths, n)


def _shapley_from_worths(worths: Dict[int, float], n: int) -> List[float]:
    factorials = [math.factorial(k) for k in range(n + 1)]
    total_perms = factorials[n]
    phi = [0.0] * n
    for i in range(n):
        bit = 1 << i
        others = ((1 << n) - 1) ^ bit
        sub = others
        while True:
            size = popcount(sub)
            weight = factorials[size] * factorials[n - size - 1] / total_perms
            marginal = worths[sub | bit] - worths[sub]
            phi[i] += weight * marginal
            if sub == 0:
                break
            sub = (sub - 1) & others
    return [0.0 if abs(v) < TOL else v for v in phi]


def shapley_bounds(game: Game) -> Tuple[List[float], List[float]]:
    """Rigorous Shapley intervals under independent coalition worth bounds."""
    n = len(game.players)
    factorials = [math.factorial(k) for k in range(n + 1)]
    total_perms = factorials[n]
    lo_vec = [0.0] * n
    hi_vec = [0.0] * n
    for i in range(n):
        bit = 1 << i
        others = ((1 << n) - 1) ^ bit
        sub = others
        while True:
            size = popcount(sub)
            weight = factorials[size] * factorials[n - size - 1] / total_perms
            a_lo, a_hi = game.worth_bounds(sub | bit)
            b_lo, b_hi = game.worth_bounds(sub)
            if a_lo == -INF or b_hi == INF:
                lo_vec[i] = -INF
            elif lo_vec[i] != -INF:
                lo_vec[i] += weight * (a_lo - b_hi)
            if a_hi == INF or b_lo == -INF:
                hi_vec[i] = INF
            elif hi_vec[i] != INF:
                hi_vec[i] += weight * (a_hi - b_lo)
            if sub == 0:
                break
            sub = (sub - 1) & others
    return lo_vec, hi_vec


# ------------------------------------------------------------- midpoint game
def midpoint_worths(game: Game) -> Optional[Dict[int, float]]:
    """Worths at bound midpoints; None if any coalition is unbounded/unknown."""
    n = len(game.players)
    out: Dict[int, float] = {}
    for mask in range(1 << n):
        w = game.point_worth(mask)
        if w is not None:
            out[mask] = w
            continue
        lo, hi = game.worth_bounds(mask)
        if lo == -INF or hi == INF:
            return None
        out[mask] = 0.5 * (lo + hi)
    return out


def sampling_blocker(game: Game) -> Optional[str]:
    """Why sampling cannot run, or None if it can."""
    n = len(game.players)
    missing = []
    for mask in range(1 << n):
        w = game.point_worth(mask)
        if w is not None:
            continue
        lo, hi = game.worth_bounds(mask)
        if lo == -INF or hi == INF:
            label = "空联盟" if mask == 0 else (
                "{" + ", ".join(game.players[k] for k in range(n)
                                 if mask & (1 << k)) + "}")
            missing.append(label)
    if missing:
        return "以下联盟既无精确值也无完整上下界，无法抽样：" + "、".join(missing)
    return None


# ------------------------------------------------------------- sampling task
@dataclass
class BatchStats:
    means: List[float]          # per-player mean marginal within the batch
    size: int                   # permutations consumed in this batch


@dataclass
class SamplingState:
    seed: int
    next_index: int = 0         # permutations already consumed (resume cursor)
    batches: List[BatchStats] = field(default_factory=list)
    batch_size: int = 200
    rng_state: Optional[int] = None  # persisted SplitMix64 state for O(1) resume
    running_sums: Optional[List[float]] = None  # exact cumulative marginals

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "next_index": self.next_index,
            "batch_size": self.batch_size,
            "rng_state": self.rng_state,
            "running_sums": self.running_sums,
            "batches": [{"means": b.means, "size": b.size} for b in self.batches],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SamplingState":
        return cls(
            seed=int(data["seed"]),
            next_index=int(data["next_index"]),
            batch_size=int(data.get("batch_size", 200)),
            rng_state=data.get("rng_state"),
            running_sums=(None if data.get("running_sums") is None
                          else list(data["running_sums"])),
            batches=[BatchStats(means=list(b["means"]), size=int(b["size"]))
                     for b in data.get("batches", [])],
        )

    def make_rng(self, n_players: int) -> SplitMix64:
        """Restore the stream at the resume cursor in O(1) when a state is
        persisted, otherwise replay deterministically from the seed."""
        rng = SplitMix64(self.seed)
        if self.rng_state is not None:
            rng.state = self.rng_state
        else:
            rng.rewind(n_players, self.next_index)
        return rng


def _permutation_marginals(order: List[int], worths: Dict[int, float],
                           n: int) -> List[float]:
    marginal = [0.0] * n
    coalition = 0
    prev_worth = worths[0]
    for player in order:
        new_coalition = coalition | (1 << player)
        marginal[player] = worths[new_coalition] - prev_worth
        coalition = new_coalition
        prev_worth = worths[new_coalition]
    return marginal


def advance_sampling(state: SamplingState, game: Game,
                     additional_permutations: int) -> dict:
    """Consume exactly `additional_permutations` NEW permutations.

    The RNG is repositioned at state.next_index on every call, so resuming a
    task replays the stream position instead of reusing or repeating draws.
    Returns a result dict with point estimate, SE, 95% CI half-width and
    batch count.
    """
    n = len(game.players)
    worths = midpoint_worths(game)
    if worths is None:
        raise ValueError(sampling_blocker(game) or "抽样前提不满足。")

    rng = state.make_rng(n)
    if state.running_sums is None:
        state.running_sums = [0.0] * n

    target = state.next_index + int(additional_permutations)
    batch_sums = [0.0] * n
    batch_count = 0
    while state.next_index < target:
        # always start from the identity permutation: the RNG stream alone
        # determines each draw, so resumed and one-shot runs stay identical
        order = list(range(n))
        rng.shuffle(order)
        marginals = _permutation_marginals(order, worths, n)
        for i in range(n):
            batch_sums[i] += marginals[i]
            state.running_sums[i] += marginals[i]
        batch_count += 1
        state.next_index += 1
        if batch_count == state.batch_size or state.next_index == target:
            means = [s / batch_count for s in batch_sums]
            state.batches.append(BatchStats(means=means, size=batch_count))
            batch_sums = [0.0] * n
            batch_count = 0
    state.rng_state = rng.state  # persist exact stream position
    return summarize_sampling(state, n)


def summarize_sampling(state: SamplingState, n: int) -> dict:
    batches = state.batches
    done = state.next_index
    if not batches:
        return {"done": 0, "estimate": [None] * n, "se": [None] * n,
                "ci95_halfwidth": [None] * n, "batch_count": 0,
                "batch_size": state.batch_size}
    B = len(batches)
    # point estimate from the exact cumulative sums: unbiased regardless of
    # how permutations were split across resumes or final partial batches
    estimate = ([s / done for s in state.running_sums]
                if state.running_sums is not None else
                [sum(b.means[i] * b.size for b in batches) / done for i in range(n)])
    if B == 1:
        # With a single batch no variance is observable; report SE as unknown.
        return {"done": done, "estimate": estimate, "se": [None] * n,
                "ci95_halfwidth": [None] * n, "batch_count": 1,
                "batch_size": state.batch_size, "note": "单批次无法估计误差，继续追加排列。"}
    # Batch means estimate the variance of the per-batch Shapley estimator.
    se = [0.0] * n
    half = [0.0] * n
    for i in range(n):
        values = [b.means[i] for b in batches]
        mean = sum(values) / B
        var = sum((v - mean) ** 2 for v in values) / (B - 1)
        se[i] = math.sqrt(var / B)
        crit = _T_CRIT_95.get(B - 1, _Z_CRIT_95)
        half[i] = crit * se[i]
    return {"done": done, "estimate": estimate, "se": se,
            "ci95_halfwidth": half, "batch_count": B,
            "batch_size": state.batch_size}


def efficiency_gap(phi: List[float], grand_worth: float) -> float:
    """Absolute violation of the efficiency axiom sum(phi) == w(N)."""
    return abs(sum(phi) - grand_worth)
