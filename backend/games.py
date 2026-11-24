"""Cooperative game model: players, coalition values, shared costs, bounds.

A coalition is represented internally by a bitmask (int). Coalition *values*
v(S) are the revenue the coalition can realize on its own. *Shared costs* c(S)
are subtracted to produce the net worth w(S) = v(S) - c(S); every Shapley /
core computation below runs on w.

A coalition entry may carry a finite value (point game) or a pair of bounds
(lo <= hi). Missing entries are NEVER silently treated as zero: they remain
explicitly unknown and every downstream routine either refuses them or
propagates the uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

INF = float("inf")
TOL = 1e-9


def subsets_of(mask: int) -> Iterable[int]:
    """Yield every submask of *mask*, including 0 and mask itself."""
    sub = mask
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


def coalition_members(mask: int, players: List[str]) -> List[str]:
    return [players[i] for i in range(len(players)) if mask & (1 << i)]


def mask_from_members(members: Iterable[str], players: List[str]) -> int:
    index = {p: i for i, p in enumerate(players)}
    mask = 0
    for m in members:
        mask |= 1 << index[m]
    return mask


def coalition_label(mask: int, players: List[str]) -> str:
    if mask == 0:
        return "∅"
    return "{" + ", ".join(coalition_members(mask, players)) + "}"


def popcount(mask: int) -> int:
    return bin(mask).count("1")


@dataclass
class CoalitionEntry:
    """One user-supplied row.

    value:  exact net-component input when finite; None if only bounds given.
    lo/hi:  inclusive bounds on the *revenue* v(S) (cost is applied separately).
            lo=-INF / hi=+INF means the side is unspecified.
    cost:   shared cost c(S) >= 0, subtracted to obtain net worth.
    locked: UI marker that the row must not be overwritten by example loads.
    """

    mask: int
    value: Optional[float] = None
    lo: float = -INF
    hi: float = INF
    cost: float = 0.0
    note: str = ""
    locked: bool = False

    def is_point(self) -> bool:
        return self.value is not None

    def is_fully_bounded(self) -> bool:
        return self.lo > -INF and self.hi < INF


@dataclass
class Game:
    players: List[str]
    entries: Dict[int, CoalitionEntry] = field(default_factory=dict)
    name: str = "untitled"

    # ------------------------------------------------------------------ build
    @classmethod
    def from_dict(cls, data: dict) -> "Game":
        players = list(data["players"])
        game = cls(players=players, name=data.get("name", "untitled"))
        for row in data.get("entries", []):
            members = row["members"]
            mask = mask_from_members(members, players)
            value = row.get("value")
            lo = row.get("lo", -INF)
            hi = row.get("hi", INF)
            # JSON cannot carry infinities; null / missing => unbounded side.
            if lo is None:
                lo = -INF
            if hi is None:
                hi = INF
            entry = CoalitionEntry(
                mask=mask,
                value=None if value is None else float(value),
                lo=float(lo),
                hi=float(hi),
                cost=float(row.get("cost", 0.0)),
                note=row.get("note", ""),
                locked=bool(row.get("locked", False)),
            )
            game.entries[mask] = entry
        # Empty coalition is normalized to zero net worth.
        game.entries.setdefault(
            0, CoalitionEntry(mask=0, value=0.0, lo=0.0, hi=0.0)
        )
        return game

    def to_dict(self) -> dict:
        rows = []
        for mask, e in sorted(self.entries.items(), key=lambda kv: (popcount(kv[0]), kv[0])):
            row = {
                "members": coalition_members(mask, self.players),
                "value": e.value,
                "lo": None if e.lo == -INF else e.lo,
                "hi": None if e.hi == INF else e.hi,
                "cost": e.cost,
                "note": e.note,
                "locked": e.locked,
            }
            rows.append(row)
        return {"name": self.name, "players": list(self.players), "entries": rows}

    # ------------------------------------------------------------- net worths
    def full_mask(self) -> int:
        return (1 << len(self.players)) - 1

    def point_worth(self, mask: int) -> Optional[float]:
        """Finite net worth w(S)=v(S)-c(S), or None if unknown / interval."""
        e = self.entries.get(mask)
        if e is None or not e.is_point():
            return None
        return e.value - e.cost

    def worth_bounds(self, mask: int) -> Tuple[float, float]:
        """Net worth interval [w_lo, w_hi]. Missing entry => (-INF, INF)."""
        e = self.entries.get(mask)
        if e is None:
            return (-INF, INF)
        cost = e.cost
        if e.is_point():
            v = e.value
            return (v - cost, v - cost)
        return (e.lo - cost if e.lo > -INF else -INF,
                e.hi - cost if e.hi < INF else INF)

    def point_worths(self) -> Dict[int, float]:
        out = {}
        for mask in self.entries:
            w = self.point_worth(mask)
            if w is not None:
                out[mask] = w
        return out

    def is_point_game(self) -> bool:
        """True only when EVERY coalition (incl. grand) has an exact value."""
        n_masks = 1 << len(self.players)
        for mask in range(n_masks):
            if self.point_worth(mask) is None:
                return False
        return True

    def grand_worth_bounds(self) -> Tuple[float, float]:
        return self.worth_bounds(self.full_mask())

    def known_masks(self) -> List[int]:
        return [m for m in self.entries if m != 0]

    def validate(self) -> List[str]:
        """Structural sanity checks; returns a list of human-readable errors."""
        errors: List[str] = []
        if not self.players:
            errors.append("游戏至少需要一个参与方。")
        if len(set(self.players)) != len(self.players):
            errors.append("参与方名称必须唯一。")
        grand = self.full_mask()
        if grand not in self.entries:
            errors.append("缺少大联盟（全体参与方）的价值，总额约束无法确定。")
        for mask, e in self.entries.items():
            if mask == 0:
                if not e.is_point() or abs(e.value) > TOL:
                    errors.append("空联盟价值必须恒为 0。")
                continue
            if e.cost < -TOL:
                errors.append(f"{coalition_label(mask, self.players)} 的共享成本不能为负。")
            if e.is_point():
                if e.lo > -INF and e.lo > e.value + TOL:
                    errors.append(f"{coalition_label(mask, self.players)} 下界高于精确值。")
                if e.hi < INF and e.hi < e.value - TOL:
                    errors.append(f"{coalition_label(mask, self.players)} 上界低于精确值。")
            else:
                if e.lo > -INF and e.hi < INF and e.lo > e.hi + TOL:
                    errors.append(f"{coalition_label(mask, self.players)} 下界超过上界。")
        return errors
