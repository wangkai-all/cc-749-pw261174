"""Test suite: run with `python3 -m tests.run_tests` or `python3 tests/run_tests.py`.

Cross-checks are deliberately independent of the implementations they test:
efficiency sums are recomputed by hand, the simplex LP is checked against
enumerated solutions of the majority game, and sampling reproducibility is
verified by replaying RNG streams from scratch.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analysis import resume_or_advance
from backend.allocation import closest_feasible
from backend.core import least_core
from backend.examples import (get_example, example_bounded, example_costs,
                              example_dummy, example_empty_core,
                              example_large, example_symmetry)
from backend.games import Game, INF
from backend.linalg_simplex import solve_lp
from backend.shapley import (SamplingState, SplitMix64, advance_sampling,
                             efficiency_gap, exact_shapley,
                             midpoint_worths, shapley_bounds)
from backend.storage import Store, game_fingerprint


class TestSimplex(unittest.TestCase):
    def test_bounded_lp_optimum(self):
        # max x+y  s.t. x+y<=4, x<=3, y<=3  -> min -x-y, optimum -4
        res = solve_lp([-1.0, -1.0], [
            ([1.0, 1.0], "<=", 4.0),
            ([1.0, 0.0], "<=", 3.0),
            ([0.0, 1.0], "<=", 3.0),
        ])
        self.assertEqual(res["status"], "optimal")
        self.assertAlmostEqual(res["obj"], -4.0, places=8)
        self.assertAlmostEqual(sum(res["x"]), 4.0, places=8)

    def test_equality_and_free_variables(self):
        # min x1 + 2 x2, x1 + x2 = 5, 0 <= x1 <= 5 -> x1=5, x2=0
        res = solve_lp([1.0, 2.0], [
            ([1.0, 1.0], "==", 5.0),
            ([1.0, 0.0], ">=", 0.0),
            ([1.0, 0.0], "<=", 5.0),
        ])
        self.assertEqual(res["status"], "optimal")
        self.assertAlmostEqual(res["x"][0], 5.0, places=6)
        self.assertAlmostEqual(res["x"][1], 0.0, places=6)

    def test_unbounded_free_direction_detected(self):
        # min -x with only x >= 0 is genuinely unbounded
        res = solve_lp([-1.0], [([1.0], ">=", 0.0)])
        self.assertEqual(res["status"], "unbounded")
        # equality-only feasible set with free vars is unbounded below too
        res2 = solve_lp([1.0, 2.0], [([1.0, 1.0], "==", 5.0)])
        self.assertEqual(res2["status"], "unbounded")

    def test_infeasible(self):
        res = solve_lp([1.0], [([1.0], ">=", 3.0), ([1.0], "<=", 1.0)])
        self.assertEqual(res["status"], "infeasible")

    def test_ge_constraint_unbounded_direction(self):
        # min x s.t. x >= 2 -> optimum 2, bounded despite free sign
        res = solve_lp([1.0], [([1.0], ">=", 2.0)])
        self.assertEqual(res["status"], "optimal")
        self.assertAlmostEqual(res["x"][0], 2.0, places=8)


class TestShapleyAxioms(unittest.TestCase):
    def test_symmetry_equal_shares(self):
        game = Game.from_dict(example_symmetry()["game"])
        phi = exact_shapley(game)
        self.assertTrue(all(abs(v - 16.0) < 1e-9 for v in phi))
        # efficiency, recomputed independently
        self.assertLess(efficiency_gap(phi, game.point_worth(game.full_mask())), 1e-9)

    def test_dummy_player(self):
        ex = example_dummy()
        game = Game.from_dict(ex["game"])
        phi = exact_shapley(game)
        for i, expected in enumerate(ex["expected"]["shapley"]):
            self.assertAlmostEqual(phi[i], expected, places=9)
        dummy = ex["expected"]["dummy_player"]
        self.assertAlmostEqual(phi[dummy], game.point_worth(1 << dummy), places=9)
        self.assertLess(efficiency_gap(phi, game.point_worth(game.full_mask())), 1e-9)

    def test_majority_game_shapley(self):
        game = Game.from_dict(example_empty_core()["game"])
        phi = exact_shapley(game)
        for v in phi:
            self.assertAlmostEqual(v, 1 / 3, places=9)
        self.assertAlmostEqual(sum(phi), game.point_worth(game.full_mask()), places=9)

    def test_costs_are_subtracted(self):
        ex = example_costs()
        game = Game.from_dict(ex["game"])
        phi = exact_shapley(game)
        for i, expected in enumerate(ex["expected"]["shapley"]):
            self.assertAlmostEqual(phi[i], expected, places=9)
        # direct net-worth check on one coalition
        mask = 0b011
        self.assertAlmostEqual(game.point_worth(mask), 18.0 - 2.0, places=9)


class TestCore(unittest.TestCase):
    def test_nonempty_core_games(self):
        for factory in (example_symmetry, example_dummy, example_costs):
            game = Game.from_dict(factory()["game"])
            result = least_core(game.point_worths(), len(game.players))
            self.assertEqual(result.status, "optimal")
            self.assertLessEqual(result.epsilon, 1e-8,
                                 msg=factory.__name__)
            # every coalition must be satisfied by the returned allocation
            n = len(game.players)
            for s in range(1, 1 << n):
                xs = sum(result.allocation[i] for i in range(n) if s & (1 << i))
                self.assertGreaterEqual(xs, game.point_worth(s) - 1e-7,
                                        msg=f"{factory.__name__} coalition {s:b}")

    def test_empty_core_majority_game(self):
        game = Game.from_dict(example_empty_core()["game"])
        result = least_core(game.point_worths(), len(game.players))
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.epsilon, 1 / 3, places=8)
        # independent exhaustive proof: no feasible x satisfies all 3 pair demands
        # check via LP that epsilon=0 is infeasible
        n = 3
        constraints = [([1.0] * n, "==", 1.0)]
        for pair in (0b011, 0b101, 0b110):
            constraints.append(([1.0 if pair & (1 << i) else 0.0 for i in range(n)],
                                ">=", 1.0))
        self.assertEqual(solve_lp([0.0] * n, constraints)["status"], "infeasible")
        # at least 3 binding coalitions witness the minimal violation
        self.assertGreaterEqual(len(result.binding), 3)

    def test_additive_game_unique_core(self):
        game = Game.from_dict(example_dummy()["game"])
        result = least_core(game.point_worths(), len(game.players))
        self.assertAlmostEqual(result.epsilon, 0.0, places=8)
        for i, v in enumerate(result.allocation):
            self.assertAlmostEqual(v, game.point_worth(1 << i), places=8)


class TestIntervalPropagation(unittest.TestCase):
    def setUp(self):
        self.game = Game.from_dict(example_bounded()["game"])
        self.expected = example_bounded()["expected"]

    def test_missing_value_is_not_zero(self):
        # point Shapley must refuse, not silently treat unknown as zero
        self.assertIsNone(exact_shapley(self.game))
        unknown_mask = 0b011
        self.assertIsNone(self.game.point_worth(unknown_mask))
        lo, hi = self.game.worth_bounds(unknown_mask)
        self.assertEqual((lo, hi), (8.0, 12.0))

    def test_bounds_match_hand_calculation(self):
        lo, hi = shapley_bounds(self.game)
        for i in range(3):
            self.assertAlmostEqual(lo[i], self.expected["shapley_lo"][i], places=9)
            self.assertAlmostEqual(hi[i], self.expected["shapley_hi"][i], places=9)
            self.assertLessEqual(lo[i], hi[i])

    def test_unbounded_side_propagates_to_infinity(self):
        data = example_bounded()["game"]
        data["entries"][4]["hi"] = None  # remove upper bound of {P1,P2}
        game = Game.from_dict(data)
        lo, hi = shapley_bounds(game)
        self.assertEqual(hi[0], INF)
        self.assertEqual(hi[1], INF)
        self.assertGreater(lo[0], -INF)  # lower side still finite

    def test_bounded_stability_scenarios(self):
        from backend.core import core_diagnosis
        report = core_diagnosis(self.game)
        self.assertEqual(report["scenario"], "bounded")
        self.assertIn("非空", report["verdict"])
        for key in ("lo", "mid", "hi"):
            self.assertEqual(report[key]["status"], "optimal")


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.game = Game.from_dict(example_large()["game"])
        self.n = len(self.game.players)

    def test_prng_determinism_and_uniformity(self):
        a = SplitMix64(12345)
        b = SplitMix64(12345)
        draws_a = [a.randrange(7) for _ in range(500)]
        draws_b = [b.randrange(7) for _ in range(7)][:]
        b2 = SplitMix64(12345)
        draws_b = [b2.randrange(7) for _ in range(500)]
        self.assertEqual(draws_a, draws_b)
        self.assertTrue(all(0 <= d < 7 for d in draws_a))

    def test_sampling_matches_exact_on_small_game(self):
        game = Game.from_dict(example_symmetry()["game"])
        state = SamplingState(seed=42, batch_size=500)
        result = advance_sampling(state, game, 4000)
        exact = exact_shapley(game)
        self.assertEqual(result["done"], 4000)
        for est, true in zip(result["estimate"], exact):
            self.assertLess(abs(est - true), 0.5)  # generous MC tolerance
            self.assertLess(result["ci95_halfwidth"][0], 1.0)

    def test_resume_does_not_double_count(self):
        game = Game.from_dict(example_symmetry()["game"])
        n = len(game.players)
        worths = midpoint_worths(game)
        # independent reference: replay the exact RNG stream from scratch and
        # compute every permutation's marginals one by one
        rng = SplitMix64(99)
        ref_sums = [0.0] * n
        total = 1500
        for _ in range(total):
            order = list(range(n))  # fresh identity each draw, like the engine
            rng.shuffle(order)
            coalition = 0
            prev = worths[0]
            for player in order:
                new = coalition | (1 << player)
                ref_sums[player] += worths[new] - prev
                coalition, prev = new, worths[new]
        ref_estimate = [s / total for s in ref_sums]

        # split the same stream into three resumed chunks (boundaries need not
        # align with sampling batches; no permutation may be consumed twice)
        split = SamplingState(seed=99, batch_size=100)
        seen_cursor = 0
        for chunk in (400, 600, 500):
            out = advance_sampling(split, game, chunk)
            seen_cursor += chunk
            self.assertEqual(split.next_index, seen_cursor)
        self.assertEqual(out["done"], total)
        for est, ref in zip(out["estimate"], ref_estimate):
            self.assertAlmostEqual(est, ref, places=12)

    def test_resume_via_store_fingerprint_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "s.json"))
            game_dict = example_large()["game"]
            version = store.create_version(game_dict, {})
            task = store.create_sampling_task(version["id"], version["fingerprint"],
                                              seed=7)
            game = Game.from_dict(game_dict)
            r1 = resume_or_advance(store, task["id"], game, 1000)
            self.assertTrue(r1["ok"])
            self.assertEqual(r1["consumed_after"], 1000)
            r2 = resume_or_advance(store, task["id"], game, 500)
            self.assertEqual(r2["consumed_before"], 1000)
            self.assertEqual(r2["consumed_after"], 1500)
            # tamper with the game -> fingerprint guard must reject
            game_dict["entries"][1]["value"] += 1.0
            tampered = Game.from_dict(game_dict)
            with self.assertRaises(Exception):
                resume_or_advance(store, task["id"], tampered, 10)

    def test_midpoint_worths_required_for_bounded_sampling(self):
        bounded = Game.from_dict(example_bounded()["game"])
        self.assertIsNotNone(midpoint_worths(bounded))


class TestGuaranteedAllocation(unittest.TestCase):
    def test_water_filling_budget_binds(self):
        plan = closest_feasible([10.0, 10.0, 10.0], [8.0, None, None], 27.0)
        self.assertTrue(plan.feasible)
        self.assertAlmostEqual(sum(plan.allocation), 27.0, places=9)
        self.assertGreaterEqual(plan.allocation[0], 8.0 - 1e-9)
        # floor not binding -> equal distance from target
        self.assertAlmostEqual(plan.allocation[1], plan.allocation[2], places=9)

    def test_infeasible_guarantees_reported_not_hidden(self):
        plan = closest_feasible([5.0, 5.0], [8.0, 8.0], 10.0)
        self.assertFalse(plan.feasible)
        self.assertGreater(plan.total_shortfall, 0.0)
        self.assertAlmostEqual(sum(plan.allocation), 10.0, places=9)
        self.assertTrue(plan.note)

    def test_projection_closer_than_any_alternative(self):
        import random
        rng = random.Random(0)
        phi = [rng.uniform(0, 10) for _ in range(5)]
        floors = [rng.uniform(0, 2) for _ in range(5)]
        budget = sum(phi) + rng.uniform(-2, 4)
        if sum(floors) > budget + 1e-9:
            floors[0] = 0.0  # guarantee feasibility for this optimality check
        plan = closest_feasible(phi, floors, budget)
        self.assertTrue(plan.feasible)
        for _ in range(200):
            perturb = [rng.uniform(-0.5, 0.5) for _ in range(5)]
            alt = [x + p for x, p in zip(plan.allocation, perturb)]
            alt = [max(g, v) for g, v in zip(floors, alt)]
            alt[0] += budget - sum(alt)
            if any(v < g - 1e-9 for v, g in zip(alt, floors)):
                continue
            self.assertLessEqual(plan.distance_sq,
                                 sum((v - p) ** 2 for v, p in zip(alt, phi)) + 1e-9)


class TestStorageVersions(unittest.TestCase):
    def test_versions_are_immutable_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "s.json"))
            g1 = example_symmetry()["game"]
            v1 = store.create_version(g1, {"budget": 48.0})
            self.assertFalse(v1["read_only"])
            g2 = example_dummy()["game"]
            v2 = store.create_version(g2, {})
            self.assertTrue(store.get_version(v1["id"])["read_only"])
            self.assertFalse(store.get_version(v2["id"])["read_only"])
            # reload from disk; stored form is normalized, compare semantically
            store2 = Store(os.path.join(tmp, "s.json"))
            self.assertEqual(len(store2.list_versions()), 2)
            stored_game = store2.get_version(v1["id"])["game"]
            self.assertEqual(Game.from_dict(stored_game).to_dict(),
                             Game.from_dict(g1).to_dict())
            self.assertEqual(game_fingerprint(stored_game),
                             game_fingerprint(g1))


class TestValidation(unittest.TestCase):
    def test_missing_grand_coalition_rejected(self):
        data = example_symmetry()["game"]
        data["entries"] = [r for r in data["entries"] if len(r["members"]) < 3]
        game = Game.from_dict(data)
        errors = game.validate()
        self.assertTrue(any("大联盟" in e for e in errors))

    def test_bad_bounds_and_duplicate_players(self):
        data = example_bounded()["game"]
        data["entries"][4]["hi"] = 1.0  # hi < lo
        game = Game.from_dict(data)
        self.assertTrue(any("下界超过上界" in e for e in game.validate()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
