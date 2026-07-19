"""Fast regression and property tests for the fixed-cap Sat-RSH model."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from sat_rsh_model import (
    cap_angle,
    cap_chord_radius,
    generate_realization,
    generate_size_matched_null,
    retained_attempt_probabilities,
    shadow_graph,
    sphere_points,
    wilson_interval,
)


class SatRSHModelTests(unittest.TestCase):
    def test_sphere_points_are_unit_norm(self) -> None:
        points = sphere_points(1000, np.random.default_rng(7))
        self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1.0))

    def test_cap_chord_matches_central_angle(self) -> None:
        theta = cap_angle(5, 500)
        self.assertAlmostEqual(cap_chord_radius(5, 500), 2.0 * np.sin(theta / 2.0))

    def test_retained_probability_definition(self) -> None:
        result = retained_attempt_probabilities(500, 2.0, 5)
        self.assertAlmostEqual(float(np.sum(result["conditional"])), 1.0)
        self.assertAlmostEqual(
            float(np.sum(result["unconditional"])),
            float(result["retain_probability"]),
        )
        self.assertGreater(float(result["retain_probability"]), 0.0)
        self.assertLess(float(result["retain_probability"]), 1.0)

    def test_fixed_seed_is_deterministic(self) -> None:
        first = generate_realization(100, 200, 2.0, 5, 12345)
        second = generate_realization(100, 200, 2.0, 5, 12345)
        self.assertTrue(np.array_equal(first.points, second.points))
        self.assertTrue(np.array_equal(first.targets, second.targets))
        self.assertEqual(first.unique_edges, second.unique_edges)

    def test_prefixes_are_nested(self) -> None:
        realization = generate_realization(100, 500, 2.0, 5, 54321)
        prefixes = [realization.prefix(value) for value in (50, 100, 200, 500)]
        counts = [len(prefix.unique_edges) for prefix in prefixes]
        self.assertEqual(counts, sorted(counts))
        for earlier, later in zip(prefixes, prefixes[1:]):
            self.assertTrue(set(earlier.unique_edges).issubset(later.unique_edges))

        connected = [
            nx.is_connected(shadow_graph(100, prefix.unique_edges))
            for prefix in prefixes
        ]
        for earlier, later in zip(connected, connected[1:]):
            self.assertFalse(earlier and not later)

    def test_size_matched_control_is_exact_and_simple(self) -> None:
        sizes = [2, 2, 3, 4, 5, 5]
        edges = generate_size_matched_null(100, sizes, 98765)
        self.assertEqual([len(edge) for edge in edges], sizes)
        self.assertEqual(len(edges), len(set(edges)))

    def test_wilson_interval_contains_estimate(self) -> None:
        low, high = wilson_interval(25, 80)
        self.assertLessEqual(low, 25 / 80)
        self.assertGreaterEqual(high, 25 / 80)


if __name__ == "__main__":
    unittest.main()
