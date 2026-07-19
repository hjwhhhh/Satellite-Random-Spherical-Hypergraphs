"""Reproducible fixed-cap Satellite Random Spherical Hypergraph model.

This module implements the model stated in the accompanying manuscript:

1. sample ground points and satellite nadir directions uniformly on S^2;
2. draw a target footprint size S from a truncated Zipf law;
3. use a deterministic cap area S/n, so the raw captured count X satisfies
   X | S=k ~ Binomial(n, k/n) marginally;
4. discard attempts with X<2, truncate attempts with X>s_max to the closest
   s_max points, and remove duplicate hyperedges.

The distinction between attempted, retained-before-deduplication, and unique
hyperedges is intentional and is used throughout the accompanying experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import shortest_path
from scipy.stats import binom, poisson


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "gray": "#6B7280",
}

GAMMA_COLORS = {
    1.5: OKABE_ITO["vermillion"],
    2.0: OKABE_ITO["blue"],
    2.5: OKABE_ITO["bluish_green"],
    3.0: OKABE_ITO["reddish_purple"],
    3.5: OKABE_ITO["orange"],
    4.0: OKABE_ITO["gray"],
}
GAMMA_MARKERS = {1.5: "s", 2.0: "o", 2.5: "^", 3.0: "D", 3.5: "v", 4.0: "P"}
SIZE_COLORS = {
    2: OKABE_ITO["blue"],
    3: OKABE_ITO["vermillion"],
    4: OKABE_ITO["bluish_green"],
    5: OKABE_ITO["reddish_purple"],
}
SIZE_MARKERS = {2: "o", 3: "s", 4: "^", 5: "D"}


def publication_style() -> dict:
    """Nature-style defaults for figures exported at final publication size.

    The main multi-panel figures are 183 mm wide.  The manuscript scales them
    slightly when embedding, leaving ordinary lettering within the 5--7 pt
    final-size range and panel letters at approximately 8 pt.
    """
    return {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def sphere_points(count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ``count`` independent uniform points on the unit sphere."""
    raw = rng.standard_normal((count, 3))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def zipf_pmf(gamma: float, s_max: int, s_min: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive exponent")
    if not isinstance(s_min, (int, np.integer)) or not isinstance(s_max, (int, np.integer)):
        raise TypeError("s_min and s_max must be integers")
    if s_min < 1 or s_max < s_min:
        raise ValueError("require 1 <= s_min <= s_max")
    sizes = np.arange(s_min, s_max + 1, dtype=int)
    weights = sizes.astype(float) ** (-gamma)
    return sizes, weights / weights.sum()


def cap_angle(target_size: int, n: int) -> float:
    """Central angular radius whose surface fraction is target_size/n."""
    if not isinstance(target_size, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise TypeError("target_size and n must be integers")
    if not 0 < target_size < n:
        raise ValueError("require 0 < target_size < n")
    return float(np.arccos(1.0 - 2.0 * target_size / n))


def cap_chord_radius(target_size: int, n: int) -> float:
    """Euclidean chord radius corresponding to :func:`cap_angle`."""
    if not isinstance(target_size, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise TypeError("target_size and n must be integers")
    if not 0 < target_size < n:
        raise ValueError("require 0 < target_size < n")
    return float(2.0 * np.sqrt(target_size / n))


def retained_attempt_probabilities(
    n: int, gamma: float, s_max: int, s_min: int = 2
) -> dict[str, np.ndarray | float]:
    """Exact one-attempt distribution before duplicate removal.

    Conditional on target S=k, the raw number of points in the cap is
    X~Binomial(n,k/n).  Attempts with X<2 are discarded and attempts with
    X>s_max are clipped to s_max.  Returned ``unconditional`` masses sum to
    the probability that an attempt is retained; ``conditional`` is the size
    PMF conditional on retention.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if 2 * s_max >= n:
        raise ValueError("the satellite interpretation requires s_max < n/2")
    target_sizes, target_pmf = zipf_pmf(gamma, s_max, s_min)
    retained_sizes = np.arange(2, s_max + 1, dtype=int)
    masses = np.zeros(len(retained_sizes), dtype=float)
    raw_mean = 0.0
    raw_second_falling = 0.0

    for target, weight in zip(target_sizes, target_pmf):
        probability = target / n
        raw_mean += weight * target
        raw_second_falling += weight * n * (n - 1) * probability**2
        for index, size in enumerate(retained_sizes[:-1]):
            masses[index] += weight * binom.pmf(size, n, probability)
        masses[-1] += weight * binom.sf(s_max - 1, n, probability)

    retain_probability = float(masses.sum())
    conditional = masses / retain_probability
    incidence_first = float(retained_sizes @ masses)
    incidence_second_falling = float((retained_sizes * (retained_sizes - 1)) @ masses)
    return {
        "sizes": retained_sizes,
        "target_pmf": target_pmf,
        "unconditional": masses,
        "conditional": conditional,
        "retain_probability": retain_probability,
        "retained_incidence_mean": incidence_first,
        "retained_pair_incidence_mean": incidence_second_falling,
        "raw_count_mean": float(raw_mean),
        "raw_count_second_falling": float(raw_second_falling),
    }


@dataclass
class Realization:
    points: np.ndarray
    anchors: np.ndarray
    targets: np.ndarray
    raw_counts: np.ndarray
    retained_by_attempt: list[frozenset[int] | None]
    retained_attempts: list[frozenset[int]]
    unique_edges: list[frozenset[int]]

    @property
    def retained_sizes(self) -> np.ndarray:
        return np.asarray([len(edge) for edge in self.retained_attempts], dtype=int)

    @property
    def unique_sizes(self) -> np.ndarray:
        return np.asarray([len(edge) for edge in self.unique_edges], dtype=int)

    @property
    def unique_fraction_of_attempts(self) -> float:
        return len(self.unique_edges) / len(self.targets)

    @property
    def unique_fraction_of_retained(self) -> float:
        if not self.retained_attempts:
            return np.nan
        return len(self.unique_edges) / len(self.retained_attempts)

    def prefix(self, attempts: int) -> "Realization":
        """Return the hypergraph induced by the first ``attempts`` satellites.

        Ground points and the maximum attempt sequence are shared across
        prefixes.  This provides an exact coupling across attempt densities:
        adding a prefix can add a new unique hyperedge but can never remove an
        existing one.
        """
        if not 0 <= attempts <= len(self.targets):
            raise ValueError("attempts must lie between zero and the generated maximum")
        retained_by_attempt = self.retained_by_attempt[:attempts]
        retained_attempts = [edge for edge in retained_by_attempt if edge is not None]
        unique_edges = list(dict.fromkeys(retained_attempts))
        return Realization(
            points=self.points,
            anchors=self.anchors[:attempts],
            targets=self.targets[:attempts],
            raw_counts=self.raw_counts[:attempts],
            retained_by_attempt=retained_by_attempt,
            retained_attempts=retained_attempts,
            unique_edges=unique_edges,
        )


def generate_realization(
    n: int,
    m: int,
    gamma: float,
    s_max: int = 5,
    seed: int | np.random.SeedSequence | None = None,
) -> Realization:
    """Generate one fixed-cap Sat-RSH realization."""
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(m, (int, np.integer)) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(s_max, (int, np.integer)) or s_max < 2:
        raise ValueError("s_max must be an integer of at least two")
    if 2 * s_max >= n:
        raise ValueError("the satellite interpretation requires s_max < n/2")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive exponent")
    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    point_seed, target_seed, anchor_seed = seed_sequence.spawn(3)
    point_rng = np.random.default_rng(point_seed)
    target_rng = np.random.default_rng(target_seed)
    anchor_rng = np.random.default_rng(anchor_seed)

    points = sphere_points(n, point_rng)
    anchors = sphere_points(m, anchor_rng)
    sizes, probabilities = zipf_pmf(gamma, s_max)
    targets = target_rng.choice(sizes, size=m, p=probabilities)
    tree = cKDTree(points)

    candidates_by_attempt: list[Sequence[int] | None] = [None] * m
    for target in sizes:
        positions = np.flatnonzero(targets == target)
        if not len(positions):
            continue
        neighborhoods = tree.query_ball_point(
            anchors[positions], cap_chord_radius(int(target), n), workers=1
        )
        for position, neighborhood in zip(positions, neighborhoods):
            candidates_by_attempt[int(position)] = neighborhood

    raw_counts = np.zeros(m, dtype=int)
    retained_by_attempt: list[frozenset[int] | None] = [None] * m
    retained_attempts: list[frozenset[int]] = []
    for attempt_index, candidate_sequence in enumerate(candidates_by_attempt):
        candidates = np.asarray(candidate_sequence, dtype=int)
        raw_counts[attempt_index] = len(candidates)
        if len(candidates) < 2:
            continue
        if len(candidates) > s_max:
            similarities = points[candidates] @ anchors[attempt_index]
            keep = np.argpartition(similarities, -s_max)[-s_max:]
            candidates = candidates[keep]
        edge = frozenset(int(value) for value in candidates)
        retained_by_attempt[attempt_index] = edge
        retained_attempts.append(edge)

    # dict preserves the first occurrence, which makes exported examples stable.
    unique_edges = list(dict.fromkeys(retained_attempts))
    return Realization(
        points=points,
        anchors=anchors,
        targets=targets,
        raw_counts=raw_counts,
        retained_by_attempt=retained_by_attempt,
        retained_attempts=retained_attempts,
        unique_edges=unique_edges,
    )


def hyperedge_degrees(n: int, edges: Iterable[frozenset[int]]) -> np.ndarray:
    degree = np.zeros(n, dtype=int)
    for edge in edges:
        degree[np.fromiter(edge, dtype=int)] += 1
    return degree


def shadow_graph(n: int, edges: Iterable[frozenset[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for edge in edges:
        graph.add_edges_from(combinations(edge, 2))
    return graph


def realization_metrics(
    realization: Realization,
    *,
    need_graph: bool = False,
    need_clustering: bool = False,
    need_paths: bool = False,
) -> dict[str, object]:
    n = len(realization.points)
    degree = hyperedge_degrees(n, realization.unique_edges)
    result: dict[str, object] = {
        "degree_sequence": degree,
        "mean_hyperedge_degree": float(degree.mean()),
        "retained_attempts": len(realization.retained_attempts),
        "unique_edges": len(realization.unique_edges),
        "unique_fraction_attempts": realization.unique_fraction_of_attempts,
        "unique_fraction_retained": realization.unique_fraction_of_retained,
        "unique_sizes": realization.unique_sizes,
        "retained_sizes": realization.retained_sizes,
    }
    if not (need_graph or need_clustering or need_paths):
        return result

    graph = shadow_graph(n, realization.unique_edges)
    result["connected"] = bool(nx.is_connected(graph))
    result["mean_shadow_degree"] = float(np.mean([value for _, value in graph.degree()]))
    if need_clustering:
        # NetworkX's count_zeros=True convention includes isolated vertices.
        result["clustering"] = float(nx.average_clustering(graph, count_zeros=True))
    if need_paths:
        component = max(nx.connected_components(graph), key=len)
        lcc = graph.subgraph(component)
        result["lcc_fraction"] = len(component) / n
        if len(component) > 1:
            # SciPy performs the all-pairs traversal in compiled code and is
            # substantially faster than a Python-level BFS from every node.
            nodes = list(component)
            adjacency = nx.to_scipy_sparse_array(
                lcc, nodelist=nodes, dtype=float, format="csr"
            )
            distances = shortest_path(
                adjacency, directed=False, unweighted=True, overwrite=False
            )
            result["apl_lcc"] = float(
                distances.sum() / (len(component) * (len(component) - 1))
            )
        else:
            result["apl_lcc"] = np.nan
    return result


def generate_size_matched_null(
    n: int, size_sequence: Sequence[int], seed: int | np.random.SeedSequence | None
) -> list[frozenset[int]]:
    """Uniform simple hypergraph with the same retained size sequence.

    Each requested size is sampled uniformly without replacement from the
    vertices.  Collisions are resampled so that the number and sizes of unique
    null hyperedges match the Sat-RSH realization exactly.
    """
    rng = np.random.default_rng(seed)
    used: set[frozenset[int]] = set()
    output: list[frozenset[int]] = []
    for size in size_sequence:
        for _ in range(10000):
            edge = frozenset(int(value) for value in rng.choice(n, int(size), replace=False))
            if edge not in used:
                used.add(edge)
                output.append(edge)
                break
        else:
            raise RuntimeError("Unable to construct a unique size-matched null hyperedge")
    return output


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return float(center - half_width), float(center + half_width)


def poisson_total_variation(degrees: np.ndarray) -> tuple[float, float, float]:
    """Return empirical mean, variance/mean, and TV from a fitted Poisson law."""
    values = np.asarray(degrees, dtype=int)
    mean = float(values.mean())
    variance = float(values.var())
    maximum = int(values.max())
    empirical = np.bincount(values, minlength=maximum + 1) / len(values)
    fitted = poisson.pmf(np.arange(maximum + 1), mean)
    tail = max(0.0, 1.0 - float(fitted.sum()))
    total_variation = 0.5 * (float(np.abs(empirical - fitted).sum()) + tail)
    return mean, variance / mean if mean else np.nan, total_variation
