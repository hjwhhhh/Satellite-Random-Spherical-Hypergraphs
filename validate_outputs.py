"""Validate the complete publication data and figure package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from sat_rsh_model import poisson_total_variation


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
EXPECTED_BASE_SEED = 42
EXPECTED_ROWS = {
    "figure1_hyperedges.csv": 179,
    "figure2_geometry.csv": 1,
    "figure3_size_distribution_trials.csv": 800,
    "figure4_degree_trials.csv": 1200,
    "figure5_connectivity_trials.csv": 6400,
    "figure6_node_degrees.csv": 50000,
    "figure6_degree_diagnostics.csv": 2,
    "figure7_clustering_trials.csv": 2730,
    "figure7_control_summary.csv": 10,
    "figure7_rewiring_diagnostics.csv": 120,
    "figure8_path_trials.csv": 1400,
    "figure8_density_change_summary.csv": 4,
}
EXPECTED_HEADERS = {
    "figure3_size_distribution_trials.csv": ["gamma", "trial", "size", "unique_count", "unique_proportion", "retained_count", "retained_proportion", "retained_attempts", "unique_edges"],
    "figure4_degree_trials.csv": ["gamma", "rho", "trial", "mean_hypergraph_vertex_degree", "attempt_level_reference", "unique_fraction_attempts", "unique_fraction_retained"],
    "figure5_connectivity_trials.csv": ["panel", "group", "rho", "trial", "connected"],
    "figure6_node_degrees.csv": ["gamma", "trial", "vertex", "degree"],
    "figure6_degree_diagnostics.csv": ["gamma", "realizations", "vertices_per_realization", "mean_degree", "variance_to_mean", "dispersion_ci_low", "dispersion_ci_high", "total_variation", "tv_ci_low", "tv_ci_high", "bootstrap_resamples", "resampling_unit"],
    "figure7_clustering_trials.csv": ["panel", "group", "rho", "trial", "model", "clustering"],
    "figure7_control_summary.csv": ["rho", "model", "paired_trials", "sat_mean", "control_mean", "mean_paired_difference", "ci95_low", "ci95_high", "positive_pairs"],
    "figure7_rewiring_diagnostics.csv": ["rho", "trial", "q", "sat_seed", "size_null_seed", "checkpoint_swaps_per_edge", "rewire_seed", "stage_successful_swaps", "stage_attempted_swaps", "cumulative_successful_swaps", "cumulative_attempted_swaps", "stage_acceptance_rate", "clustering", "degree_sequence_sha256", "size_sequence_sha256", "sat_degree_sequence_sha256", "sat_size_sequence_sha256"],
    "figure8_path_trials.csv": ["panel", "group", "rho", "trial", "apl_lcc", "lcc_fraction", "connected", "mean_hypergraph_vertex_degree"],
}
EXPECTED_FIGURES = list(range(1, 9))


def csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        return sum(1 for _ in rows)


def read_dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approximately_equal(first: float, second: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(first, second, rel_tol=tolerance, abs_tol=tolerance)


def audit_scientific_invariants(failures: list[str]) -> None:
    tables = {
        name: read_dict_rows(DATA_DIR / name)[1]
        for name in EXPECTED_HEADERS
        if (DATA_DIR / name).is_file()
    }

    figure3 = tables.get("figure3_size_distribution_trials.csv", [])
    for key in {(row["gamma"], row["trial"]) for row in figure3}:
        subset = [row for row in figure3 if (row["gamma"], row["trial"]) == key]
        if len(subset) != 4:
            failures.append(f"Figure 3 group {key!r} does not contain four sizes")
            continue
        unique_total = sum(int(row["unique_count"]) for row in subset)
        retained_total = sum(int(row["retained_count"]) for row in subset)
        if unique_total != int(subset[0]["unique_edges"]):
            failures.append(f"Figure 3 unique counts do not sum for {key!r}")
        if retained_total != int(subset[0]["retained_attempts"]):
            failures.append(f"Figure 3 retained counts do not sum for {key!r}")
        if not approximately_equal(sum(float(row["unique_proportion"]) for row in subset), 1.0):
            failures.append(f"Figure 3 unique proportions do not sum to one for {key!r}")
        if not approximately_equal(sum(float(row["retained_proportion"]) for row in subset), 1.0):
            failures.append(f"Figure 3 retained proportions do not sum to one for {key!r}")

    figure5 = tables.get("figure5_connectivity_trials.csv", [])
    grouped5: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in figure5:
        connected = int(row["connected"])
        if connected not in (0, 1):
            failures.append("Figure 5 connected indicator is not binary")
        grouped5[(row["panel"], row["group"], row["trial"])].append(
            (float(row["rho"]), connected)
        )
    for key, values in grouped5.items():
        sequence = [value for _, value in sorted(values)]
        if sequence != sorted(sequence):
            failures.append(f"Figure 5 connectivity is not prefix-monotone for {key!r}")

    figure7 = tables.get("figure7_clustering_trials.csv", [])
    if any(not 0.0 <= float(row["clustering"]) <= 1.0 for row in figure7):
        failures.append("Figure 7 contains clustering outside [0,1]")
    rewiring = tables.get("figure7_rewiring_diagnostics.csv", [])
    checkpoints_by_trial: dict[str, list[int]] = defaultdict(list)
    for row in rewiring:
        checkpoint = int(row["checkpoint_swaps_per_edge"])
        q = int(row["q"])
        checkpoints_by_trial[row["trial"]].append(checkpoint)
        if row["degree_sequence_sha256"] != row["sat_degree_sequence_sha256"]:
            failures.append("Figure 7 rewiring changed a vertex-degree sequence")
        if row["size_sequence_sha256"] != row["sat_size_sequence_sha256"]:
            failures.append("Figure 7 rewiring changed a hyperedge-size sequence")
        if int(row["cumulative_successful_swaps"]) != checkpoint * q:
            failures.append("Figure 7 rewiring successful-swap count is inconsistent")
        if not 0.0 < float(row["stage_acceptance_rate"]) <= 1.0:
            failures.append("Figure 7 rewiring acceptance rate is outside (0,1]")
    if any(sorted(values) != [20, 50, 100, 200] for values in checkpoints_by_trial.values()):
        failures.append("Figure 7 rewiring checkpoints are incomplete")

    figure8 = tables.get("figure8_path_trials.csv", [])
    grouped8: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in figure8:
        if int(row["connected"]) not in (0, 1):
            failures.append("Figure 8 connected indicator is not binary")
        if not 0.0 <= float(row["lcc_fraction"]) <= 1.0:
            failures.append("Figure 8 LCC fraction is outside [0,1]")
        grouped8[(row["panel"], row["group"], row["trial"])].append(row)
    for key, values in grouped8.items():
        ordered = sorted(values, key=lambda row: float(row["rho"]))
        for field in ("connected", "lcc_fraction", "mean_hypergraph_vertex_degree"):
            sequence = [float(row[field]) for row in ordered]
            if any(later + 1e-12 < earlier for earlier, later in zip(sequence, sequence[1:])):
                failures.append(f"Figure 8 {field} is not prefix-monotone for {key!r}")

    degrees = tables.get("figure6_node_degrees.csv", [])
    diagnostics = tables.get("figure6_degree_diagnostics.csv", [])
    for summary in diagnostics:
        gamma = summary["gamma"]
        values = np.asarray(
            [int(row["degree"]) for row in degrees if row["gamma"] == gamma],
            dtype=int,
        )
        mean, dispersion, total_variation = poisson_total_variation(values)
        for observed, expected, label in (
            (float(summary["mean_degree"]), mean, "mean"),
            (float(summary["variance_to_mean"]), dispersion, "dispersion"),
            (float(summary["total_variation"]), total_variation, "TV"),
        ):
            if not approximately_equal(observed, expected):
                failures.append(f"Figure 6 {label} summary is inconsistent for gamma={gamma}")

    # Shared baselines must be exact row-for-row copies, not new simulations.
    baseline_pairs = [
        (figure5, lambda row: row["panel"] == "gamma" and float(row["group"]) == 2.0, lambda row: row["panel"] == "smax" and int(float(row["group"])) == 5, ("rho", "trial", "connected"), "Figure 5"),
        (figure7, lambda row: row["panel"] == "gamma" and float(row["group"]) == 2.0, lambda row: row["panel"] == "smax" and int(float(row["group"])) == 5, ("rho", "trial", "clustering"), "Figure 7(a,b)"),
        (figure8, lambda row: row["panel"] == "gamma" and float(row["group"]) == 2.0, lambda row: row["panel"] == "n" and int(float(row["group"])) == 300, ("rho", "trial", "apl_lcc", "lcc_fraction", "connected", "mean_hypergraph_vertex_degree"), "Figure 8"),
    ]
    for table, first_filter, second_filter, fields, label in baseline_pairs:
        first = sorted(tuple(row[field] for field in fields) for row in table if first_filter(row))
        second = sorted(tuple(row[field] for field in fields) for row in table if second_filter(row))
        if first != second:
            failures.append(f"{label} shared baseline is inconsistent")


def main() -> int:
    failures: list[str] = []
    manifest_path = DATA_DIR / "experiment_manifest.json"
    if not manifest_path.is_file():
        failures.append("missing data/experiment_manifest.json")
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("figures") != EXPECTED_FIGURES:
            failures.append(
                "experiment manifest must list all figures 1--8; found "
                f"{manifest.get('figures')!r}"
            )
        if manifest.get("base_seed") != EXPECTED_BASE_SEED:
            failures.append("unexpected base seed in experiment manifest")
        if manifest.get("license") != "MIT":
            failures.append("experiment manifest does not record the MIT license")
        if manifest.get("analysis_version") != 2:
            failures.append("unexpected analysis version in experiment manifest")
        if set(manifest.get("parameter_grids", {})) != {f"figure{number}" for number in EXPECTED_FIGURES}:
            failures.append("experiment manifest does not contain all parameter grids")
        for relative, expected_hash in manifest.get("output_sha256", {}).items():
            path = ROOT / relative
            if not path.is_file() or file_sha256(path) != expected_hash:
                failures.append(f"SHA-256 mismatch for {relative}")

    for filename, expected in EXPECTED_ROWS.items():
        path = DATA_DIR / filename
        if not path.is_file():
            failures.append(f"missing data/{filename}")
            continue
        observed = csv_data_rows(path)
        if observed != expected:
            failures.append(
                f"data/{filename} has {observed} rows; expected {expected}"
            )
        expected_header = EXPECTED_HEADERS.get(filename)
        if expected_header is not None:
            observed_header, _ = read_dict_rows(path)
            if observed_header != expected_header:
                failures.append(
                    f"data/{filename} has unexpected columns: {observed_header!r}"
                )

    audit_scientific_invariants(failures)

    for number in EXPECTED_FIGURES:
        pdf_path = FIGURE_DIR / f"figure{number}.pdf"
        png_path = FIGURE_DIR / f"figure{number}.png"
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            failures.append(f"missing or empty figures/figure{number}.pdf")
        if not png_path.is_file() or png_path.stat().st_size == 0:
            failures.append(f"missing or empty figures/figure{number}.png")
            continue
        with Image.open(png_path) as image:
            if image.mode != "RGB":
                failures.append(
                    f"figures/figure{number}.png is {image.mode}; expected RGB"
                )
            dpi = image.info.get("dpi", (0.0, 0.0))
            if any(abs(float(value) - 600.0) > 1.0 for value in dpi):
                failures.append(
                    f"figures/figure{number}.png reports {dpi!r}; expected 600 dpi"
                )

    if failures:
        print("Publication output validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Publication output validation passed: files, schemas, hashes, scientific invariants, and shared baselines are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
