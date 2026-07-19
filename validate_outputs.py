"""Validate the complete publication data and figure package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
EXPECTED_ROWS = {
    "figure1_hyperedges.csv": 178,
    "figure2_geometry.csv": 1,
    "figure3_size_distribution_trials.csv": 800,
    "figure4_degree_trials.csv": 1200,
    "figure5_connectivity_trials.csv": 6400,
    "figure6_node_degrees.csv": 50000,
    "figure7_clustering_trials.csv": 2700,
    "figure8_path_trials.csv": 1400,
}
EXPECTED_FIGURES = list(range(1, 9))


def csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        return sum(1 for _ in rows)


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
        if manifest.get("base_seed") != 20260718:
            failures.append("unexpected base seed in experiment manifest")
        if manifest.get("license") != "MIT":
            failures.append("experiment manifest does not record the MIT license")

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

    print("Publication output validation passed: Figures 1--8 and all data files are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
