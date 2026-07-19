# Satellite Random Spherical Hypergraphs

This repository is the standalone reproducibility package for the experiments
reported in *Satellite Random Spherical Hypergraphs*. It contains the exact
fixed-cap Sat-RSH model, the Figure 1--8 generation pipeline, trial-level data,
publication figures, software metadata, and automated validation checks.

Repository: <https://github.com/hjwhhhh/Satellite-Random-Spherical-Hypergraphs>

Historical kNN-GRSH scripts are intentionally excluded because they do not
generate the results reported in the manuscript.

## Repository contents

- `sat_rsh_model.py`: model generator, exact retained-attempt mixture, graph
  projections and metrics, Wilson intervals, and the size-matched control.
- `reproduce_all.py`: regenerates Figures 1--8, trial-level CSV files, and run
  metadata.
- `validate_outputs.py`: verifies the complete publication output package.
- `tests/`: fast model-level regression and property tests.
- `requirements.txt`: exact package versions used for the reported results.
- `data/`: committed trial-level CSV files and the complete experiment
  manifest.
- `figures/`: committed vector PDFs and 600-dpi RGB PNGs.
- `CITATION.cff`: GitHub-compatible software citation metadata.
- `LICENSE`: MIT software license.
- `LICENSE-DATA.md`: CC BY 4.0 terms for committed data and figures.

## Installation

The reported experiments were run with Python 3.13.9. Clone the repository,
create an isolated environment, and install the pinned dependencies:

```bash
git clone https://github.com/hjwhhhh/Satellite-Random-Spherical-Hypergraphs.git
cd Satellite-Random-Spherical-Hypergraphs
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Fast checks

Run the model-level tests:

```bash
python -m unittest discover -s tests -v
```

Figures 1 and 2 provide a quick installation check:

```bash
python reproduce_all.py --figures 1 2 --workers 1
```

This partial command writes `data/last_run_manifest.json`; it deliberately does
not overwrite the committed full-paper `data/experiment_manifest.json`.

## Full reproduction

Regenerate all experiments at the trial counts reported in the manuscript:

```bash
python reproduce_all.py --workers 4
```

The full command writes Figures 1--8, all trial-level CSV files, and
`data/experiment_manifest.json`. Figure 8 computes exact all-pairs distances in
the largest connected component and is normally the slowest stage. The worker
count controls concurrent independent realizations and does not alter seeds or
numerical results.

After the run, validate the publication package:

```bash
python validate_outputs.py
```

The validator checks the complete file set, CSV row counts, full manifest,
600-dpi metadata, and true RGB image mode.

## Expected data files

| File | Data rows | Content |
|---|---:|---|
| `figure1_hyperedges.csv` | 178 | Unique hyperedges in both example realizations |
| `figure2_geometry.csv` | 1 | Footprint angles and normalized altitude |
| `figure3_size_distribution_trials.csv` | 800 | Trial-level retained-size counts and proportions |
| `figure4_degree_trials.csv` | 1200 | Degree and unique-edge-yield statistics |
| `figure5_connectivity_trials.csv` | 6400 | Coupled finite-size connectivity indicators |
| `figure6_node_degrees.csv` | 50000 | Vertex degrees for the Poisson comparison |
| `figure7_clustering_trials.csv` | 2700 | Sat-RSH and control clustering values |
| `figure8_path_trials.csv` | 1400 | LCC paths, coverage, connectivity, and mean degree |

All data are synthetic and are regenerated from the fixed base seed. No human,
animal, confidential, or third-party data are included.

## Figure mapping

1. Example Sat-RSH realizations on the sphere.
2. Spherical footprint geometry.
3. Retained hyperedge-size distributions.
4. Hypergraph vertex-degree summary statistics.
5. Empirical finite-size connectivity transitions.
6. Hypergraph vertex-degree distributions.
7. Clustering relative to a size-matched non-geometric control.
8. Largest-component path lengths, component fractions, and mean hypergraph
   vertex degrees.

## Reproducibility conventions

- Base seed: `20260718`.
- Independent streams are derived with `numpy.random.SeedSequence`.
- Figures 5 and 8 evaluate nested prefixes of one maximum attempt sequence
  within each trial, making sample-level connectivity nondecreasing in density.
- Repeated parameter combinations in Figures 5, 7, and 8 reuse the same
  trial-level realization across the relevant panels.
- Error bars are sample standard deviations unless a caption states otherwise.
- Connectivity bands are 95% Wilson score intervals.
- Clustering assigns zero to isolated and degree-one vertices.
- Path length is computed in the largest connected component and exported with
  its vertex fraction.
- The Figure 7 control matches the vertex set, number of unique hyperedges, and
  complete hyperedge-size sequence, but not the vertex-degree sequence.

Every run records the Python version, operating system, package versions,
worker count, requested figures, seed rule, repository URL, and license. A
partial run records its scope separately and cannot replace the full-paper
manifest.

## Figure format and fonts

Each figure is exported as a vector PDF with embedded TrueType fonts and as a
600-dpi, 8-bit RGB PNG. The Okabe--Ito color-vision-deficiency-safe palette is
supplemented with marker shapes and line styles.

The reported figures use Arial. On systems without Arial, Matplotlib may use a
fallback sans-serif font; numerical data remain identical, but small layout
differences can occur. Install Arial before reproduction when pixel-level
visual matching is required.

## Citation and archival release

Use the repository citation exposed by `CITATION.cff`. For a submitted or
published article, also cite the manuscript itself. A tagged GitHub release can
be archived through Zenodo to obtain an immutable software DOI.

## License

The software is released under the MIT License; see `LICENSE`. The committed
trial-level data and figures are released under the Creative Commons
Attribution 4.0 International license; see `LICENSE-DATA.md`. Reuse should cite
the repository and the accompanying manuscript.
