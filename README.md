# Satellite Random Spherical Hypergraphs

This directory is the standalone reproducibility package for the experiments
reported in *Satellite Random Spherical Hypergraphs*. It contains the exact
fixed-cap Sat-RSH model and the figure-generation pipeline used for Figures
1--8 of the manuscript.

Historical kNN-GRSH scripts elsewhere in the working project are intentionally
excluded because they do not generate the results in the current manuscript.

## Contents

- `sat_rsh_model.py` implements the fixed-cap Sat-RSH generator, retained-size
  probabilities, graph projections and metrics, Wilson intervals, and the
  size-matched non-geometric control.
- `reproduce_all.py` regenerates every publication figure, all trial-level CSV
  files, and a machine-readable experiment manifest.
- `requirements.txt` pins the package versions used for the reported results.

Generated files are written to `figures/` and `data/` inside this directory.
The script creates both directories when needed.

## Installation

The reported experiments were run with Python 3.13.9. From this directory,
create an isolated environment and install the pinned dependencies:

```bash
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

## Quick verification

Figures 1 and 2 are deterministic and fast enough for a basic installation
check:

```bash
python reproduce_all.py --figures 1 2 --workers 1
```

This command should create vector PDFs, 600-dpi PNGs, two CSV files, and
`data/experiment_manifest.json`.

## Full reproduction

To regenerate all experiments with the trial counts reported in the paper:

```bash
python reproduce_all.py --workers 4
```

To reproduce selected figures only:

```bash
python reproduce_all.py --figures 3 4 5 --workers 4
```

Figure 8 computes exact all-pairs distances in the largest connected component
and is normally the slowest stage.

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
- Figures 5 and 8 use nested prefixes of a common maximum-attempt sequence
  within each trial. This preserves the correct marginal model and makes each
  sample-level connectivity curve nondecreasing in density.
- Repeated parameter combinations in Figures 5, 7, and 8 reuse a common set of
  trial realizations across panels.
- Error bars are sample standard deviations unless a figure states otherwise.
- Connectivity bands are 95% Wilson score intervals.
- Clustering assigns coefficient zero to isolated and degree-one vertices.
- Path length is computed only in the largest connected component; its vertex
  fraction is exported with the path statistic.
- The Figure 7 control matches the vertex set, final number of unique
  hyperedges, and final unique-hyperedge size sequence. It does not match the
  hypergraph vertex-degree sequence.

Every run records the Python version, operating system, package versions,
requested figures, worker count, seed rule, and reported trial counts in
`data/experiment_manifest.json`.

## Output format

- Figures are exported as vector PDFs with embedded TrueType fonts and as
  600-dpi RGB PNGs.
- The Okabe--Ito colour-vision-deficiency-safe palette is supplemented with
  marker shapes and line styles.
- Trial-level statistics are stored as UTF-8 CSV files with headers.

## License

No software license has been selected in this working copy. Add a `LICENSE`
file before public release; without an explicit license, public availability
does not grant general reuse rights.
