"""Regenerate every Sat-RSH experiment, data export, and vector figure.

Run from the ``论文模板`` directory with

    python sat_rsh_experiments/reproduce_all.py

The default run uses the trial counts reported in the manuscript.  All random
seeds are deterministic functions of ``BASE_SEED`` and the figure parameters.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.stats import poisson, t

from sat_rsh_model import (
    GAMMA_COLORS,
    GAMMA_MARKERS,
    OKABE_ITO,
    SIZE_COLORS,
    SIZE_MARKERS,
    cap_angle,
    generate_realization,
    generate_size_matched_null,
    poisson_total_variation,
    publication_style,
    realization_metrics,
    retained_attempt_probabilities,
    shadow_graph,
    wilson_interval,
    zipf_pmf,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
BASE_SEED = 20260718
GAMMAS = (1.5, 2.0, 2.5, 3.0)
SMAX_VALUES = (2, 3, 4, 5)

plt.rcParams.update(publication_style())


def stable_seed(*coordinates: int) -> int:
    return int(np.random.SeedSequence([BASE_SEED, *coordinates]).generate_state(1)[0])


def save_figure(fig: plt.Figure, number: int) -> None:
    """Save editable vector art and a 600-dpi RGB review/submission bitmap."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"figure{number}.{suffix}",
            format=suffix,
            dpi=600,
            bbox_inches=None,
            facecolor="white",
        )
    plt.close(fig)


def panel_title(ax, label: str, title: str, *, pad: float = 5.0) -> None:
    """Set a left-aligned title with a bold upright Nature-style panel label."""
    ax.set_title(rf"$\mathbf{{{label}}}$  {title}", loc="left", pad=pad)


def clean_axes(ax) -> None:
    """Use an open-axis style while retaining all required ticks and labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parallel_map(label: str, worker, jobs: list[tuple], workers: int) -> list:
    output = [None] * len(jobs)
    completed = 0
    checkpoint = max(1, len(jobs) // 10)
    print(f"{label}: {len(jobs)} independent realizations", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, *job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            output[futures[future]] = future.result()
            completed += 1
            if completed % checkpoint == 0 or completed == len(jobs):
                print(f"  {label}: {completed}/{len(jobs)}", flush=True)
    return output


def trial_worker(n, m, gamma, s_max, seed, graph, clustering, paths):
    realization = generate_realization(n, m, gamma, s_max, seed)
    metrics = realization_metrics(
        realization,
        need_graph=graph,
        need_clustering=clustering,
        need_paths=paths,
    )
    return realization, metrics


def prefix_trial_worker(
    n, max_m, gamma, s_max, seed, prefix_counts, graph, clustering, paths
):
    """Evaluate coupled density points from one maximum attempt sequence."""
    realization = generate_realization(n, max_m, gamma, s_max, seed)
    return [
        realization_metrics(
            realization.prefix(int(attempts)),
            need_graph=graph,
            need_clustering=clustering,
            need_paths=paths,
        )
        for attempts in prefix_counts
    ]


def draw_hyperedge(ax, points: np.ndarray, edge: frozenset[int], color: str) -> None:
    indices = np.fromiter(edge, dtype=int)
    vertices = points[indices]
    if len(vertices) == 2:
        # Draw the geodesic rather than a chord through the sphere.
        p, q = vertices
        omega = float(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))
        if omega < 1e-10:
            curve = np.vstack((p, q))
        else:
            t = np.linspace(0.0, 1.0, 18)
            curve = (
                np.sin((1.0 - t) * omega)[:, None] * p
                + np.sin(t * omega)[:, None] * q
            ) / np.sin(omega)
        ax.plot(*curve.T, color=color, lw=0.75, alpha=0.70, zorder=2)
        return
    if len(vertices) == 3:
        collection = Poly3DCollection([vertices], facecolor=color, edgecolor=color, alpha=0.14)
        collection.set_linewidth(0.45)
        ax.add_collection3d(collection)
        return
    try:
        hull = ConvexHull(vertices)
        faces = [vertices[simplex] for simplex in hull.simplices]
    except Exception:
        faces = [vertices]
    collection = Poly3DCollection(faces, facecolor=color, edgecolor=color, alpha=0.09)
    collection.set_linewidth(0.35)
    ax.add_collection3d(collection)


def make_figure1() -> None:
    n, m, s_max = 100, 160, 5
    fig = plt.figure(figsize=(7.2, 2.75))
    rows: list[dict] = []
    for panel, gamma in enumerate((2.0, 4.0), start=1):
        realization = generate_realization(n, m, gamma, s_max, stable_seed(1, panel))
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        u = np.linspace(0, 2 * np.pi, 42)
        v = np.linspace(0, np.pi, 22)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color="#CBD5E1", lw=0.28, alpha=0.32)
        for edge in realization.unique_edges:
            draw_hyperedge(ax, realization.points, edge, SIZE_COLORS[len(edge)])
        ax.scatter(
            realization.points[:, 0],
            realization.points[:, 1],
            realization.points[:, 2],
            s=5.2,
            color="#1F2937",
            edgecolor="white",
            linewidth=0.18,
            depthshade=False,
            zorder=5,
        )
        counts = {size: int(np.sum(realization.unique_sizes == size)) for size in range(2, 6)}
        subtitle = ", ".join(f"$N_{size}={counts[size]}$" for size in range(2, 6))
        panel_title(ax, chr(96 + panel), rf"$\gamma={gamma:.1f}$" + "\n" + subtitle, pad=-4)
        ax.view_init(elev=18, azim=35)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 1), zoom=1.30)
        ax.set_axis_off()
        for attempt, edge in enumerate(realization.unique_edges):
            rows.append(
                {
                    "panel_gamma": gamma,
                    "unique_edge_index": attempt,
                    "size": len(edge),
                    "vertices": " ".join(str(value) for value in sorted(edge)),
                }
            )

    handles = [
        Line2D([0], [0], color=SIZE_COLORS[size], lw=2.5, label=rf"$|e|={size}$")
        for size in range(2, 6)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015), handlelength=2.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.15, wspace=0.02)
    save_figure(fig, 1)
    write_csv("figure1_hyperedges.csv", ["panel_gamma", "unique_edge_index", "size", "vertices"], rows)
    print("Figure 1 complete", flush=True)


def make_figure2() -> None:
    h = 0.35
    satellite_radius = 1.0 + h
    theta = float(np.arccos(1.0 / satellite_radius))
    alpha = np.pi / 2.0 - theta
    satellite = np.array([0.0, satellite_radius])
    tangent_right = np.array([np.sin(theta), np.cos(theta)])
    tangent_left = np.array([-np.sin(theta), np.cos(theta)])

    fig, ax = plt.subplots(figsize=(4.65, 3.75))
    circle = np.linspace(0, 2 * np.pi, 500)
    ax.fill(np.cos(circle), np.sin(circle), color=OKABE_ITO["sky_blue"], alpha=0.12)
    ax.plot(np.cos(circle), np.sin(circle), color=OKABE_ITO["black"], lw=1.0)
    cap_arc = np.linspace(np.pi / 2 - theta, np.pi / 2 + theta, 160)
    ax.plot(np.cos(cap_arc), np.sin(cap_arc), color=OKABE_ITO["vermillion"], lw=3.2, alpha=0.68)
    for tangent in (tangent_left, tangent_right):
        ax.plot([satellite[0], tangent[0]], [satellite[1], tangent[1]], ls="--", lw=0.9, color=OKABE_ITO["orange"])
    ax.plot([0, 0], [0, satellite_radius + 0.08], ls=":", lw=0.9, color=OKABE_ITO["gray"])
    ax.plot([-0.32, 0.58], [satellite_radius, satellite_radius], ls=":", lw=0.9, color=OKABE_ITO["gray"])
    ax.scatter(*satellite, marker="*", s=115, color=OKABE_ITO["orange"], edgecolor="black", linewidth=0.45, zorder=6)
    ax.text(-0.08, satellite_radius + 0.10, "satellite", color="black", weight="bold", ha="right", va="center")
    ax.text(0.42, satellite_radius + 0.035, "local horizon", color="black", ha="center", va="bottom")
    ax.scatter(0, 0, marker="+", s=44, color="black")
    ax.text(0.04, -0.075, "$O$")
    ax.scatter(0, 1, s=18, color="black", zorder=6)
    ax.text(0.04, 0.91, "nadir point $q$")

    central_arc = np.linspace(np.pi / 2 - theta, np.pi / 2, 80)
    ax.plot(0.42 * np.cos(central_arc), 0.42 * np.sin(central_arc), color=OKABE_ITO["blue"], lw=1.35)
    central_mid = np.pi / 2 - theta / 2
    ax.text(0.48 * np.cos(central_mid), 0.48 * np.sin(central_mid), r"$\theta$", color="black", fontsize=8.0)

    down = np.array([0.0, -1.0])
    line = tangent_right - satellite
    line /= np.linalg.norm(line)
    angle_down = np.arctan2(down[1], down[0])
    angle_line = np.arctan2(line[1], line[0])
    arc_angles = np.linspace(angle_down, angle_line, 80)
    ax.plot(
        satellite[0] + 0.18 * np.cos(arc_angles),
        satellite[1] + 0.18 * np.sin(arc_angles),
        color=OKABE_ITO["bluish_green"],
        lw=1.35,
    )
    alpha_mid = (angle_down + angle_line) / 2.0
    ax.text(
        satellite[0] + 0.23 * np.cos(alpha_mid),
        satellite[1] + 0.23 * np.sin(alpha_mid),
        r"$\alpha$",
        color="black",
        fontsize=8.0,
        ha="center",
        va="center",
    )

    depression_angles = np.linspace(angle_line, 0.0, 80)
    ax.plot(
        satellite[0] + 0.27 * np.cos(depression_angles),
        satellite[1] + 0.27 * np.sin(depression_angles),
        color=OKABE_ITO["blue"],
        lw=1.35,
    )
    depression_mid = angle_line / 2.0
    ax.text(
        satellite[0] + 0.33 * np.cos(depression_mid),
        satellite[1] + 0.33 * np.sin(depression_mid),
        r"$\theta$",
        color="black",
        fontsize=8.0,
        ha="center",
        va="center",
    )
    ax.annotate("", xy=(0, 1), xytext=(0, satellite_radius), arrowprops=dict(arrowstyle="<->", color=OKABE_ITO["reddish_purple"], lw=1.0))
    ax.text(-0.075, 1.16, "$h$", color="black", ha="right")

    rng = np.random.default_rng(stable_seed(2, 1))
    ground_angles = rng.uniform(0, 2 * np.pi, 18)
    in_cap = np.abs(np.angle(np.exp(1j * (ground_angles - np.pi / 2)))) <= theta
    ax.scatter(np.cos(ground_angles[~in_cap]), np.sin(ground_angles[~in_cap]), s=14, color=OKABE_ITO["gray"], zorder=5, label="outside footprint")
    ax.scatter(np.cos(ground_angles[in_cap]), np.sin(ground_angles[in_cap]), s=20, color=OKABE_ITO["vermillion"], edgecolor="black", linewidth=0.3, zorder=5, label="captured ground point")
    ax.annotate(
        "footprint arc",
        xy=tangent_left,
        xytext=(-1.04, 1.13),
        ha="center",
        va="center",
        color="black",
        arrowprops=dict(arrowstyle="-", color=OKABE_ITO["vermillion"], lw=0.8),
    )

    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, labelspacing=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.08, 1.53)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_figure(fig, 2)
    write_csv(
        "figure2_geometry.csv",
        ["h_over_R", "theta_central_rad", "alpha_off_nadir_rad", "depression_from_horizon_rad"],
        [{"h_over_R": h, "theta_central_rad": theta, "alpha_off_nadir_rad": alpha, "depression_from_horizon_rad": theta}],
    )
    print("Figure 2 complete", flush=True)


def make_figure3(workers: int) -> None:
    n, rho, s_max, trials = 500, 3.0, 5, 50
    jobs = []
    labels = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for trial in range(trials):
            jobs.append((n, int(n * rho), gamma, s_max, stable_seed(3, gamma_index, trial), False, False, False))
            labels.append((gamma, trial))
    results = parallel_map("Figure 3", trial_worker, jobs, workers)

    rows: list[dict] = []
    proportions: dict[float, list[np.ndarray]] = {gamma: [] for gamma in GAMMAS}
    for (gamma, trial), (realization, metrics) in zip(labels, results):
        counts = np.asarray([np.sum(realization.unique_sizes == size) for size in range(2, 6)], dtype=float)
        props = counts / counts.sum()
        proportions[gamma].append(props)
        for size, count, prop in zip(range(2, 6), counts.astype(int), props):
            rows.append({"gamma": gamma, "trial": trial, "size": size, "unique_count": count, "unique_proportion": prop, "retained_attempts": metrics["retained_attempts"], "unique_edges": metrics["unique_edges"]})
    write_csv("figure3_size_distribution_trials.csv", ["gamma", "trial", "size", "unique_count", "unique_proportion", "retained_attempts", "unique_edges"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    sizes = np.arange(2, 6)
    width = 0.18
    calibration_x, calibration_y = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        values = np.vstack(proportions[gamma])
        means, standard_deviations = values.mean(axis=0), values.std(axis=0, ddof=1)
        offset = (gamma_index - 1.5) * width
        axes[0].bar(sizes + offset, means, width=width, color=GAMMA_COLORS[gamma], alpha=0.78)
        axes[0].errorbar(sizes + offset, means, yerr=standard_deviations, fmt="none", ecolor=GAMMA_COLORS[gamma], elinewidth=0.75, capsize=1.5, capthick=0.75)
        exact = retained_attempt_probabilities(n, gamma, s_max)
        target = exact["target_pmf"]
        retained = exact["conditional"]
        axes[0].plot(sizes + offset, target, marker="x", ls="none", color="black", ms=3.8, mew=0.8)
        axes[0].plot(sizes + offset, retained, marker="_", ls="none", color="black", ms=6.5, mew=1.2)
        calibration_x.extend(retained)
        calibration_y.extend(means)
        for size_index, (x_value, y_value, y_error) in enumerate(zip(retained, means, standard_deviations)):
            axes[1].errorbar(
                x_value,
                y_value,
                yerr=y_error,
                fmt=SIZE_MARKERS[int(sizes[size_index])],
                color=GAMMA_COLORS[gamma],
                markeredgecolor="white",
                markeredgewidth=0.35,
                ls="none",
                capsize=1.8,
                elinewidth=0.75,
                capthick=0.75,
            )

    lower = min(calibration_x + calibration_y) * 0.94
    upper = max(calibration_x + calibration_y) * 1.04
    axes[1].plot([lower, upper], [lower, upper], color=OKABE_ITO["gray"], ls="--", lw=0.9, zorder=0)
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[0].set_xticks(sizes)
    axes[0].set_xlabel("Retained hyperedge size $|e|$")
    axes[0].set_ylabel("Proportion")
    axes[1].set_xlabel("Exact retained-attempt probability")
    axes[1].set_ylabel("Observed unique-edge probability")
    panel_title(axes[0], "a", "Target and realized size laws")
    panel_title(axes[1], "b", "Final edges versus retained-attempt law")
    semantic_handles = [
        Patch(facecolor="#9CA3AF", edgecolor="none", alpha=0.78, label=r"final unique (mean $\pm$ SD)"),
        Line2D([0], [0], marker="x", color="black", ls="none", label="target Zipf"),
        Line2D([0], [0], marker="_", color="black", ls="none", markersize=7, label="retained-attempt mixture"),
    ]
    axes[0].legend(handles=semantic_handles, frameon=False, loc="upper right", fontsize=5.8, handlelength=1.5)
    size_handles = [
        tuple(
            Line2D(
                [0],
                [0],
                marker=SIZE_MARKERS[size],
                markerfacecolor=GAMMA_COLORS[gamma],
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=4.2,
                color="none",
            )
            for gamma in GAMMAS
        )
        for size in sizes
    ]
    size_labels = [rf"$|e|={size}$" for size in sizes]
    axes[1].legend(
        handles=size_handles,
        labels=size_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        frameon=False,
        loc="lower right",
        ncol=1,
        fontsize=5.5,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    gamma_handles = [Line2D([0], [0], color=GAMMA_COLORS[gamma], lw=2.2, label=rf"$\gamma={gamma}$") for gamma in GAMMAS]
    fig.legend(handles=gamma_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, handlelength=1.8)
    for ax in axes:
        clean_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.82, wspace=0.34)
    save_figure(fig, 3)
    print("Figure 3 complete", flush=True)


def make_figure4(workers: int) -> None:
    n, s_max, trials = 500, 5, 30
    rhos = np.arange(0.5, 5.01, 0.5)
    jobs, labels = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        for rho_index, rho in enumerate(rhos):
            for trial in range(trials):
                jobs.append((n, int(round(n * rho)), gamma, s_max, stable_seed(4, gamma_index, rho_index, trial), False, False, False))
                labels.append((gamma, rho, trial))
    results = parallel_map("Figure 4", trial_worker, jobs, workers)
    rows = []
    for (gamma, rho, trial), (_, metrics) in zip(labels, results):
        reference = rho * retained_attempt_probabilities(n, gamma, s_max)["retained_incidence_mean"]
        rows.append({"gamma": gamma, "rho": rho, "trial": trial, "mean_hyperedge_degree": metrics["mean_hyperedge_degree"], "attempt_level_reference": reference, "unique_fraction_attempts": metrics["unique_fraction_attempts"], "unique_fraction_retained": metrics["unique_fraction_retained"]})
    write_csv("figure4_degree_trials.csv", list(rows[0]), rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for gamma in GAMMAS:
        means, sds, references, fractions, fraction_sds = [], [], [], [], []
        for rho in rhos:
            subset = [row for row in rows if row["gamma"] == gamma and row["rho"] == rho]
            observed = np.asarray([row["mean_hyperedge_degree"] for row in subset])
            fraction = np.asarray([row["unique_fraction_attempts"] for row in subset])
            means.append(observed.mean()); sds.append(observed.std(ddof=1))
            references.append(subset[0]["attempt_level_reference"])
            fractions.append(fraction.mean()); fraction_sds.append(fraction.std(ddof=1))
        color, marker = GAMMA_COLORS[gamma], GAMMA_MARKERS[gamma]
        axes[0].errorbar(rhos, means, yerr=sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, capsize=1.4, capthick=0.7, elinewidth=0.7)
        axes[0].plot(rhos, references, color=color, ls="--", lw=0.9)
        axes[1].errorbar(references, means, yerr=sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, ls="none", capsize=1.4, capthick=0.7, elinewidth=0.7)
        axes[2].errorbar(rhos, fractions, yerr=fraction_sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, capsize=1.4, capthick=0.7, elinewidth=0.7)
    maximum = 1.04 * max(
        max(float(row["attempt_level_reference"]) for row in rows),
        max(float(row["mean_hyperedge_degree"]) for row in rows),
    )
    axes[1].plot([0, maximum], [0, maximum], color=OKABE_ITO["gray"], ls="--", lw=0.9, zorder=0)
    axes[0].set(xlabel=r"$\rho=m/n$", ylabel="Mean hypergraph vertex degree")
    axes[1].set(xlabel="Attempt-level reference", ylabel="Mean hypergraph vertex degree")
    axes[2].set(xlabel=r"$\rho=m/n$", ylabel="Unique hyperedges / attempts")
    axes[0].set_ylim(0, maximum)
    axes[1].set_xlim(0, maximum)
    axes[1].set_ylim(0, maximum)
    axes[1].set_aspect("equal", adjustable="box")
    axes[2].set_ylim(0.40, 0.74)
    panel_title(axes[0], "a", "Degree scaling")
    panel_title(axes[1], "b", "Deduplication gap")
    panel_title(axes[2], "c", "Unique-edge yield")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", label="final unique"),
            Line2D([0], [0], color="black", ls="--", label="attempt reference"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=5.8,
        handlelength=1.8,
    )
    gamma_handles = [Line2D([0], [0], color=GAMMA_COLORS[gamma], marker=GAMMA_MARKERS[gamma], label=rf"$\gamma={gamma}$") for gamma in GAMMAS]
    fig.legend(handles=gamma_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, handlelength=1.4)
    for ax in axes:
        clean_axes(ax)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.19, top=0.80, wspace=0.42)
    save_figure(fig, 4)
    print("Figure 4 complete", flush=True)


def make_figure5(workers: int) -> None:
    n, trials = 500, 80
    rhos = np.arange(0.5, 5.01, 0.5)
    prefix_counts = tuple(int(round(n * rho)) for rho in rhos)
    jobs, labels = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        for trial in range(trials):
            jobs.append((n, prefix_counts[-1], gamma, 5, stable_seed(5, 0, gamma_index, trial), prefix_counts, True, False, False))
            labels.append(("gamma", gamma, trial))
    # The gamma=2, s_max=5 baseline is generated once above and reused in
    # both panels.  Only the remaining size cutoffs require new realizations.
    for size_index, s_max in enumerate(SMAX_VALUES[:-1]):
        for trial in range(trials):
            jobs.append((n, prefix_counts[-1], 2.0, s_max, stable_seed(5, 1, size_index, trial), prefix_counts, True, False, False))
            labels.append(("smax", s_max, trial))
    results = parallel_map("Figure 5 coupled density trials", prefix_trial_worker, jobs, workers)
    rows = []
    for (panel, group, trial), metrics_by_rho in zip(labels, results):
        for rho, metrics in zip(rhos, metrics_by_rho):
            row = {"panel": panel, "group": group, "rho": rho, "trial": trial, "connected": int(metrics["connected"])}
            rows.append(row)
            if panel == "gamma" and float(group) == 2.0:
                rows.append({**row, "panel": "smax", "group": 5})
    write_csv("figure5_connectivity_trials.csv", ["panel", "group", "rho", "trial", "connected"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    for panel_index, (panel, groups) in enumerate((("gamma", GAMMAS), ("smax", SMAX_VALUES))):
        for group in groups:
            estimates, lows, highs = [], [], []
            for rho in rhos:
                values = [row["connected"] for row in rows if row["panel"] == panel and row["group"] == group and row["rho"] == rho]
                successes = int(sum(values))
                low, high = wilson_interval(successes, len(values))
                estimates.append(successes / len(values)); lows.append(low); highs.append(high)
            if panel == "gamma":
                color, marker, label = GAMMA_COLORS[float(group)], GAMMA_MARKERS[float(group)], rf"$\gamma={group}$"
            else:
                color, marker, label = SIZE_COLORS[int(group)], SIZE_MARKERS[int(group)], rf"$s_{{\max}}={group}$"
            axes[panel_index].plot(rhos, estimates, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.35, label=label)
            axes[panel_index].fill_between(rhos, lows, highs, color=color, alpha=0.10, linewidth=0)
        axes[panel_index].axhline(0.5, color="#9CA3AF", ls=":", lw=0.75, zorder=0)
        axes[panel_index].set_xlabel(r"$\rho=m/n$")
        axes[panel_index].set_ylabel(r"$P(\mathrm{connected})$")
        axes[panel_index].set_ylim(-0.03, 1.03)
        axes[panel_index].set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        axes[panel_index].legend(frameon=False, loc="upper left")
        clean_axes(axes[panel_index])
    panel_title(axes[0], "a", r"Tail exponent ($s_{\max}=5$)")
    panel_title(axes[1], "b", r"Size cutoff ($\gamma=2$)")
    axes[1].annotate(
        r"$s_{\max}=2$: 0/80 connected",
        xy=(4.8, 0.0),
        xytext=(3.15, 0.15),
        ha="center",
        va="center",
        fontsize=5.8,
        arrowprops=dict(arrowstyle="-", color=OKABE_ITO["gray"], lw=0.7),
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.90, wspace=0.28)
    save_figure(fig, 5)
    print("Figure 5 complete", flush=True)


def make_figure6(workers: int) -> None:
    n, rho, s_max, trials = 500, 3.0, 5, 50
    # Compare a central reference value with the upper endpoint of the
    # predeclared primary exponent sweep.
    considered = (2.0, 3.0)
    jobs, labels = [], []
    for gamma_index, gamma in enumerate(considered):
        for trial in range(trials):
            jobs.append((n, int(n * rho), gamma, s_max, stable_seed(6, gamma_index, trial), False, False, False))
            labels.append((gamma, trial))
    results = parallel_map("Figure 6", trial_worker, jobs, workers)
    rows, grouped = [], {gamma: [] for gamma in considered}
    for (gamma, trial), (_, metrics) in zip(labels, results):
        degrees = np.asarray(metrics["degree_sequence"], dtype=int)
        grouped[gamma].append(degrees)
        for vertex, degree in enumerate(degrees):
            rows.append({"gamma": gamma, "trial": trial, "vertex": vertex, "degree": int(degree)})
    write_csv("figure6_node_degrees.csv", ["gamma", "trial", "vertex", "degree"], rows)

    maximum = max(int(degrees.max()) for values in grouped.values() for degrees in values)
    support = np.arange(0, maximum + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True, sharey=True)
    for ax, gamma in zip(axes, considered):
        degrees = np.concatenate(grouped[gamma])
        mean, dispersion, total_variation = poisson_total_variation(degrees)
        trial_pmfs = np.vstack(
            [np.bincount(values, minlength=maximum + 1)[: maximum + 1] / len(values) for values in grouped[gamma]]
        )
        observed = trial_pmfs.mean(axis=0)
        critical = t.ppf(0.975, len(trial_pmfs) - 1)
        ci95 = critical * trial_pmfs.std(axis=0, ddof=1) / np.sqrt(len(trial_pmfs))
        ax.bar(support, observed, width=0.78, color=OKABE_ITO["sky_blue"], alpha=0.76, edgecolor="white", linewidth=0.3, label="Observed mean")
        ax.errorbar(support, observed, yerr=ci95, fmt="none", ecolor=OKABE_ITO["blue"], elinewidth=0.55, capsize=1.0, capthick=0.55, label="95% CI")
        ax.plot(support, poisson.pmf(support, mean), color=OKABE_ITO["black"], marker="o", ms=2.2, lw=0.95, label=rf"Poisson($\widehat\lambda={mean:.2f}$)")
        ax.text(0.97, 0.72, rf"$\mathrm{{Var}}/\mathrm{{Mean}}={dispersion:.2f}$" + "\n" + rf"$d_{{TV}}={total_variation:.3f}$", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
        ax.set_xlabel("Hypergraph vertex degree $d$")
        ax.set_xlim(-0.5, maximum + 0.5)
        ax.set_xticks(np.arange(0, maximum + 1, 3))
        clean_axes(ax)
    axes[0].set_ylabel("Probability")
    panel_title(axes[0], "a", rf"$\gamma={considered[0]}$")
    panel_title(axes[1], "b", rf"$\gamma={considered[1]}$")
    axes[0].legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.90, wspace=0.18)
    save_figure(fig, 6)
    print("Figure 6 complete", flush=True)


def null_worker(n, m, gamma, s_max, seed, null_seed):
    realization = generate_realization(n, m, gamma, s_max, seed)
    sat_graph = shadow_graph(n, realization.unique_edges)
    sat_clustering = nx.average_clustering(sat_graph, count_zeros=True)
    null_edges = generate_size_matched_null(n, realization.unique_sizes, null_seed)
    null_graph = shadow_graph(n, null_edges)
    null_clustering = nx.average_clustering(null_graph, count_zeros=True)
    return sat_clustering, null_clustering


def make_figure7(workers: int) -> None:
    n, trials = 300, 30
    rhos = np.arange(1.0, 5.01, 0.5)
    jobs, labels = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        for rho_index, rho in enumerate(rhos):
            for trial in range(trials):
                jobs.append((n, int(round(n * rho)), gamma, 5, stable_seed(7, 0, gamma_index, rho_index, trial), True, True, False))
                labels.append(("gamma", gamma, rho, trial))
    for size_index, s_max in enumerate(SMAX_VALUES[:-1]):
        for rho_index, rho in enumerate(rhos):
            for trial in range(trials):
                jobs.append((n, int(round(n * rho)), 2.0, s_max, stable_seed(7, 1, size_index, rho_index, trial), True, True, False))
                labels.append(("smax", s_max, rho, trial))
    results = parallel_map("Figure 7 parameter sweeps", trial_worker, jobs, workers)
    rows = []
    for (panel, group, rho, trial), (_, metrics) in zip(labels, results):
        row = {"panel": panel, "group": group, "rho": rho, "trial": trial, "model": "Sat-RSH", "clustering": metrics["clustering"]}
        rows.append(row)
        if panel == "gamma" and float(group) == 2.0:
            rows.append({**row, "panel": "smax", "group": 5})

    null_jobs, null_labels = [], []
    for rho_index, rho in enumerate(rhos):
        for trial in range(trials):
            null_jobs.append((n, int(round(n * rho)), 2.0, 5, stable_seed(7, 2, rho_index, trial), stable_seed(7, 3, rho_index, trial)))
            null_labels.append((rho, trial))
    null_results = parallel_map("Figure 7 size-matched null", null_worker, null_jobs, workers)
    for (rho, trial), (sat_value, null_value) in zip(null_labels, null_results):
        rows.append({"panel": "control", "group": 5, "rho": rho, "trial": trial, "model": "Sat-RSH", "clustering": sat_value})
        rows.append({"panel": "control", "group": 5, "rho": rho, "trial": trial, "model": "Size-matched null", "clustering": null_value})
    write_csv("figure7_clustering_trials.csv", ["panel", "group", "rho", "trial", "model", "clustering"], rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    configurations = (("gamma", GAMMAS), ("smax", SMAX_VALUES))
    for ax, (panel, groups) in zip(axes[:2], configurations):
        for group in groups:
            means, sds = [], []
            for rho in rhos:
                values = np.asarray([row["clustering"] for row in rows if row["panel"] == panel and row["group"] == group and row["rho"] == rho])
                means.append(values.mean()); sds.append(values.std(ddof=1))
            if panel == "gamma":
                color, marker, label = GAMMA_COLORS[float(group)], GAMMA_MARKERS[float(group)], rf"$\gamma={group}$"
            else:
                color, marker, label = SIZE_COLORS[int(group)], SIZE_MARKERS[int(group)], rf"$s_{{\max}}={group}$"
            ax.errorbar(rhos, means, yerr=sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, capsize=1.3, capthick=0.65, elinewidth=0.65, label=label)
        ax.set_xlabel(r"$\rho=m/n$")
        ax.set_ylabel("Mean local clustering")
        ax.legend(frameon=False, fontsize=5.8, handlelength=1.4, labelspacing=0.25)
        clean_axes(ax)
    axes[0].set_ylim(0.54, 0.66)
    axes[1].set_ylim(0.0, 0.66)
    panel_title(axes[0], "a", r"Tail exponent ($s_{\max}=5$)")
    panel_title(axes[1], "b", r"Size cutoff ($\gamma=2$)")
    null_styles = (("Sat-RSH", OKABE_ITO["blue"], "o", "-"), ("Size-matched null", OKABE_ITO["orange"], "s", "--"))
    null_curves = {}
    for model, color, marker, linestyle in null_styles:
        means, sds = [], []
        for rho in rhos:
            values = np.asarray([row["clustering"] for row in rows if row["panel"] == "control" and row["model"] == model and row["rho"] == rho])
            means.append(values.mean()); sds.append(values.std(ddof=1))
        null_curves[model] = np.asarray(means)
        axes[2].errorbar(rhos, means, yerr=sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, ls=linestyle, capsize=1.3, capthick=0.65, elinewidth=0.65, label=model)
    axes[2].fill_between(rhos, null_curves["Size-matched null"], null_curves["Sat-RSH"], color=OKABE_ITO["gray"], alpha=0.08, linewidth=0, zorder=0)
    axes[2].set_xlabel(r"$\rho=m/n$")
    axes[2].set_ylabel("Mean local clustering")
    axes[2].set_ylim(0.16, 0.66)
    panel_title(axes[2], "c", r"Size-matched control ($\gamma=2$)")
    axes[2].legend(frameon=False, fontsize=5.8, handlelength=1.7)
    clean_axes(axes[2])
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.19, top=0.90, wspace=0.38)
    save_figure(fig, 7)
    print("Figure 7 complete", flush=True)


def make_figure8(workers: int) -> None:
    trials, s_max = 25, 5
    rhos = np.arange(1.5, 5.01, 0.5)
    jobs, labels = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        prefix_counts = tuple(int(round(300 * rho)) for rho in rhos)
        for trial in range(trials):
            jobs.append((300, prefix_counts[-1], gamma, s_max, stable_seed(8, 0, gamma_index, trial), prefix_counts, True, False, True))
            labels.append(("gamma", gamma, trial))
    n_values = (200, 300, 500)
    for n_index, n in enumerate(n_values):
        if n == 300:
            continue
        prefix_counts = tuple(int(round(n * rho)) for rho in rhos)
        for trial in range(trials):
            jobs.append((n, prefix_counts[-1], 2.0, s_max, stable_seed(8, 1, n_index, trial), prefix_counts, True, False, True))
            labels.append(("n", n, trial))
    results = parallel_map("Figure 8 coupled density trials", prefix_trial_worker, jobs, workers)
    rows = []
    for (panel, group, trial), metrics_by_rho in zip(labels, results):
        for rho, metrics in zip(rhos, metrics_by_rho):
            row = {
                "panel": panel,
                "group": group,
                "rho": rho,
                "trial": trial,
                "apl_lcc": metrics["apl_lcc"],
                "lcc_fraction": metrics["lcc_fraction"],
                "connected": int(metrics["connected"]),
                "mean_hyperedge_degree": metrics["mean_hyperedge_degree"],
            }
            rows.append(row)
            if panel == "gamma" and float(group) == 2.0:
                rows.append({**row, "panel": "n", "group": 300})
    write_csv(
        "figure8_path_trials.csv",
        ["panel", "group", "rho", "trial", "apl_lcc", "lcc_fraction", "connected", "mean_hyperedge_degree"],
        rows,
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85), sharex="col")
    n_colors = {200: OKABE_ITO["vermillion"], 300: OKABE_ITO["blue"], 500: OKABE_ITO["bluish_green"]}
    n_markers = {200: "s", 300: "o", 500: "^"}
    for column, (panel, groups) in enumerate((("gamma", GAMMAS), ("n", n_values))):
        for group in groups:
            apl_means, apl_sds, lcc_means, lcc_sds = [], [], [], []
            for rho in rhos:
                subset = [row for row in rows if row["panel"] == panel and row["group"] == group and row["rho"] == rho]
                apl = np.asarray([row["apl_lcc"] for row in subset], dtype=float)
                lcc = np.asarray([row["lcc_fraction"] for row in subset], dtype=float)
                apl_means.append(np.nanmean(apl)); apl_sds.append(np.nanstd(apl, ddof=1))
                lcc_means.append(lcc.mean()); lcc_sds.append(lcc.std(ddof=1))
            if panel == "gamma":
                color, marker, label = GAMMA_COLORS[float(group)], GAMMA_MARKERS[float(group)], rf"$\gamma={group}$"
            else:
                color, marker, label = n_colors[int(group)], n_markers[int(group)], rf"$n={group}$"
            axes[0, column].errorbar(rhos, apl_means, yerr=apl_sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, capsize=1.3, capthick=0.65, elinewidth=0.65, label=label)
            axes[1, column].errorbar(rhos, lcc_means, yerr=lcc_sds, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.3, capsize=1.3, capthick=0.65, elinewidth=0.65, label=label)
        axes[0, column].set_ylabel("Mean LCC path length")
        axes[1, column].set_ylabel("LCC fraction")
        axes[1, column].set_xlabel(r"$\rho=m/n$")
        # Display the complete mean +/- SD range at low density.
        axes[1, column].set_ylim(0.45, 1.05)
        axes[1, column].set_yticks((0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        axes[0, column].legend(frameon=False, fontsize=5.8, handlelength=1.4)
        clean_axes(axes[0, column])
        clean_axes(axes[1, column])
    panel_title(axes[0, 0], "a", r"Paths by tail exponent ($n=300$)")
    panel_title(axes[0, 1], "b", r"Paths by system size ($\gamma=2$)")
    panel_title(axes[1, 0], "c", r"Component coverage by $\gamma$")
    panel_title(axes[1, 1], "d", "Component coverage by $n$")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.95, hspace=0.34, wspace=0.28)
    save_figure(fig, 8)
    print("Figure 8 complete", flush=True)


def write_manifest(arguments) -> None:
    manifest = {
        "model": "fixed-cap Sat-RSH with discard, truncation, and duplicate removal",
        "base_seed": BASE_SEED,
        "seed_rule": "numpy.random.SeedSequence([base_seed, figure, sweep, parameter_index, trial]); unused coordinates are omitted",
        "density_coupling": "Figures 5 and 8 generate one maximum attempt sequence per independent trial and evaluate nested prefixes for all rho values",
        "shared_baselines": "Identical parameter combinations appearing in multiple sweep panels of Figures 5, 7, and 8 reuse the same trial-level realizations",
        "figure6_comparison": "gamma=2 and gamma=3; gamma=3 is the upper endpoint of the primary exponent sweep",
        "figure7_control": "same vertices, number of unique hyperedges, and final unique-hyperedge size sequence; vertex degrees are not matched",
        "figure8_export": "apl_lcc, lcc_fraction, connected, and mean_hyperedge_degree are exported for every trial and density",
        "workers": arguments.workers,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "matplotlib", "networkx")
        },
        "reported_trials": {"figure3": 50, "figure4": 30, "figure5": 80, "figure6": 50, "figure7": 30, "figure8": 25},
        "figure_export": {
            "vector": "PDF with embedded TrueType fonts",
            "raster": "RGB PNG at 600 dpi",
            "maximum_width_mm": 183,
            "font_family": "Arial",
            "palette": "Okabe-Ito colour-vision-deficiency-safe palette",
        },
        "figures": [int(value) for value in arguments.figures],
    }
    with (DATA_DIR / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--figures", nargs="+", default=[str(value) for value in range(1, 9)], choices=[str(value) for value in range(1, 9)])
    arguments = parser.parse_args()
    requested = {int(value) for value in arguments.figures}
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    functions = {1: make_figure1, 2: make_figure2, 3: lambda: make_figure3(arguments.workers), 4: lambda: make_figure4(arguments.workers), 5: lambda: make_figure5(arguments.workers), 6: lambda: make_figure6(arguments.workers), 7: lambda: make_figure7(arguments.workers), 8: lambda: make_figure8(arguments.workers)}
    for number in range(1, 9):
        if number in requested:
            functions[number]()
    write_manifest(arguments)
    print("All requested fixed-cap Sat-RSH outputs are complete.", flush=True)


if __name__ == "__main__":
    main()
