"""Regenerate every Sat-RSH experiment, data export, and publication figure.

Run from this repository directory with

    python reproduce_all.py

The default run uses the trial counts reported in the manuscript.  All random
seeds are deterministic functions of ``BASE_SEED`` and the figure parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.stats import poisson, t

from sat_rsh_model import (
    GAMMA_COLORS,
    GAMMA_MARKERS,
    OKABE_ITO,
    SIZE_COLORS,
    SIZE_MARKERS,
    cap_angle,
    generate_degree_size_preserving_control,
    generate_realization,
    generate_size_matched_null,
    hypergraph_vertex_degrees,
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
BASE_SEED = 42
GAMMAS = (1.5, 2.0, 2.5, 3.0)
SMAX_VALUES = (2, 3, 4, 5)
ALL_FIGURES = tuple(range(1, 9))
REPOSITORY_URL = "https://github.com/hjwhhhh/Satellite-Random-Spherical-Hypergraphs"

plt.rcParams.update(publication_style())


def stable_seed(*coordinates: int) -> int:
    return int(np.random.SeedSequence([BASE_SEED, *coordinates]).generate_state(1)[0])


def save_figure(fig: plt.Figure, number: int) -> None:
    """Save editable vector art and a 600-dpi RGB review/submission bitmap."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"figure{number}.pdf"
    png_path = FIGURE_DIR / f"figure{number}.png"
    fig.savefig(
        pdf_path,
        format="pdf",
        dpi=600,
        bbox_inches=None,
        facecolor="white",
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=600,
        bbox_inches=None,
        facecolor="white",
        transparent=False,
    )
    # Matplotlib writes PNGs with an alpha channel even when the canvas is
    # opaque. Entropy recommends 8-bit RGB figures, so remove that redundant
    # channel while preserving 600-dpi metadata.
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(600, 600))
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


def integer_sequence_digest(values) -> str:
    """Stable SHA-256 digest for an integer sequence."""
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


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
    fig = plt.figure(figsize=(7.2, 3.55))
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
        subtitle = r",\;".join(f"N_{size}={counts[size]}" for size in range(2, 6))
        fig.text(
            0.018 + 0.5 * (panel - 1),
            0.985,
            rf"$\mathbf{{{chr(96 + panel)}}}\quad \gamma={gamma:.1f};\quad {subtitle}$",
            ha="left",
            va="top",
            fontsize=9.0,
        )
        ax.view_init(elev=18, azim=35)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 1), zoom=1.68)
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
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.004), handlelength=2.0)
    fig.subplots_adjust(left=-0.035, right=1.035, top=0.925, bottom=0.095, wspace=-0.12)
    save_figure(fig, 1)
    write_csv("figure1_hyperedges.csv", ["panel_gamma", "unique_edge_index", "size", "vertices"], rows)
    print("Figure 1 complete", flush=True)


def make_figure2() -> None:
    radius, h = 1.0, 0.35
    satellite_radius = radius + h
    theta = float(np.arccos(radius / satellite_radius))
    alpha = np.pi / 2.0 - theta
    satellite = np.array([0.0, satellite_radius])
    tangent_right = np.array([np.sin(theta), np.cos(theta)])
    tangent_left = np.array([-np.sin(theta), np.cos(theta)])

    # Restrained, colour-vision-deficiency-friendly semantic palette.  Geometry
    # is dark navy; the footprint and theta share teal; the two satellite-side
    # angles use purple and amber so the complementary relation is immediate.
    navy = "#25364A"
    blue = "#477FA8"
    teal = "#0F8F82"
    teal_light = "#DDF2EE"
    amber = "#D99100"
    purple = "#7967A8"
    earth_fill = "#EAF2F7"
    muted = "#7A8797"
    pale = "#D7E0E8"
    ink = "#17212B"

    def draw_satellite(axis, x: float, y: float, scale: float = 1.0) -> None:
        """Draw a compact vector satellite centered on the geometric point S."""
        body_w, body_h = 0.105 * scale, 0.075 * scale
        panel_w, panel_h = 0.115 * scale, 0.055 * scale
        gap = 0.020 * scale
        for sign in (-1, 1):
            panel_x = x + sign * (body_w / 2 + gap + panel_w / 2) - panel_w / 2
            panel = Rectangle(
                (panel_x, y - panel_h / 2),
                panel_w,
                panel_h,
                facecolor=blue,
                edgecolor=navy,
                linewidth=0.65,
                zorder=9,
            )
            axis.add_patch(panel)
            for fraction in (1 / 3, 2 / 3):
                axis.plot(
                    [panel_x + fraction * panel_w, panel_x + fraction * panel_w],
                    [y - panel_h / 2, y + panel_h / 2],
                    color="white",
                    lw=0.32,
                    alpha=0.85,
                    zorder=10,
                )
        axis.add_patch(
            Rectangle(
                (x - body_w / 2, y - body_h / 2),
                body_w,
                body_h,
                facecolor=navy,
                edgecolor="white",
                linewidth=0.55,
                zorder=11,
            )
        )
        axis.plot([x, x], [y + body_h / 2, y + 0.105 * scale], color=navy, lw=0.75, zorder=10)
        axis.scatter(x, y + 0.112 * scale, s=8.0 * scale, color=amber, edgecolor=navy, linewidth=0.45, zorder=11)

    def draw_angle(axis, center, radius_value, start, stop, color, label, label_radius):
        values = np.linspace(start, stop, 120)
        axis.plot(
            center[0] + radius_value * np.cos(values),
            center[1] + radius_value * np.sin(values),
            color=color,
            lw=1.55,
            solid_capstyle="round",
            zorder=7,
        )
        middle = 0.5 * (start + stop)
        axis.text(
            center[0] + label_radius * np.cos(middle),
            center[1] + label_radius * np.sin(middle),
            label,
            color=color,
            ha="center",
            va="center",
            weight="bold",
            zorder=8,
        )

    def draw_right_angle(axis, vertex, point_a, point_b, size=0.070):
        u = (point_a - vertex) / np.linalg.norm(point_a - vertex)
        v = (point_b - vertex) / np.linalg.norm(point_b - vertex)
        corners = np.vstack((vertex + size * u, vertex + size * (u + v), vertex + size * v))
        axis.plot(corners[:, 0], corners[:, 1], color=muted, lw=0.85, zorder=8)

    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.24, 1.0),
        left=0.018,
        right=0.988,
        bottom=0.035,
        top=0.92,
        wspace=0.075,
    )
    overview = fig.add_subplot(grid[0, 0])
    construction = fig.add_subplot(grid[0, 1])

    # (a) Physical cross-section: the cap is shown as an area, not only an arc,
    # and both tangent rays meet the exact geometric horizon points T_- and T_+.
    overview.add_patch(Circle((0, 0), radius, facecolor=earth_fill, edgecolor=navy, linewidth=1.15, zorder=0))
    cap_angles = np.linspace(np.pi / 2 - theta, np.pi / 2 + theta, 260)
    cap_points = np.column_stack((np.cos(cap_angles), np.sin(cap_angles)))
    overview.add_patch(
        Polygon(cap_points, closed=True, facecolor=teal_light, edgecolor="none", alpha=0.96, zorder=1)
    )
    overview.plot(cap_points[:, 0], cap_points[:, 1], color=teal, lw=3.1, solid_capstyle="round", zorder=5)

    for tangent in (tangent_left, tangent_right):
        overview.plot(
            [satellite[0], tangent[0]],
            [satellite[1], tangent[1]],
            color=navy,
            lw=1.05,
            zorder=3,
        )
    overview.plot([0, 0], [0, satellite_radius], ls=(0, (2, 2.4)), color=muted, lw=0.85, zorder=2)
    overview.plot([0, tangent_right[0]], [0, tangent_right[1]], color=navy, lw=0.9, zorder=2)
    draw_right_angle(overview, tangent_right, np.array([0.0, 0.0]), satellite, size=0.060)

    overview.annotate(
        "",
        xy=(-0.105, satellite_radius - 0.015),
        xytext=(-0.105, 1.015),
        arrowprops=dict(arrowstyle="<->", color=amber, lw=1.15, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    overview.text(-0.145, 1.175, "$h$", color=amber, ha="right", va="center", weight="bold")

    start_angle = np.arctan2(tangent_right[1], tangent_right[0])
    draw_angle(overview, np.array([0.0, 0.0]), 0.31, start_angle, np.pi / 2, teal, r"$\theta$", 0.40)

    station_angles = np.array([0.35, 0.67, 0.91, 1.13, 1.36, 1.58, 1.81, 2.03, 2.27, 2.53, 2.82])
    station_in_cap = np.abs(station_angles - np.pi / 2) <= theta
    station_radius = 1.008
    overview.scatter(
        station_radius * np.cos(station_angles[~station_in_cap]),
        station_radius * np.sin(station_angles[~station_in_cap]),
        s=15,
        color=muted,
        edgecolor="white",
        linewidth=0.45,
        zorder=7,
    )
    overview.scatter(
        station_radius * np.cos(station_angles[station_in_cap]),
        station_radius * np.sin(station_angles[station_in_cap]),
        s=21,
        color=teal,
        edgecolor="white",
        linewidth=0.55,
        zorder=7,
    )

    draw_satellite(overview, *satellite, scale=1.0)
    overview.text(0.0, 1.495, "Satellite $S$", color=ink, weight="bold", ha="center", va="bottom")
    overview.scatter(0, 0, s=13, color=navy, zorder=8)
    overview.scatter(0, 1, s=15, color=navy, zorder=8)
    overview.scatter(*tangent_right, s=18, color=navy, edgecolor="white", linewidth=0.45, zorder=8)
    overview.text(0.045, -0.035, "$O$", color=ink, ha="left", va="top")
    overview.text(0.045, 0.925, "$Q$ (nadir)", color=ink, ha="left", va="top")
    overview.text(tangent_right[0] + 0.045, tangent_right[1] + 0.015, "$T_+$", color=ink, ha="left", va="bottom")
    overview.annotate(
        "Horizon footprint",
        xy=(-0.72, 0.71),
        xytext=(-1.08, 1.18),
        color=teal,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=teal, lw=0.9),
    )
    panel_title(overview, "a", "Spherical horizon footprint", pad=2)
    overview.set_aspect("equal")
    overview.set_xlim(-1.16, 1.16)
    overview.set_ylim(-0.36, 1.58)
    overview.axis("off")

    # (b) Exact right-triangle construction.  It uses the same normalized
    # geometry as panel (a), so every angle and the right-angle marker are exact.
    origin = np.array([0.0, 0.0])
    sat = np.array([0.0, satellite_radius])
    tangent = tangent_right.copy()
    construction.plot([origin[0], sat[0]], [origin[1], sat[1]], color=navy, lw=1.2, zorder=2)
    construction.plot([origin[0], tangent[0]], [origin[1], tangent[1]], color=navy, lw=1.2, zorder=2)
    construction.plot([sat[0], tangent[0]], [sat[1], tangent[1]], color=navy, lw=1.35, zorder=3)
    construction.plot([sat[0], 1.22], [sat[1], sat[1]], ls=(0, (2, 2.4)), color=muted, lw=0.85, zorder=1)
    draw_right_angle(construction, tangent, origin, sat, size=0.075)

    line_of_sight_angle = np.arctan2(tangent[1] - sat[1], tangent[0] - sat[0])
    draw_angle(construction, origin, 0.23, start_angle, np.pi / 2, teal, r"$\theta$", 0.31)
    draw_angle(construction, sat, 0.18, -np.pi / 2, line_of_sight_angle, purple, r"$\alpha$", 0.245)
    draw_angle(
        construction,
        sat,
        0.31,
        line_of_sight_angle,
        0.0,
        amber,
        r"$\delta_{\mathrm{dep}}$",
        0.39,
    )

    draw_satellite(construction, *sat, scale=0.92)
    construction.scatter(*origin, s=15, color=navy, zorder=8)
    construction.scatter(*tangent, s=18, color=navy, edgecolor="white", linewidth=0.45, zorder=8)
    construction.text(-0.055, -0.055, "$O$", color=ink, ha="right", va="top")
    construction.text(-0.08, 1.49, "$S$", color=ink, weight="bold", ha="right", va="bottom")
    construction.text(tangent[0] + 0.05, tangent[1] - 0.005, "$T$", color=ink, ha="left", va="center")
    construction.text(-0.075, 0.69, "$R+h$", color=ink, rotation=90, ha="right", va="center")
    construction.text(0.31, 0.29, "$R=1$", color=ink, rotation=np.degrees(start_angle), ha="center", va="top")
    construction.text(1.18, 1.38, "Local horizon", color=muted, ha="right", va="bottom")
    construction.text(
        1.28,
        1.12,
        r"$\alpha+\theta=\pi/2$" + "\n" + r"$\delta_{\mathrm{dep}}=\theta$",
        color=ink,
        ha="right",
        va="top",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=pale, linewidth=0.8),
    )
    panel_title(construction, "b", "Exact angular construction", pad=2)
    construction.set_aspect("equal")
    construction.set_xlim(-0.25, 1.34)
    construction.set_ylim(-0.16, 1.60)
    construction.axis("off")

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
    unique_proportions: dict[float, list[np.ndarray]] = {gamma: [] for gamma in GAMMAS}
    retained_proportions: dict[float, list[np.ndarray]] = {gamma: [] for gamma in GAMMAS}
    for (gamma, trial), (realization, metrics) in zip(labels, results):
        unique_counts = np.asarray(
            [np.sum(realization.unique_sizes == size) for size in range(2, 6)],
            dtype=float,
        )
        retained_counts = np.asarray(
            [np.sum(realization.retained_sizes == size) for size in range(2, 6)],
            dtype=float,
        )
        unique_props = unique_counts / unique_counts.sum()
        retained_props = retained_counts / retained_counts.sum()
        unique_proportions[gamma].append(unique_props)
        retained_proportions[gamma].append(retained_props)
        for size, unique_count, unique_prop, retained_count, retained_prop in zip(
            range(2, 6),
            unique_counts.astype(int),
            unique_props,
            retained_counts.astype(int),
            retained_props,
        ):
            rows.append(
                {
                    "gamma": gamma,
                    "trial": trial,
                    "size": size,
                    "unique_count": unique_count,
                    "unique_proportion": unique_prop,
                    "retained_count": retained_count,
                    "retained_proportion": retained_prop,
                    "retained_attempts": metrics["retained_attempts"],
                    "unique_edges": metrics["unique_edges"],
                }
            )
    write_csv(
        "figure3_size_distribution_trials.csv",
        [
            "gamma",
            "trial",
            "size",
            "unique_count",
            "unique_proportion",
            "retained_count",
            "retained_proportion",
            "retained_attempts",
            "unique_edges",
        ],
        rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    sizes = np.arange(2, 6)
    width = 0.18
    calibration_x, calibration_y = [], []
    for gamma_index, gamma in enumerate(GAMMAS):
        values = np.vstack(unique_proportions[gamma])
        retained_values = np.vstack(retained_proportions[gamma])
        means, standard_deviations = values.mean(axis=0), values.std(axis=0, ddof=1)
        retained_means = retained_values.mean(axis=0)
        retained_standard_errors = retained_values.std(axis=0, ddof=1) / np.sqrt(trials)
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
                retained_means[size_index],
                yerr=t.ppf(0.975, trials - 1) * retained_standard_errors[size_index],
                fmt=SIZE_MARKERS[int(sizes[size_index])],
                color=GAMMA_COLORS[gamma],
                markerfacecolor="white",
                markeredgewidth=0.75,
                ls="none",
                capsize=1.5,
                elinewidth=0.6,
                capthick=0.6,
                zorder=3,
            )
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
    axes[1].set_ylabel("Observed probability")
    panel_title(axes[0], "a", "Target and realized size laws")
    panel_title(axes[1], "b", "Mixture check and deduplication")
    semantic_handles = [
        Patch(facecolor="#9CA3AF", edgecolor="none", alpha=0.78, label=r"final unique (mean $\pm$ SD)"),
        Line2D([0], [0], marker="x", color="black", ls="none", label="target Zipf"),
        Line2D([0], [0], marker="_", color="black", ls="none", markersize=7, label="retained-attempt mixture"),
    ]
    axes[0].legend(handles=semantic_handles, frameon=False, loc="upper right", fontsize=7.2, handlelength=1.5)
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
    size_legend = axes[1].legend(
        handles=size_handles,
        labels=size_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        frameon=False,
        loc="lower right",
        ncol=1,
        fontsize=7.1,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    axes[1].add_artist(size_legend)
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", color="none", label="retained attempts (95% CI)"),
            Line2D([0], [0], marker="o", markerfacecolor=OKABE_ITO["gray"], markeredgecolor="white", color="none", label=r"final unique (mean $\pm$ SD)"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=7.0,
        handletextpad=0.35,
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
        rows.append({"gamma": gamma, "rho": rho, "trial": trial, "mean_hypergraph_vertex_degree": metrics["mean_hypergraph_vertex_degree"], "attempt_level_reference": reference, "unique_fraction_attempts": metrics["unique_fraction_attempts"], "unique_fraction_retained": metrics["unique_fraction_retained"]})
    write_csv("figure4_degree_trials.csv", list(rows[0]), rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for gamma in GAMMAS:
        means, sds, references, fractions, fraction_sds = [], [], [], [], []
        for rho in rhos:
            subset = [row for row in rows if row["gamma"] == gamma and row["rho"] == rho]
            observed = np.asarray([row["mean_hypergraph_vertex_degree"] for row in subset])
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
        max(float(row["mean_hypergraph_vertex_degree"]) for row in rows),
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
        fontsize=7.2,
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
    diagnostic_rows = []
    for ax, gamma in zip(axes, considered):
        degrees = np.concatenate(grouped[gamma])
        mean, dispersion, total_variation = poisson_total_variation(degrees)
        trial_pmfs = np.vstack(
            [np.bincount(values, minlength=maximum + 1)[: maximum + 1] / len(values) for values in grouped[gamma]]
        )
        trial_counts = trial_pmfs * n
        bootstrap_rng = np.random.default_rng(stable_seed(6, 9, int(gamma * 10)))
        bootstrap_resamples = 10000
        bootstrap_weights = bootstrap_rng.multinomial(
            len(grouped[gamma]),
            np.full(len(grouped[gamma]), 1.0 / len(grouped[gamma])),
            size=bootstrap_resamples,
        )
        bootstrap_counts = bootstrap_weights @ trial_counts
        bootstrap_probabilities = bootstrap_counts / bootstrap_counts.sum(axis=1, keepdims=True)
        bootstrap_means = bootstrap_probabilities @ support
        bootstrap_second_moments = bootstrap_probabilities @ (support**2)
        bootstrap_variances = bootstrap_second_moments - bootstrap_means**2
        bootstrap_dispersions = bootstrap_variances / bootstrap_means
        fitted = poisson.pmf(support[None, :], bootstrap_means[:, None])
        fitted_tail = np.maximum(0.0, 1.0 - fitted.sum(axis=1))
        bootstrap_tv = 0.5 * (
            np.abs(bootstrap_probabilities - fitted).sum(axis=1) + fitted_tail
        )
        dispersion_low, dispersion_high = np.quantile(
            bootstrap_dispersions, (0.025, 0.975)
        )
        tv_low, tv_high = np.quantile(bootstrap_tv, (0.025, 0.975))
        diagnostic_rows.append(
            {
                "gamma": gamma,
                "realizations": len(grouped[gamma]),
                "vertices_per_realization": n,
                "mean_degree": mean,
                "variance_to_mean": dispersion,
                "dispersion_ci_low": dispersion_low,
                "dispersion_ci_high": dispersion_high,
                "total_variation": total_variation,
                "tv_ci_low": tv_low,
                "tv_ci_high": tv_high,
                "bootstrap_resamples": bootstrap_resamples,
                "resampling_unit": "realization",
            }
        )
        observed = trial_pmfs.mean(axis=0)
        critical = t.ppf(0.975, len(trial_pmfs) - 1)
        ci95 = critical * trial_pmfs.std(axis=0, ddof=1) / np.sqrt(len(trial_pmfs))
        ax.bar(support, observed, width=0.78, color=OKABE_ITO["sky_blue"], alpha=0.76, edgecolor="white", linewidth=0.3, label="Observed mean")
        ax.errorbar(support, observed, yerr=ci95, fmt="none", ecolor=OKABE_ITO["blue"], elinewidth=0.55, capsize=1.0, capthick=0.55, label="95% CI")
        ax.plot(support, poisson.pmf(support, mean), color=OKABE_ITO["black"], marker="o", ms=2.2, lw=0.95, label=rf"Poisson($\widehat\lambda={mean:.2f}$)")
        ax.text(0.97, 0.72, rf"$\mathrm{{Var}}/\mathrm{{Mean}}={dispersion:.2f}$" + "\n" + rf"$d_{{TV}}={total_variation:.3f}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
        ax.set_xlabel("Hypergraph vertex degree $d$")
        ax.set_xlim(-0.5, maximum + 0.5)
        ax.set_xticks(np.arange(0, maximum + 1, 3))
        clean_axes(ax)
    axes[0].set_ylabel("Probability")
    panel_title(axes[0], "a", rf"$\gamma={considered[0]}$")
    panel_title(axes[1], "b", rf"$\gamma={considered[1]}$")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.2, handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.90, wspace=0.18)
    save_figure(fig, 6)
    write_csv(
        "figure6_degree_diagnostics.csv",
        list(diagnostic_rows[0]),
        diagnostic_rows,
    )
    print("Figure 6 complete", flush=True)


def null_worker(n, m, gamma, s_max, seed, null_seed, rewire_stage_seeds):
    realization = generate_realization(n, m, gamma, s_max, seed)
    sat_graph = shadow_graph(n, realization.unique_edges)
    sat_clustering = nx.average_clustering(sat_graph, count_zeros=True)
    null_edges = generate_size_matched_null(n, realization.unique_sizes, null_seed)
    null_graph = shadow_graph(n, null_edges)
    null_clustering = nx.average_clustering(null_graph, count_zeros=True)
    output = {
        "sat_clustering": sat_clustering,
        "size_null_clustering": null_clustering,
        "q": len(realization.unique_edges),
        "size_sequence_sha256": integer_sequence_digest(
            sorted(realization.unique_sizes)
        ),
        "sat_degree_sequence_sha256": integer_sequence_digest(
            hypergraph_vertex_degrees(n, realization.unique_edges)
        ),
        "size_null_degree_sequence_sha256": integer_sequence_digest(
            hypergraph_vertex_degrees(n, null_edges)
        ),
        "rewiring": [],
    }
    if rewire_stage_seeds is not None:
        rewired_edges = realization.unique_edges
        cumulative_successful = 0
        cumulative_attempted = 0
        previous_checkpoint = 0
        for checkpoint, rewire_seed in zip((20, 50, 100, 200), rewire_stage_seeds):
            stage_successful = (checkpoint - previous_checkpoint) * len(rewired_edges)
            rewired_edges, diagnostics = generate_degree_size_preserving_control(
                n,
                rewired_edges,
                rewire_seed,
                stage_successful,
            )
            cumulative_successful += int(diagnostics["successful_swaps"])
            cumulative_attempted += int(diagnostics["attempted_swaps"])
            rewired_graph = shadow_graph(n, rewired_edges)
            output["rewiring"].append(
                {
                    "checkpoint_swaps_per_edge": checkpoint,
                    "rewire_seed": rewire_seed,
                    "stage_successful_swaps": int(diagnostics["successful_swaps"]),
                    "stage_attempted_swaps": int(diagnostics["attempted_swaps"]),
                    "cumulative_successful_swaps": cumulative_successful,
                    "cumulative_attempted_swaps": cumulative_attempted,
                    "stage_acceptance_rate": diagnostics["acceptance_rate"],
                    "clustering": nx.average_clustering(
                        rewired_graph, count_zeros=True
                    ),
                    "degree_sequence_sha256": integer_sequence_digest(
                        hypergraph_vertex_degrees(n, rewired_edges)
                    ),
                    "size_sequence_sha256": integer_sequence_digest(
                        sorted(map(len, rewired_edges))
                    ),
                }
            )
            previous_checkpoint = checkpoint
    return output


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
            sat_seed = stable_seed(7, 2, rho_index, trial)
            size_null_seed = stable_seed(7, 3, rho_index, trial)
            rewire_seeds = (
                tuple(stable_seed(7, 4, rho_index, trial, stage) for stage in range(4))
                if np.isclose(rho, 3.0)
                else None
            )
            null_jobs.append(
                (
                    n,
                    int(round(n * rho)),
                    2.0,
                    5,
                    sat_seed,
                    size_null_seed,
                    rewire_seeds,
                )
            )
            null_labels.append((rho, trial, sat_seed, size_null_seed))
    null_results = parallel_map(
        "Figure 7 non-geometric controls", null_worker, null_jobs, workers
    )
    rewiring_rows = []
    for (rho, trial, sat_seed, size_null_seed), result in zip(
        null_labels, null_results
    ):
        rows.append({"panel": "control", "group": 5, "rho": rho, "trial": trial, "model": "Sat-RSH", "clustering": result["sat_clustering"]})
        rows.append({"panel": "control", "group": 5, "rho": rho, "trial": trial, "model": "Size-matched null", "clustering": result["size_null_clustering"]})
        for checkpoint in result["rewiring"]:
            rewiring_rows.append(
                {
                    "rho": rho,
                    "trial": trial,
                    "q": result["q"],
                    "sat_seed": sat_seed,
                    "size_null_seed": size_null_seed,
                    **checkpoint,
                    "sat_degree_sequence_sha256": result[
                        "sat_degree_sequence_sha256"
                    ],
                    "sat_size_sequence_sha256": result[
                        "size_sequence_sha256"
                    ],
                }
            )
        if result["rewiring"]:
            rows.append(
                {
                    "panel": "control",
                    "group": 5,
                    "rho": rho,
                    "trial": trial,
                    "model": "Degree-and-size-preserving rewired",
                    "clustering": result["rewiring"][-1]["clustering"],
                }
            )
    write_csv("figure7_clustering_trials.csv", ["panel", "group", "rho", "trial", "model", "clustering"], rows)
    write_csv(
        "figure7_rewiring_diagnostics.csv",
        list(rewiring_rows[0]),
        rewiring_rows,
    )

    summary_rows = []
    for model in (
        "Size-matched null",
        "Degree-and-size-preserving rewired",
    ):
        model_rhos = sorted(
            {
                float(row["rho"])
                for row in rows
                if row["panel"] == "control" and row["model"] == model
            }
        )
        for rho in model_rhos:
            sat_values = np.asarray(
                [
                    row["clustering"]
                    for row in rows
                    if row["panel"] == "control"
                    and row["model"] == "Sat-RSH"
                    and row["rho"] == rho
                ]
            )
            control_values = np.asarray(
                [
                    row["clustering"]
                    for row in rows
                    if row["panel"] == "control"
                    and row["model"] == model
                    and row["rho"] == rho
                ]
            )
            differences = sat_values - control_values
            critical = t.ppf(0.975, len(differences) - 1)
            half_width = critical * differences.std(ddof=1) / np.sqrt(
                len(differences)
            )
            summary_rows.append(
                {
                    "rho": rho,
                    "model": model,
                    "paired_trials": len(differences),
                    "sat_mean": sat_values.mean(),
                    "control_mean": control_values.mean(),
                    "mean_paired_difference": differences.mean(),
                    "ci95_low": differences.mean() - half_width,
                    "ci95_high": differences.mean() + half_width,
                    "positive_pairs": int(np.sum(differences > 0)),
                }
            )
    write_csv(
        "figure7_control_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )

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
        ax.legend(frameon=False, fontsize=7.2, handlelength=1.4, labelspacing=0.25)
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
    rewired_values = np.asarray(
        [
            row["clustering"]
            for row in rows
            if row["panel"] == "control"
            and row["model"] == "Degree-and-size-preserving rewired"
        ]
    )
    axes[2].errorbar(
        [3.0],
        [rewired_values.mean()],
        yerr=[rewired_values.std(ddof=1)],
        color=OKABE_ITO["bluish_green"],
        marker="^",
        markeredgecolor="white",
        markeredgewidth=0.35,
        ls="none",
        capsize=1.8,
        capthick=0.7,
        elinewidth=0.7,
        label="Degree+size-preserving",
        zorder=4,
    )
    axes[2].set_xlabel(r"$\rho=m/n$")
    axes[2].set_ylabel("Mean local clustering")
    axes[2].set_ylim(0.16, 0.66)
    panel_title(axes[2], "c", "Non-geometric controls")
    axes[2].legend(frameon=False, fontsize=7.0, handlelength=1.6)
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
                "mean_hypergraph_vertex_degree": metrics["mean_hypergraph_vertex_degree"],
            }
            rows.append(row)
            if panel == "gamma" and float(group) == 2.0:
                rows.append({**row, "panel": "n", "group": 300})
    write_csv(
        "figure8_path_trials.csv",
        ["panel", "group", "rho", "trial", "apl_lcc", "lcc_fraction", "connected", "mean_hypergraph_vertex_degree"],
        rows,
    )

    density_change_rows = []
    for gamma in GAMMAS:
        start_rows = sorted(
            (
                row
                for row in rows
                if row["panel"] == "gamma"
                and row["group"] == gamma
                and row["rho"] == rhos[0]
            ),
            key=lambda row: row["trial"],
        )
        end_rows = sorted(
            (
                row
                for row in rows
                if row["panel"] == "gamma"
                and row["group"] == gamma
                and row["rho"] == rhos[-1]
            ),
            key=lambda row: row["trial"],
        )
        summary = {
            "gamma": gamma,
            "rho_start": rhos[0],
            "rho_end": rhos[-1],
            "paired_trials": len(start_rows),
        }
        for metric in (
            "apl_lcc",
            "lcc_fraction",
            "mean_hypergraph_vertex_degree",
        ):
            differences = np.asarray(
                [end[metric] - start[metric] for start, end in zip(start_rows, end_rows)],
                dtype=float,
            )
            critical = t.ppf(0.975, len(differences) - 1)
            half_width = critical * differences.std(ddof=1) / np.sqrt(
                len(differences)
            )
            summary[f"{metric}_mean_change"] = differences.mean()
            summary[f"{metric}_ci95_low"] = differences.mean() - half_width
            summary[f"{metric}_ci95_high"] = differences.mean() + half_width
        density_change_rows.append(summary)
    write_csv(
        "figure8_density_change_summary.csv",
        list(density_change_rows[0]),
        density_change_rows,
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
        axes[0, column].legend(frameon=False, fontsize=7.2, handlelength=1.4)
        clean_axes(axes[0, column])
        clean_axes(axes[1, column])
    panel_title(axes[0, 0], "a", r"Paths by tail exponent ($n=300$)")
    panel_title(axes[0, 1], "b", r"Paths by system size ($\gamma=2$)")
    panel_title(axes[1, 0], "c", r"Component coverage by $\gamma$")
    panel_title(axes[1, 1], "d", "Component coverage by $n$")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.95, hspace=0.34, wspace=0.28)
    save_figure(fig, 8)
    print("Figure 8 complete", flush=True)


def write_manifest(arguments, requested_figures: list[int]) -> Path:
    output_hashes = {}
    for number in requested_figures:
        candidates = sorted(DATA_DIR.glob(f"figure{number}_*.csv")) + [
            FIGURE_DIR / f"figure{number}.pdf",
            FIGURE_DIR / f"figure{number}.png",
        ]
        for path in candidates:
            if path.is_file():
                output_hashes[str(path.relative_to(HERE)).replace("\\", "/")] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
    manifest = {
        "analysis_version": 2,
        "model": "fixed-cap Sat-RSH with discard, truncation, and duplicate removal",
        "base_seed": BASE_SEED,
        "repository": REPOSITORY_URL,
        "license": "MIT",
        "seed_rule": "stable_seed(*coordinates) uses numpy.random.SeedSequence([base_seed, *coordinates]); coordinate tuples are explicit at every call site and may include sweep, parameter, density, trial, control, and rewiring-stage indices",
        "density_coupling": "Figures 5 and 8 generate one maximum attempt sequence per independent trial and evaluate nested prefixes for all rho values",
        "shared_baselines": "Figure 5 panels (a,b), Figure 7 panels (a,b), and the relevant Figure 8 panels reuse identical trial-level realizations for shared parameter combinations; Figure 7(c) uses an independent paired-control stream",
        "figure6_comparison": "gamma=2 and gamma=3; gamma=3 is the upper endpoint of the primary exponent sweep",
        "figure6_uncertainty": "10000 realization-block bootstrap resamples for variance-to-mean and total-variation diagnostics",
        "figure7_controls": {
            "size_matched": "same vertices, number of unique hyperedges, and complete final unique-hyperedge size sequence; vertex degrees are not matched",
            "degree_and_size_preserving": "at rho=3, incidence switches preserve every vertex degree, every hyperedge size, edge count, and simplicity; checkpoints after 20q, 50q, 100q, and 200q accepted switches, with q the number of unique hyperedges; no uniform-sampling claim",
        },
        "figure8_export": "apl_lcc, lcc_fraction, connected, and mean_hypergraph_vertex_degree are exported for every trial and density",
        "parameter_grids": {
            "figure1": {"n": 100, "rho": 1.6, "gamma": [2.0, 4.0], "s_max": 5},
            "figure2": {"h_over_R": 0.35},
            "figure3": {"n": 500, "rho": 3.0, "gamma": list(GAMMAS), "s_max": 5},
            "figure4": {"n": 500, "rho": list(np.arange(0.5, 5.01, 0.5)), "gamma": list(GAMMAS), "s_max": 5},
            "figure5": {"n": 500, "rho": list(np.arange(0.5, 5.01, 0.5)), "gamma": list(GAMMAS), "s_max": list(SMAX_VALUES)},
            "figure6": {"n": 500, "rho": 3.0, "gamma": [2.0, 3.0], "s_max": 5},
            "figure7": {"n": 300, "rho": list(np.arange(1.0, 5.01, 0.5)), "gamma": list(GAMMAS), "s_max": list(SMAX_VALUES), "degree_size_control_rho": 3.0},
            "figure8": {"n": [200, 300, 500], "rho": list(np.arange(1.5, 5.01, 0.5)), "gamma": list(GAMMAS), "s_max": 5},
        },
        "workers": arguments.workers,
        "command": ["python", "reproduce_all.py", "--workers", arguments.workers, "--figures", *requested_figures],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "matplotlib", "networkx", "Pillow")
        },
        "reported_trials": {"figure3": 50, "figure4": 30, "figure5": 80, "figure6": 50, "figure7": 30, "figure8": 25},
        "figure_export": {
            "vector": "PDF with embedded TrueType fonts",
            "raster": "RGB PNG at 600 dpi",
            "maximum_width_mm": 183,
            "font_family": "Arial",
            "palette": (
                "colour-vision-deficiency-aware palettes: Okabe-Ito for statistical "
                "panels; restrained navy/teal/amber/purple semantics for Figure 2"
            ),
        },
        "figures": requested_figures,
        "output_sha256": output_hashes,
    }
    filename = (
        "experiment_manifest.json"
        if requested_figures == list(ALL_FIGURES)
        else "last_run_manifest.json"
    )
    output_path = DATA_DIR / filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--figures",
        nargs="+",
        default=[str(value) for value in ALL_FIGURES],
        choices=[str(value) for value in ALL_FIGURES],
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be a positive integer")
    requested = sorted({int(value) for value in arguments.figures})
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    functions = {1: make_figure1, 2: make_figure2, 3: lambda: make_figure3(arguments.workers), 4: lambda: make_figure4(arguments.workers), 5: lambda: make_figure5(arguments.workers), 6: lambda: make_figure6(arguments.workers), 7: lambda: make_figure7(arguments.workers), 8: lambda: make_figure8(arguments.workers)}
    for number in range(1, 9):
        if number in requested:
            functions[number]()
    manifest_path = write_manifest(arguments, requested)
    print(f"Run manifest: {manifest_path.relative_to(HERE)}", flush=True)
    print("All requested fixed-cap Sat-RSH outputs are complete.", flush=True)


if __name__ == "__main__":
    main()
