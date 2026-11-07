# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
plots.py
--------
Builds the figures used in the README/report and the paper from the
CSVs and timelines produced by experiments.py.
"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

GREEN, RED, AMBER, INDIGO, GRAY = "#2F6E4F", "#8C2F2B", "#9A6B1E", "#333F6B", "#5B5E66"


def fig_conditioned_rates(agg):
    """Bar chart: reactive vs predictive vs a representative static pool
    (keep_warm_10) across three traffic conditions."""
    rows = agg.set_index("policy")
    policies = ["reactive", "keep_warm_10", "predictive"]
    labels = ["Reactive\n(baseline)", "Keep-warm\n(N=10)", "Predictive"]
    conditions = ["onset_cold_start_rate_mean", "ramp_transition_cold_start_rate_mean", "peak_sudden_cold_start_rate_mean"]
    cond_labels = ["Onset\n(first 10s, all bursts)", "Ramp transition\n(ramping bursts only)", "Peak\n(sudden bursts only)"]

    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = [GRAY, AMBER, INDIGO]
    for i, pol in enumerate(policies):
        vals = [rows.loc[pol, c] * 100 for c in conditions]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_ylabel("Cold-start rate (%)")
    ax.set_title("Cold-start rate by traffic condition and policy")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "conditioned_rates.png"), dpi=170)
    plt.close(fig)


def fig_cost_frontier(agg):
    """Cost (container-hours) vs. ramp-transition cold-start rate, showing
    the keep-warm sweep as a frontier and reactive/predictive as points."""
    rows = agg.set_index("policy")
    keep_warm_ns = [2, 5, 10, 15, 20, 30, 45]
    xs = [rows.loc[f"keep_warm_{n}", "container_hours_mean"] for n in keep_warm_ns]
    ys = [rows.loc[f"keep_warm_{n}", "ramp_transition_cold_start_rate_mean"] * 100 for n in keep_warm_ns]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(xs, ys, "o-", color=GRAY, linewidth=1.5, markersize=5, label="Static keep-warm pool (sweep over N)")
    for n, x, y in zip(keep_warm_ns, xs, ys):
        ax.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=GRAY)

    rx, ry = rows.loc["reactive", "container_hours_mean"], rows.loc["reactive", "ramp_transition_cold_start_rate_mean"] * 100
    px, py = rows.loc["predictive", "container_hours_mean"], rows.loc["predictive", "ramp_transition_cold_start_rate_mean"] * 100
    ax.scatter([rx], [ry], color=RED, s=90, zorder=5, label="Reactive (baseline)")
    ax.scatter([px], [py], color=GREEN, s=90, zorder=5, marker="D", label="Predictive")

    ax.set_xlabel("Mean container-hours (cost proxy)")
    ax.set_ylabel("Ramp-transition cold-start rate (%)")
    ax.set_title("Cost vs. ramp-transition cold-start rate")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cost_frontier.png"), dpi=170)
    plt.close(fig)


def fig_timeline(timelines):
    """Container count over time around the first ramping burst
    (start=6000, ramp=30, plateau=400) for reactive vs predictive vs keep_warm_10."""
    window = (5900, 6500)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    styles = {"reactive": (RED, "-"), "predictive": (GREEN, "-"), "keep_warm_10": (GRAY, "--")}
    labels = {"reactive": "Reactive", "predictive": "Predictive", "keep_warm_10": "Keep-warm (N=10)"}
    for pol, (color, ls) in styles.items():
        tl = timelines[pol]
        ts = [row[0] for row in tl if window[0] <= row[0] <= window[1]]
        containers = [row[2] for row in tl if window[0] <= row[0] <= window[1]]
        ax.plot([t - 6000 for t in ts], containers, ls, color=color, label=labels[pol], linewidth=1.6)

    ax.axvspan(0, 30, color=AMBER, alpha=0.15, label="Ramp (0-30s)")
    ax.set_xlabel("Seconds relative to burst start")
    ax.set_ylabel("Warm container count")
    ax.set_title("Container count around a ramping burst (15x over 30s, then sustained)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "timeline_ramp.png"), dpi=170)
    plt.close(fig)


def fig_sudden_timeline(timelines):
    """Container count around a sudden burst (start=9000, mult=40x, ramp=0)."""
    window = (8950, 9200)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    styles = {"reactive": (RED, "-"), "predictive": (GREEN, "-"), "keep_warm_10": (GRAY, "--")}
    labels = {"reactive": "Reactive", "predictive": "Predictive", "keep_warm_10": "Keep-warm (N=10)"}
    for pol, (color, ls) in styles.items():
        tl = timelines[pol]
        ts = [row[0] for row in tl if window[0] <= row[0] <= window[1]]
        containers = [row[2] for row in tl if window[0] <= row[0] <= window[1]]
        ax.plot([t - 9000 for t in ts], containers, ls, color=color, label=labels[pol], linewidth=1.6)
    ax.axvline(0, color=RED, linestyle=":", linewidth=1, label="Burst starts (instantaneous, 40x)")
    ax.set_xlabel("Seconds relative to burst start")
    ax.set_ylabel("Warm container count")
    ax.set_title("Container count around a sudden, unannounced burst (40x, no ramp)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "timeline_sudden.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    agg = pd.read_csv(os.path.join(OUT_DIR, "aggregated_results.csv"))
    with open(os.path.join(OUT_DIR, "timelines.json")) as f:
        timelines = json.load(f)

    fig_conditioned_rates(agg)
    fig_cost_frontier(agg)
    fig_timeline(timelines)
    fig_sudden_timeline(timelines)
    print("Figures written to", OUT_DIR)
