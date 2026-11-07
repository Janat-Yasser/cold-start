# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
main.py
-------
Runs the full pipeline: experiment sweep -> figures.

Usage: python main.py
"""
from src.experiments import run_all
from src.plots import fig_conditioned_rates, fig_cost_frontier, fig_timeline, fig_sudden_timeline
import json
import os

if __name__ == "__main__":
    df, agg, timelines = run_all()

    fig_conditioned_rates(agg)
    fig_cost_frontier(agg)
    fig_timeline(timelines)
    fig_sudden_timeline(timelines)

    print("\n=== Aggregated results ===")
    cols = ["policy", "onset_cold_start_rate_mean", "ramp_transition_cold_start_rate_mean",
            "peak_sudden_cold_start_rate_mean", "container_hours_mean"]
    print(agg[cols].round(4).to_string(index=False))
    print("\nFigures and CSVs written to outputs/")
