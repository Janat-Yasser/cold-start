"""
experiments.py
--------------
Defines the shared workload (traffic pattern) and runs every policy
across multiple random seeds, saving per-run and aggregated results.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulate import (
    run_simulation, make_arrival_rate_fn,
    ReactivePolicy, KeepWarmPolicy, PredictivePolicy,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

TOTAL_SECONDS = 4 * 3600  # 4 simulated hours
BASE_RATE = 1.0            # requests/sec during normal (idle-ish) traffic
# A mix of sudden, unannounced spikes (ramp=0) and sharply-ramping
# traffic build-ups (ramp>0 but fast), the two archetypal patterns
# discussed in the autoscaling literature: a viral link / retry storm
# (sudden) versus a fast marketing-push ramp-up (ramping). Service times
# are set heavier than the earlier revision (see SERVICE_KW below) so
# that bursts require double-digit concurrent containers, giving the
# provisioning *policy* something real to get right or wrong.
BURSTS = [
    dict(start=1800,  ramp=0,  plateau=180, multiplier=25, kind="sudden"),
    dict(start=6000,  ramp=30, plateau=400, multiplier=15, kind="ramping"),
    dict(start=9000,  ramp=0,  plateau=120, multiplier=40, kind="sudden"),
    dict(start=11000, ramp=45, plateau=500, multiplier=20, kind="ramping"),
    dict(start=13500, ramp=0,  plateau=90,  multiplier=50, kind="sudden"),
]
ONSET_WINDOW = 10   # seconds after a burst starts considered "onset"
PEAK_WINDOW = 15    # seconds, measured once a burst reaches full intensity
SEEDS = [1, 2, 3, 4, 5]

SERVICE_KW = dict(mean=0.80, sd=0.20, minimum=0.10)
COLDSTART_KW = dict(mean=1.0, sd=0.30, minimum=0.20)


def build_policies():
    policies = [ReactivePolicy()]
    for n in [2, 5, 10, 15, 20, 30, 45]:
        policies.append(KeepWarmPolicy(n))
    policies.append(PredictivePolicy(window=20, lead_time=10, safety_factor=1.3,
                                      mean_service_time=SERVICE_KW["mean"]))
    return policies


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    rate_fn = make_arrival_rate_fn(base_rate=BASE_RATE, bursts=BURSTS)
    policies = build_policies()

    rows = []
    timelines = {}  # keep one representative (seed=1) timeline per policy for plotting
    for policy in policies:
        for seed in SEEDS:
            result = run_simulation(
                policy, rate_fn, total_seconds=TOTAL_SECONDS, seed=seed,
                service_time_kwargs=SERVICE_KW, cold_start_kwargs=COLDSTART_KW,
                bursts=BURSTS, onset_window=ONSET_WINDOW, peak_window=PEAK_WINDOW,
            )
            if seed == SEEDS[0]:
                timelines[policy.name] = result["timeline"]
            row = {k: v for k, v in result.items() if k != "timeline"}
            rows.append(row)
            print(f"{policy.name:14s} seed={seed}  overall={row['cold_start_rate']:.3f}  "
                  f"onset={row['onset_cold_start_rate']:.3f}  "
                  f"ramp_transition={row['ramp_transition_cold_start_rate']:.3f}  "
                  f"peak_sudden={row['peak_sudden_cold_start_rate']:.3f}  "
                  f"peak_ramping={row['peak_ramping_cold_start_rate']:.3f}  "
                  f"container_hours={row['container_hours']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "raw_runs.csv"), index=False)

    agg = df.groupby("policy").agg(
        cold_start_rate_mean=("cold_start_rate", "mean"),
        cold_start_rate_std=("cold_start_rate", "std"),
        burst_cold_start_rate_mean=("burst_cold_start_rate", "mean"),
        burst_cold_start_rate_std=("burst_cold_start_rate", "std"),
        onset_cold_start_rate_mean=("onset_cold_start_rate", "mean"),
        onset_cold_start_rate_std=("onset_cold_start_rate", "std"),
        peak_sudden_cold_start_rate_mean=("peak_sudden_cold_start_rate", "mean"),
        peak_sudden_cold_start_rate_std=("peak_sudden_cold_start_rate", "std"),
        peak_ramping_cold_start_rate_mean=("peak_ramping_cold_start_rate", "mean"),
        peak_ramping_cold_start_rate_std=("peak_ramping_cold_start_rate", "std"),
        ramp_transition_cold_start_rate_mean=("ramp_transition_cold_start_rate", "mean"),
        ramp_transition_cold_start_rate_std=("ramp_transition_cold_start_rate", "std"),
        ramp_transition_p99_ms_mean=("ramp_transition_p99_ms", "mean"),
        latency_p50_ms_mean=("latency_p50_ms", "mean"),
        latency_p95_ms_mean=("latency_p95_ms", "mean"),
        latency_p99_ms_mean=("latency_p99_ms", "mean"),
        onset_p95_ms_mean=("onset_p95_ms", "mean"),
        onset_p99_ms_mean=("onset_p99_ms", "mean"),
        peak_sudden_p99_ms_mean=("peak_sudden_p99_ms", "mean"),
        peak_ramping_p99_ms_mean=("peak_ramping_p99_ms", "mean"),
        container_hours_mean=("container_hours", "mean"),
        container_hours_std=("container_hours", "std"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "aggregated_results.csv"), index=False)

    with open(os.path.join(OUT_DIR, "timelines.json"), "w") as f:
        json.dump(timelines, f)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({
            "total_seconds": TOTAL_SECONDS, "base_rate": BASE_RATE, "bursts": BURSTS,
            "seeds": SEEDS, "service_time": SERVICE_KW, "cold_start": COLDSTART_KW,
        }, f, indent=2)

    return df, agg, timelines


if __name__ == "__main__":
    df, agg, timelines = run_all()
    print("\n=== Aggregated results ===")
    print(agg.to_string(index=False))
