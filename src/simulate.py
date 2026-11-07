# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
simulate.py
-----------
A discrete-time (1-second tick) simulator of a serverless (FaaS) platform,
used to compare cold-start mitigation strategies under bursty traffic.

Model
-----
- Requests arrive each second according to a time-varying Poisson rate
  (a steady base rate plus a small number of traffic bursts).
- Each request needs a "container" to run in. A container is either:
    idle   - warm and waiting, can serve the next request instantly
    busy   - currently serving a request
  If no idle container is available, a new one is created, which incurs
  a cold-start penalty (an extra delay) before it can serve the request.
- After finishing a request, a container goes idle. If it stays idle for
  longer than `idle_timeout` seconds without being reused, it is torn
  down (removed), matching how real FaaS platforms recycle containers.
- A "policy" decides how many containers should be proactively kept
  warm (a "floor") at any given time. This is the only thing that
  differs between the three strategies compared in this project.

This is a simplified model of a real FaaS platform (see README for the
specific simplifications and their justification), designed to isolate
the effect of the provisioning *policy* while holding the request
workload and hardware assumptions fixed across all three strategies.
"""
import numpy as np
from collections import deque


# ---------------------------------------------------------------------
# Workload model
# ---------------------------------------------------------------------
def make_arrival_rate_fn(base_rate=3.0, bursts=None):
    """bursts: list of dicts with keys start, ramp, plateau, multiplier, kind.
    Rate rises linearly from base_rate to base_rate*multiplier over `ramp`
    seconds (0 = instantaneous/sudden), then holds at the peak for
    `plateau` seconds. Returns a function t -> expected arrivals/sec."""
    bursts = bursts or []

    def rate(t):
        r = base_rate
        for b in bursts:
            start, ramp, plateau, mult = b["start"], b["ramp"], b["plateau"], b["multiplier"]
            if start <= t < start + ramp:
                frac = (t - start) / max(ramp, 1e-9)
                r = base_rate + (base_rate * mult - base_rate) * frac
            elif start + ramp <= t < start + ramp + plateau:
                r = base_rate * mult
        return r
    return rate


def burst_tag(t, bursts, onset_window=10, peak_window=10):
    """Returns (in_burst, in_onset, peak_kind, in_ramp_transition) for time t."""
    in_burst = False
    in_onset = False
    peak_kind = None
    in_ramp_transition = False
    for b in bursts:
        start, ramp, plateau, kind = b["start"], b["ramp"], b["plateau"], b["kind"]
        total = ramp + plateau
        if start <= t < start + total:
            in_burst = True
        if start <= t < start + onset_window:
            in_onset = True
        if ramp > 0 and start <= t < start + ramp:
            in_ramp_transition = True
        peak_start = start + ramp
        if peak_start <= t < peak_start + min(plateau, peak_window):
            peak_kind = kind
    return in_burst, in_onset, peak_kind, in_ramp_transition


def sample_service_time(rng, mean=0.15, sd=0.05, minimum=0.02):
    return float(max(minimum, rng.normal(mean, sd)))


def sample_cold_start(rng, mean=1.0, sd=0.30, minimum=0.20):
    return float(max(minimum, rng.normal(mean, sd)))


# ---------------------------------------------------------------------
# Policies: each defines how many containers should be kept warm ("floor")
# at time t, given a rolling window of recent per-second arrival counts.
# ---------------------------------------------------------------------
class ReactivePolicy:
    """Baseline: no proactive warming at all. Containers are created only
    in direct response to a request that has nowhere warm to land."""
    name = "reactive"
    idle_timeout = 300  # seconds a container may sit idle before teardown

    def min_floor(self, t, recent_arrivals):
        return 0


class KeepWarmPolicy:
    """A fixed-size pool of N containers is kept warm at all times (via
    floor protection in the simulator's teardown step); any *additional*
    containers spun up reactively during a burst that exceeds the pool
    size are still torn down after the normal idle timeout, exactly as
    in the baseline policy. Only the first N are permanently protected."""
    idle_timeout = 300  # same recycling behavior as the baseline for overflow containers

    def __init__(self, n):
        self.n = n
        self.name = f"keep_warm_{n}"

    def min_floor(self, t, recent_arrivals):
        return self.n


class PredictivePolicy:
    """Forecasts near-future demand from a trend-adjusted moving average
    of the last `window` seconds of arrivals, splitting the window into
    an older and newer half to estimate a simple linear trend, then
    provisions a floor sized to the forecast rate (with a safety margin).
    This is deliberately simple (a coarse analogue of Holt's linear
    trend method) rather than a learned model, to keep the mechanism
    auditable and cheap to run."""
    idle_timeout = 300

    def __init__(self, window=60, lead_time=15, safety_factor=1.3,
                 mean_service_time=0.15, min_floor_size=1):
        self.window = window
        self.lead_time = lead_time
        self.safety_factor = safety_factor
        self.mean_service_time = mean_service_time
        self.min_floor_size = min_floor_size
        self.name = "predictive"

    def min_floor(self, t, recent_arrivals):
        if len(recent_arrivals) < 4:
            return self.min_floor_size
        k = min(self.window, len(recent_arrivals))
        window_vals = list(recent_arrivals)[-k:]
        half = max(2, k // 2)
        older, newer = window_vals[:half], window_vals[half:]
        if not newer:
            newer = older
        older_mean, newer_mean = np.mean(older), np.mean(newer)
        trend_per_tick = (newer_mean - older_mean) / max(half, 1)
        forecast_rate = max(0.0, newer_mean + trend_per_tick * self.lead_time)
        target = forecast_rate * self.mean_service_time * self.safety_factor
        return max(self.min_floor_size, int(np.ceil(target)))


# ---------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------
def run_simulation(policy, rate_fn, total_seconds=14400, seed=0,
                    service_time_kwargs=None, cold_start_kwargs=None,
                    window=60, bursts=None, onset_window=10, peak_window=10):
    rng = np.random.default_rng(seed)
    service_time_kwargs = service_time_kwargs or {}
    cold_start_kwargs = cold_start_kwargs or {}
    bursts = bursts or []

    containers = []  # list of dicts: state, free_at (float sim-time), idle_since (float)
    recent_arrivals = deque(maxlen=window)

    # Each entry: (latency_ms, is_cold, in_burst, in_onset, peak_kind)
    records = []
    container_seconds = 0.0
    timeline = []  # (t, arrivals, n_containers, cum_cold_starts)
    cum_cold = 0

    initial_floor = policy.min_floor(0, recent_arrivals)
    for _ in range(initial_floor):
        containers.append({"state": "idle", "idle_since": 0.0, "free_at": 0.0})

    for t in range(total_seconds):
        lam = rate_fn(t)
        n = int(rng.poisson(lam))
        recent_arrivals.append(n)
        tb, to, pk, tr = burst_tag(t, bursts, onset_window, peak_window)

        # Sub-second arrival timestamps within [t, t+1), processed in order
        # so a container freed earlier in the second can be reused later
        # in the same second -- this is the key fix versus one-slot-per-
        # second-per-container, which was artificially capping each
        # container's throughput far below its real (sub-second) capacity.
        offsets = np.sort(rng.uniform(0.0, 1.0, size=n)) if n > 0 else []
        for off in offsets:
            ts = t + float(off)
            for c in containers:
                if c["state"] == "busy" and c["free_at"] <= ts:
                    c["state"] = "idle"
                    c["idle_since"] = c["free_at"]

            idle_containers = [c for c in containers if c["state"] == "idle"]
            if idle_containers:
                c = idle_containers[-1]
                svc = sample_service_time(rng, **service_time_kwargs)
                c["state"] = "busy"
                c["free_at"] = ts + svc
                records.append((svc * 1000.0, False, tb, to, pk, tr))
            else:
                cold = sample_cold_start(rng, **cold_start_kwargs)
                svc = sample_service_time(rng, **service_time_kwargs)
                containers.append({"state": "busy", "free_at": ts + cold + svc, "idle_since": None})
                records.append(((cold + svc) * 1000.0, True, tb, to, pk, tr))
                cum_cold += 1

        # End-of-second bookkeeping: refresh any containers that finished
        # during this second but had no later arrival to refresh them,
        # then apply the policy's floor (proactive top-up) and teardown.
        boundary = t + 1.0
        for c in containers:
            if c["state"] == "busy" and c["free_at"] <= boundary:
                c["state"] = "idle"
                c["idle_since"] = c["free_at"]

        target_floor = policy.min_floor(t, recent_arrivals)
        if len(containers) < target_floor:
            for _ in range(target_floor - len(containers)):
                containers.append({"state": "idle", "idle_since": boundary, "free_at": boundary})

        idle_now = [c for c in containers if c["state"] == "idle"]
        busy_now = len(containers) - len(idle_now)
        keep_idle_count = max(0, target_floor - busy_now)
        idle_now.sort(key=lambda c: c["idle_since"], reverse=True)
        protected_ids = set(id(c) for c in idle_now[:keep_idle_count])

        survivors = []
        for c in containers:
            if c["state"] == "busy":
                survivors.append(c)
            elif id(c) in protected_ids:
                survivors.append(c)
            elif (boundary - c["idle_since"]) < policy.idle_timeout:
                survivors.append(c)
        containers = survivors

        container_seconds += len(containers)
        timeline.append((t, n, len(containers), cum_cold))

    def summarize(subset):
        if not subset:
            return dict(n=0, cold_start_rate=0.0, p50=0.0, p95=0.0, p99=0.0, mean=0.0)
        lat = np.array([r[0] for r in subset])
        cold = np.array([r[1] for r in subset])
        return dict(
            n=len(subset),
            cold_start_rate=float(cold.mean()),
            p50=float(np.percentile(lat, 50)),
            p95=float(np.percentile(lat, 95)),
            p99=float(np.percentile(lat, 99)),
            mean=float(lat.mean()),
        )

    overall = summarize(records)
    burst_only = summarize([r for r in records if r[2]])
    onset_only = summarize([r for r in records if r[3]])
    peak_sudden = summarize([r for r in records if r[4] == "sudden"])
    peak_ramping = summarize([r for r in records if r[4] == "ramping"])
    ramp_transition = summarize([r for r in records if r[5]])
    total_requests = len(records)
    cold_starts = int(sum(1 for r in records if r[1]))

    return {
        "policy": policy.name,
        "seed": seed,
        "total_requests": total_requests,
        "cold_starts": cold_starts,
        "warm_hits": total_requests - cold_starts,
        "cold_start_rate": overall["cold_start_rate"],
        "latency_p50_ms": overall["p50"],
        "latency_p95_ms": overall["p95"],
        "latency_p99_ms": overall["p99"],
        "latency_mean_ms": overall["mean"],
        "burst_cold_start_rate": burst_only["cold_start_rate"],
        "burst_p95_ms": burst_only["p95"],
        "burst_p99_ms": burst_only["p99"],
        "burst_n": burst_only["n"],
        "onset_cold_start_rate": onset_only["cold_start_rate"],
        "onset_p95_ms": onset_only["p95"],
        "onset_p99_ms": onset_only["p99"],
        "onset_n": onset_only["n"],
        "peak_sudden_cold_start_rate": peak_sudden["cold_start_rate"],
        "peak_sudden_p99_ms": peak_sudden["p99"],
        "peak_sudden_n": peak_sudden["n"],
        "peak_ramping_cold_start_rate": peak_ramping["cold_start_rate"],
        "peak_ramping_p99_ms": peak_ramping["p99"],
        "peak_ramping_n": peak_ramping["n"],
        "ramp_transition_cold_start_rate": ramp_transition["cold_start_rate"],
        "ramp_transition_p95_ms": ramp_transition["p95"],
        "ramp_transition_p99_ms": ramp_transition["p99"],
        "ramp_transition_n": ramp_transition["n"],
        "container_seconds": container_seconds,
        "container_hours": container_seconds / 3600.0,
        "timeline": timeline,
    }
