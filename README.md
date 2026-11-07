# Serverless Cold-Start Mitigation Simulator

A discrete-event(ish) simulator of a FaaS (serverless function) platform, built to
compare three cold-start mitigation strategies under bursty traffic:

1. **Reactive (baseline)** — no proactive warming. A new container is only
   created when a request arrives and no warm container is available.
2. **Keep-warm pool** — a fixed number of containers (N) are always kept
   warm, regardless of current demand. Swept over N = 2, 5, 10, 15, 20, 30, 45.
3. **Predictive pre-warming** — forecasts near-future demand from a
   trend-adjusted moving average of recent arrivals, and proactively
   provisions a floor of containers sized to the forecast.

No cloud account, Kubernetes cluster, or GPU is needed — everything runs as a
pure Python simulation.

## How it works

Time is modeled continuously (sub-second precision) within 1-second
accounting ticks. Each second:
- Requests arrive following a time-varying Poisson process (a steady low
  base rate, plus scripted traffic bursts).
- Each request needs a container. If a warm one is free, it's used
  (fast). If not, a new one is created, incurring a cold-start penalty.
- Containers that finish serving go idle, and are torn down after 300
  seconds of inactivity (unless a policy is actively protecting them as
  part of its "floor").

Five bursts are injected across a 4-hour simulated window: three **sudden**
step-function spikes (no warning, the "viral link" scenario) and two
**ramping** bursts that build up over 30-45 seconds (the "marketing push"
scenario). This mix matters — see Results.

## Running it

```bash
pip install -r requirements.txt
python src/experiments.py   # runs all 9 policies x 5 random seeds, saves CSVs
python src/plots.py         # builds the 4 figures from the results
```

Outputs land in `outputs/`:
- `raw_runs.csv` — every individual run (policy x seed)
- `aggregated_results.csv` — mean/std across seeds per policy
- `config.json` — exact workload parameters used
- `conditioned_rates.png`, `cost_frontier.png`, `timeline_ramp.png`, `timeline_sudden.png`

## Results

| Policy | Onset cold-start rate | Ramp-transition rate | Sudden-peak rate | Container-hours |
|---|---|---|---|---|
| Reactive | 15.5% | 4.6% | 10.6% | 50.4 |
| Keep-warm (N=2) | 14.9% | 4.7% | 10.3% | 50.0 |
| Keep-warm (N=10) | 12.0% | 3.3% | 8.6% | 64.3 |
| Keep-warm (N=20) | 8.8% | 0.8% | 6.2% | 95.5 |
| Keep-warm (N=45) | 1.2% | 0.0% | 0.9% | 182.2 |
| **Predictive** | 15.4% | **3.3%** | 10.6% | 53.9 |

Three findings fall out of this:

1. **Sudden, unannounced bursts hurt everyone equally** (except very large,
   expensive static pools). There's no getting around this: if there's no
   leading signal in the traffic, no amount of cleverness in the policy
   can anticipate it — only brute-force over-provisioning helps, and that
   costs 2-4x more container-hours.
2. **Predictive pre-warming earns its keep specifically on ramping
   traffic**: it cuts the ramp-transition cold-start rate from 4.6% to
   3.3% (a ~29% relative reduction) — matching what a keep-warm(N=10) pool
   achieves, while costing about as much as keep-warm(N=2-5) (53.9 vs
   64.3 container-hours). It provides essentially no benefit for sudden
   bursts or steady-state traffic, which is exactly what you'd expect
   from a method that works by extrapolating a trend that has to exist
   in the first place.
3. **Static keep-warm pools trade cost for reliability cleanly and
   predictably** — the cost/cold-start-rate frontier in `cost_frontier.png`
   is smooth and monotonic, which is a useful sanity check that the
   simulator behaves the way a real system should.

## Two bugs worth knowing about (and why they mattered)

Building this surfaced two real modeling bugs, both worth mentioning
because they're the kind of thing that silently invalidates a simulation
without throwing an error:

- **Tick granularity was hiding all the signal.** The first version used
  1-second ticks and only let a container serve one request per tick,
  even though real service times are far shorter (~150ms-800ms). That
  artificially capped each container's throughput and made cold starts
  look 10x more common than they should be. Fixed by processing arrivals
  at exact sub-second timestamps within each tick.
- **The predictive policy's window was longer than the bursts it was
  supposed to predict.** Its 20-second forecasting window meant it
  produced no forecast at all until 20 seconds of data had accumulated —
  by which point a 30-second ramp was already 60%+ over. Fixed by
  forecasting from whatever partial window is available instead of
  waiting for a full one.

Both are documented in the code comments and are worth reading if you're
adapting this for your own workload — they're easy mistakes to make and
easy to miss, since the simulation runs fine and produces plausible-looking
(if quietly wrong) numbers either way.

## Honest limitations

- This is a simplified model. Real FaaS platforms have per-tenant
  isolation, multiple concurrency-per-container modes, network cold
  starts (VPC attachment, provisioned concurrency warm-up), and
  language-runtime-specific cold-start distributions (interpreted
  languages cold-start much faster than JVM-based ones).
- The predictive policy uses a simple trend-extrapolation forecaster, not
  a trained model. It's intentionally simple so its behavior is fully
  auditable, at the cost of missing more complex seasonal patterns a
  learned model might catch.
- Traffic here is scripted (known bursts at known times) rather than
  drawn from a real production trace; the qualitative findings (sudden
  vs. ramping bursts behave very differently) should transfer, but exact
  percentages are specific to this workload's parameters.

## License

© 2026 Janat Yasser. All rights reserved. See [LICENSE](LICENSE).
