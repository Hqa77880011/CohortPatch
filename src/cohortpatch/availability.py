"""Controlled independent, bursty, heterogeneous and correlated availability."""

import argparse
import json
from pathlib import Path

import numpy as np


def process(clients, attempts, regime, rng, *, probability=.7, offline_mean=5,
            correlation=.25, activity=None):
    if clients < 1 or attempts < 1 or not 0 < probability < 1 or offline_mean < 1:
        raise ValueError("Positive sizes, probability in (0,1), mean offline duration >=1 required")
    p = np.linspace(.4,.95,clients) if regime == "heterogeneous" else np.full(clients,probability)
    if regime in ("independent","heterogeneous"):
        return rng.random((attempts,clients)) < p
    if regime == "activity_correlated":
        if activity is None or np.shape(activity) != (clients,):
            raise ValueError("Activity-correlated stress needs a fixed target composition vector")
        p = np.clip(.4+.55*np.asarray(activity),.05,.95)
        return rng.random((attempts,clients)) < p
    # Stationary two-state chains: P(offline->online)=1/L,
    # P(online->offline)=(1-p)/(pL). No source sensor trace is called an outage log.
    on = 1/offline_mean
    off = (1-probability)/(probability*offline_mean)
    if off > 1:
        raise ValueError("Requested stationary probability and offline duration are incompatible")
    state = rng.random(clients) < probability
    result = []
    for _ in range(attempts):
        state = rng.random(clients) < np.where(state,1-off,on)
        if regime == "correlated":
            shared = rng.random() < probability
            state = np.where(rng.random(clients)<correlation,shared,state)
        result.append(state.copy())
    return np.asarray(result)


def bootstrap_forecasts(training, current, horizon, samples, rng):
    """Resample joint training blocks conditioned on nearest public current state."""
    candidates = [(row[i],row[i+1:i+1+horizon]) for row in training for i in range(len(row)-horizon)]
    if not candidates:
        raise ValueError("Independent training runs must be longer than the forecast horizon")
    distance = np.array([np.count_nonzero(state != current) for state,_ in candidates])
    nearest = np.flatnonzero(distance == distance.min())
    return [np.flatnonzero(future).tolist() for index in rng.choice(nearest,samples)
            for future in candidates[index][1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", required=True, help="Role-separated sensor NPZ; enrolled count is derived")
    parser.add_argument("--role", type=int, choices=[0,3,4], default=0)
    parser.add_argument("--regime", choices=["independent","bursty","heterogeneous","correlated","activity_correlated"], default="independent")
    parser.add_argument("--attempts", type=int, default=200); parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--train-runs", type=int, default=5); parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--forecast-samples", type=int, default=64)
    parser.add_argument("--probability", type=float, choices=[.5,.7,.9], default=.7)
    parser.add_argument("--offline-mean", type=int, choices=[2,5,10], default=5)
    parser.add_argument("--capacity", type=int, default=10); parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=3); parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warm-rosters", help="Fixed JSON warm-start rosters; never silently discarded")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sensor = dict(np.load(args.sensor,allow_pickle=False))
    ids = np.unique(sensor["client"][sensor["split"]==args.role]); n = len(ids)
    if n < args.k or args.capacity < args.threshold:
        raise ValueError("Actual enrolled-role population cannot support this public policy")
    activity = np.array([np.mean(sensor["y"][sensor["client"]==i]==0) for i in ids])
    rng = np.random.default_rng(args.seed)
    make = lambda: process(n,args.attempts,args.regime,rng,probability=args.probability,offline_mean=args.offline_mean,activity=activity)
    training = [make() for _ in range(args.train_runs)]
    warm = json.loads(Path(args.warm_rosters).read_text(encoding="utf-8")) if args.warm_rosters else []
    records = []
    for run in range(args.runs):
        states = make()
        forecasts = [bootstrap_forecasts(training,current,args.horizon,args.forecast_samples,rng) for current in states]
        records.append(dict(id=f"{args.seed}:{args.regime}:{run}",family=f"{args.seed}:{run}",clients=n,k=args.k,
                            threshold=args.threshold,capacity=min(n,args.capacity),seed=args.seed+run,
                            availability=[np.flatnonzero(row).tolist() for row in states],forecasts=forecasts,
                            forecast_horizon=args.horizon,warm_start_rosters=warm,
                            heterogeneous_availability=args.regime=="heterogeneous",sensor_client_ids=ids.tolist(),
                            source="controlled process, not measured mobile availability"))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(row)+"\n" for row in records),encoding="utf-8")
    output.with_suffix(".forecast.json").write_text(json.dumps(dict(training_runs=[row.astype(int).tolist() for row in training],
        forecast="joint conditioned block bootstrap",seed=args.seed,sensor_client_ids=ids.tolist(),
        disjoint_random_stream_segments=True,regime=args.regime,probability=args.probability,offline_mean=args.offline_mean)),encoding="utf-8")


if __name__ == "__main__":
    main()
