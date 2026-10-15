# CohortPatch

Implementation of **Survivor-aware historical partition refinement for multi-round secure aggregation**.

Maintains complete participation-history classes. A multiple-choice dynamic program selects legal child/complement splits under capacity, release threshold, opportunity debt and forecast fragmentation cost.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/CohortPatch.git
cd CohortPatch
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

A scenario declares enrolled clients, k, cryptographic threshold, capacity, actual post-upload availability and public forecast samples for every round.

```bash
python -m cohortpatch prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m cohortpatch run --data data/workload.jsonl --output runs/main
python -m cohortpatch suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. cohortpatch uses fragmentation and fairness; fixed_groups freezes initial groups; greedy_safe removes future/fairness scores; no_history, no_complement, no_fragmentation and no_fairness are mechanism ablations.

## Evaluation and visualization

Minimum history-class size, violations, included fraction, final class count, fairness, successful rounds and exact history-only identity attribution accuracy.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m cohortpatch plot --inputs runs/comparison/cohortpatch --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

This core authorizes the exact post-upload roster before plaintext release. Applications must freeze that roster and retain uncertain history. Cryptographic masking is delegated to the application's secure aggregation implementation. History-only attribution is a posterior endpoint; activity-model F1 requires actual local training updates.


## Uncertainty and paired comparisons

```bash
python -m cohortpatch.statistics --records runs/main/cohortpatch/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Train the activity utility model

Export HAR/WISDM features as NPZ arrays x [samples,features], zero-based y, enrolled client IDs and split (0=train, 2=test). Use the exact roster selected for one scheduling episode:

```bash
python -m pip install -e '.[activity]'
python -m cohortpatch.activity --data data/activity.npz --records runs/main/cohortpatch/records.jsonl --case-id 17:0 --output runs/activity
```

Local learners start from the same model each round and the authorized updates are equally averaged. Output includes held-out utility macro-F1, accuracy and server-visible aggregate updates. Protocol masking remains the secure aggregation adapter's responsibility.
