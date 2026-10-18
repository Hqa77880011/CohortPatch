# CohortPatch

Implementation of **Survivor-aware historical partition refinement for multi-round secure aggregation**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

A scenario declares enrolled clients, k, cryptographic threshold, capacity, actual post-upload availability and public forecast samples for every round.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m cohortpatch prepare --count 200 --seed 17 --output data/generated.jsonl
```

## Execution and evaluation

```bash
python -m cohortpatch run --data data/generated.jsonl --output runs/main
```

## Comparisons and ablations

```bash
python -m cohortpatch suite --data data/generated.jsonl --output runs/comparison
```

Use `--methods` to select: `cohortpatch`, `fixed_groups`, `greedy_safe`, `no_history`, `no_complement`, `no_fragmentation`, `no_fairness`.

## Outputs

`runs/main/cohortpatch/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m cohortpatch plot --inputs runs/comparison/cohortpatch --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m cohortpatch.statistics --records runs/main/cohortpatch/records.jsonl --output results/intervals.json
```
