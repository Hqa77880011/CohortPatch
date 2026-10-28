# CohortPatch

An implementation of **CohortPatch: Survivor-Aware Historical Partition Refinement for Multi-Round Secure Aggregation**. It includes two-sided history-safe rosters, public-forecast fragmentation cost, an exact capacity DP and a durable ideal aggregation adapter.

## Installation

Run the commands from the repository root in an activated environment.

~~~bash
python -m pip install -e ".[activity]"
~~~

Python 3.11+ is required. The activity extra supplies PyTorch for utility training; scheduling itself needs no GPU.

## Data

~~~bash
python -m cohortpatch prepare --count 4 --seed 17 --output data/demo.jsonl
~~~

Each JSONL case declares enrolled clients, historical k, separate cryptographic threshold, capacity, post-upload availability and independent public forecasts. IDs are zero-based enrolled IDs. Forecasts contain flattened sample/time availability sets; forecast_horizon converts their average to the mean sum over the horizon. Never use realized future test availability as the forecast.

configs/small.json charges two shared warm-start releases yielding (6,3,3) from one class. Warm rounds count toward usable opportunities, participation, training and release costs. initial_classes instead imports already retained history and does not establish that its setup was free.

Fit a first-order public predictor only on separate training traces:

~~~bash
python -m cohortpatch prepare --count 4 --seed 11 --output data/forecast_train.jsonl
python -m cohortpatch.forecast --train data/forecast_train.jsonl --data data/demo.jsonl --horizon 5 --samples 64 --seed 17 --output data/forecast_test.jsonl
~~~

For [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), unpack the original distribution:

~~~bash
python -m cohortpatch.data --har-root "data/UCI HAR Dataset" --output data/activity.npz --seed 20261028
~~~

The official 21/9 training/test subject assignment is retained. Training subjects are assigned 12 enrolled, 3 utility-validation, 3 auxiliary-fitting and 3 auxiliary-validation roles. Only enrolled/utility-validation and official test features enter the utility NPZ; the role JSON retains auxiliary identities. Normalization fits enrolled training rows only. Class count is defined by training labels. Availability is a controlled system condition, not a measured sensor dataset trace.

## Scheduling, utility training and inference

~~~bash
python -m cohortpatch run --data data/forecast_test.jsonl --config configs/small.json --output runs/main
python -m cohortpatch.activity --data data/activity.npz --records runs/main/cohortpatch/records.jsonl --case-id 17:0 --steps 1 --batch-size 32 --seed 17 --output runs/activity
python -m cohortpatch.predict --data data/activity.npz --checkpoint runs/activity/activity.pt --output runs/activity/predictions.json
~~~

Utility NPZ schema: x[N,D], zero-based y[N], client[N], and split[N] with 0/1/2 for training/validation/test. The adapter uses official HAR feature vectors and a 64-unit GELU MLP, SGD learning rate 0.03, equal client weights and stateless local training. Durable roster reservation/freeze precedes each ideal sum and model update. Choose a fresh training output directory. Checkpoints include architecture, train mean/std, seed and optimizer settings; evaluation/reload use eval mode and saved preprocessing.

## Evaluation and paper correspondence

| Paper module or metric | Code entry | Configuration or command |
| --- | --- | --- |
| Two-sided historical children and survivor options | core.legal_size; core.refine | k; availability |
| Debt, finite-horizon loss and capacity DP | core.choose; forecast.main | fairness/fragmentation/retry weights |
| Minimum history size, violations, usable inclusion, Jain fairness | core.score; core.aggregate | run; plot |
| Equal-weight utility accuracy and macro-F1 | activity.main; predict.main | utility commands above |
| Durable roster drift/recovery enforcement | protocol.RosterLedger | reserve/freeze/release; recovery test |
| Frozen reconstruction and static profile assignment | attacks.main | separately declared diagnostics below |

Included fraction uses original usable client-rounds, including paid warm rounds. Minimum size is the worst retained prefix; history_violations counts undersized classes at prefixes. history_only_identity_accuracy is the uniform-within-history-class diagnostic, not a bound under arbitrary background knowledge.

~~~bash
python -m cohortpatch.attacks --records runs/main/cohortpatch/records.jsonl --case-id 17:0 --updates data/frozen_aggregates.npz --reference data/frozen_profiles.npz --mode frozen --output runs/frozen_attack.json
~~~

updates[R,D] must contain actual equal-weight aggregate updates, and profiles[N,D] the declared frozen references. The assignment mode instead gives a static independent-profile diagnostic. Frozen linear reconstruction must not be interpreted as dynamic trajectory reconstruction.

## Baselines and ablations

~~~bash
python -m cohortpatch suite --data data/forecast_test.jsonl --config configs/small.json --output runs/comparison
~~~

Methods: cohortpatch, fixed_groups, greedy_safe, no_history, no_complement, no_fragmentation, no_fairness. All share public availability/forecasts. The fairness ablation removes debt from ordering and score. Unsafe history/complement diagnostics are not authorized by the protected utility backend.

The local sum adapter is an ideal backend, not Flower SecAgg+ cryptography. WISDM ingestion, the manuscript CNN, dynamic trajectory/attribute attacks and their support gate, full Multi-RoundSecAgg integration, and measured protocol/restart bytes are not implemented. Forecasts use a first-order model rather than correlated bootstrap. CLI CPU/RSS concern the local process, not network protocol overhead.

## Implementation provenance

The inherited core, CLI and statistics modules are finite-model implementation code. The original server experiment backend and raw records were not available in this checkout. Version 0.2 extends and checks this implementation. Generated fixtures and the checks below are functional validation, not evidence for the manuscript's reported performance. No manuscript result table is presented as an implementation benchmark.

## Outputs and verification

Each method writes records.jsonl, metrics.json and run_config.json. Records retain failed cases, seeds and clusters; the manifest retains the command and dependency versions. Evaluator records may contain private truth or local state; they are not a privacy-filtered service API. Runtime data, weights, SQLite databases and figures are excluded from Git.

~~~bash
python -m cohortpatch.statistics --records runs/main/cohortpatch/records.jsonl --resamples 2000 --seed 17 --output runs/main/intervals.json
python -m cohortpatch plot --inputs runs/main/cohortpatch --metrics included_fraction jain_fairness --output runs/figures
python -m unittest discover -s tests -v
~~~

Use --reference OTHER/records.jsonl for paired cluster comparisons with identical case IDs. Clusters use the supplied cluster field, otherwise the case ID. Generic cluster intervals do not replace a manuscript-specific multi-seed analysis. Plots and summary.csv use the same aggregate values as metrics.json. Choose fresh output paths on reruns.

Validated on Windows with Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 and psutil 7.1.0.

The existing MIT code license is retained; datasets and models retain their own licenses.

Additional validated environment: PyTorch 2.7.1+cu118 on CPU. Checks include four-case suites, legal split sizes, warm histories, crash/roster replay, and two synthetic federated rounds with 14 participating local SGD updates. Loss/gradients were finite, every model parameter changed, and reload predictions agreed within tolerance. Real HAR training and cryptographic experiments were not run.
