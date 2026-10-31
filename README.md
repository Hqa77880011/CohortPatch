# CohortPatch

An independent implementation of **CohortPatch: Survivor-Aware Historical Partition Refinement for Multi-Round Secure Aggregation**. It includes two-sided history-safe rosters, public forecasts, exact capacity DP, real Flower SecAgg+, sensor CNN training and the appendix attack suite. It is not an official author artifact.

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
| Minimum history size, violations, usable inclusion, p10 inclusion, Jain fairness | core.score; core.aggregate | run; plot |
| Equal-weight utility accuracy and macro-F1 | activity.main; predict.main | utility commands above |
| Durable roster drift/recovery enforcement | protocol.RosterLedger | reserve/freeze/release; recovery test |
| Frozen reconstruction and static profile assignment | attacks.main | separately declared diagnostics below |

Included fraction uses original usable client-rounds, including paid warm rounds. p10_inclusion_fraction is the 10th percentile of each client's included/usable opportunities among clients with at least one opportunity; p10_inclusion_count instead uses raw inclusion counts across every enrolled client. Invalid or duplicate availability/forecast IDs and nonpositive policy thresholds are rejected before scheduling. Minimum size is the worst retained prefix; history_violations counts undersized classes at prefixes. history_only_identity_accuracy is the uniform-within-history-class diagnostic, not a bound under arbitrary background knowledge.

~~~bash
python -m cohortpatch.attacks --records runs/main/cohortpatch/records.jsonl --case-id 17:0 --updates data/frozen_aggregates.npz --reference data/frozen_profiles.npz --mode frozen --output runs/frozen_attack.json
~~~

updates[R,D] must contain actual equal-weight aggregate updates, and profiles[N,D] the declared frozen references. The assignment mode instead gives a static independent-profile diagnostic. Frozen linear reconstruction must not be interpreted as dynamic trajectory reconstruction.

## Baselines and ablations

~~~bash
python -m cohortpatch suite --data data/forecast_test.jsonl --config configs/small.json --output runs/comparison
~~~

The original controls are cohortpatch, fixed_groups, greedy_safe, no_history, no_complement, no_fragmentation and no_fairness. All share public availability/forecasts. The fairness ablation removes debt from ordering and score. Ordinary aggregation retains the common debt/retry objective without historical constraints. NDA/no-complement training records use historical k=1 while retaining the exact-roster gate and common crypto threshold; they never claim target-k protection. Protected methods enforce the declared historical k.

The original `cohortpatch.data` feature-vector/MLP path above remains a labeled feature control. The primary inertial-window/CNN path and protocol/attack entries are below. CLI CPU/RSS concern the local process. Native SecAgg+ byte counters measure serialized application protobuf messages, not TCP/TLS packet captures.

## Primary sensor, protocol and attack entries

```bash
python -m cohortpatch.sensors --source har --root "data/UCI HAR Dataset" --output data/har-cnn.npz
python -m cohortpatch.sensors --source wisdm --root data/wisdm --output data/wisdm-cnn.npz
python -m cohortpatch.availability --sensor data/har-cnn.npz --regime bursty --offline-mean 5 --attempts 200 --output data/availability.jsonl
python -m cohortpatch run --data data/availability.jsonl --output runs/cnn-rosters
```

`sensors` reads actual HAR total-acceleration XYZ windows and WISDM smartphone accelerometers (128 samples, stride 64, 20 Hz). Corrupt rows, time resets, activity boundaries and acquisition gaps split WISDM segments. HAR retains official utility-test subjects. Defaults reserve four auxiliary-fit and four auxiliary-validation users: HAR has 10 enrolled/3 utility-validation/9 official-test subjects; WISDM derives its 60/20/20 roles before reserving auxiliary users (normally 22 enrolled). Exact retained counts and timestamps are saved in the role manifest; source totals never replace enrolled N.

Availability regimes are independent (`--probability .5/.7/.9`), bursty (stationary .7, offline means 2/5/10), heterogeneous (.4--.95), correlated outages and explicitly activity-correlated stress. Joint conditional block-bootstrap forecasts use independently generated training runs; the legacy first-order predictor remains a separate condition. `--warm-rosters` imports paid retained releases. Auxiliary runs use `--role 3` or `--role 4` and their actual populations.

```bash
python -m venv .venv-secaggplus
.venv-secaggplus/bin/python -m pip install flwr==1.23.0 numpy==2.1.3
python -m cohortpatch.activity --data data/har-cnn.npz --records runs/cnn-rosters/cohortpatch/records.jsonl --case-id 17:bursty:0 --architecture cnn --backend secaggplus --secagg-python .venv-secaggplus/bin/python --output runs/cnn
```

On Windows the isolated executable is `.venv-secaggplus/Scripts/python.exe`. Keep Flower in its own environment; the utility process can use the existing PyTorch environment. The CNN has 32/64 channels, kernels 5/3, ReLU, temporal pooling, global average pooling, Adam, equal client weights and stateless local optimizers. Training-role-only normalization and L2 clipping precede secure encoding. The backend executes unchanged Flower 1.23.0 client mods and all four protocol stages, including fresh key exchange, shares, masks and unmasking. It checks the actual masked roster immediately before unmasking. Drift, dropout or interrupted reservations cannot authorize a fresh unnoticed release; native released sums are committed durably and retries return them. The live training backend aggregates clipped **updates**, then adds them to the public model.

`server_aggregates.npz` contains public pre-round models, averages, rosters, availability, layer coordinates and run IDs. Separate restricted `scorer_updates.npz` contains actual individual updates and a mask: inactive client-rounds have no invented ground truth. It must not be given to an attacker except for the explicitly declared colluders or post-hoc scoring.

```bash
python -m cohortpatch.trajectory --data runs/target/server_aggregates.npz --truth runs/target/scorer_updates.npz --validation runs/auxval/server_aggregates.npz --validation-truth runs/auxval/scorer_updates.npz --mode dynamic --output runs/dynamic-attack
python -m cohortpatch.profiles split --sensor data/wisdm-cnn.npz --output data/profile-blocks
python -m cohortpatch.profiles build --sensor data/profile-blocks/prior.npz --trace runs/target/server_aggregates.npz --checkpoint runs/target/activity.pt --manifest data/profile-blocks/blocks.json --output runs/named-priors
python -m cohortpatch.attack_data --manifest data/attack-runs.json --sensor data/wisdm-cnn.npz --mode full --output data/attribute
python -m cohortpatch.attribute --data data/attribute/attribute.npz --output runs/attribute
python -m cohortpatch.disaggregation --data data/disaggregation.npz --clients 10 --output runs/disaggregation
```

`trajectory` implements frozen ridge/pseudoinverse and dynamic temporal projected gradient, five restarts, the manuscript alpha/iteration validation grids, declared-colluder subtraction, actual observed-coordinate reconstruction error and one-to-one cosine assignment. Named references recompute stateless clipped local updates at every public model on chronologically disjoint held-out recording blocks. Train the victim on `profile-blocks/victim.npz`; its checkpoint must establish the claimed block boundary and full-window gap. HAR without verifiable acquisition metadata produces `unsupported_profile_prior`.

`attack-runs.json` is a list of `{trace,run_id,role}`, with roles 0/1/2 for independent auxiliary fitting/validation/target runs and the corresponding actual sensor roles 3/4/0. The attribute entry builds only public features, freezes the auxiliary-fit walking-fraction median, checks two independent users per class per role and acquisition-block absence confounding, then trains the two-layer 64-unit GRU with training-only scaling and validation stopping. Unsupported cells have no AUC. Availability-only and metadata variants are explicit options. Validation adequacy is reported rather than assumed.

`disaggregation` independently implements [Lam et al. Eqs. 5--7](https://proceedings.mlr.press/v139/lam21b.html): hard-threshold SVD, exact binary quadratic linearization, participation summaries and an excluded-solution uniqueness check. The unlicensed author source is not copied or imported. `sums`, `summary_matrix`, `counts` are the original side-information condition; `known_roster` selects the documented known-matrix adaptation. FedAvg drift remains a noisy fixed-vector approximation; the dynamic temporal attack is evaluated separately.

Additional scheduler controls are `mr_fixed_k`, `mr_plus`, `no_refinement`, `class_count`, `no_retry` and `future_oracle`, alongside the existing warm-start, no-history and lambda-zero controls. Original fixed-K MR requires divisible N/T and K/T and is explicit-only; MR+ enlarges the remainder and selects legal variable-capacity unions with the stated heterogeneous fairness fallback. The exact future oracle is bounded to N<=12 and horizon<=6. TEE auditing is a protection-boundary reference, not a fabricated numerical baseline.

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

CPU functional checks on 2026-11-01 include eleven source tests, actual CNN gradients/parameter updates, dynamic reconstruction, recording-block gaps, public-feature isolation, original/MR+ roster arithmetic, and real Flower four-stage execution. A two-round/four-participant CNN run made eight actual clipped local updates through SecAgg+, saved/reloaded its model, and recorded real protobuf bytes. Independent small GRU and disaggregation checks exercise fitting/optimization without full training. Earlier MLP, suite, forecast, plotting and statistics checks remain functional fixtures. Full HAR/WISDM acquisition, 200-attempt/five-seed training, network captures and manuscript result reproduction remain validation limits; no generated fixture is reported as a paper result.
