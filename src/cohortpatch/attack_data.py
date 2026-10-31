"""Build public-only GRU features from independent actual utility-training runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .attribute import composition_labels, public_features, support_gate


def build(manifest, sensor, *, mode="full", rounds=None):
    labels, threshold, fractions = composition_labels(sensor)
    sequences, y, roles, identities, runs = [], [], [], [], []
    for row in manifest:
        trace = dict(np.load(row["trace"], allow_pickle=False))
        ids = trace["target_client_ids"]
        length = len(trace["roster"]) if rounds is None else rounds
        if len(trace["roster"]) < length:
            raise ValueError("Each complete attack run must contain the fixed public prefix")
        for target, identity in enumerate(ids):
            identity = int(identity)
            expected_sensor_role = {0: 3, 1: 4, 2: 0}[row["role"]]
            if not np.any((sensor["client"] == identity) & (sensor["split"] == expected_sensor_role)):
                raise ValueError("Trace identity does not belong to its independent auxiliary/target role")
            sequences.append(public_features(trace, target, mode)[:length])
            y.append(labels[identity]); roles.append(row["role"]); identities.append(identity); runs.append(row["run_id"])
    if len({len(s) for s in sequences}) != 1:
        raise ValueError("Fix --rounds when actual traces have different prefix lengths")
    arrays = dict(x=np.asarray(sequences, np.float32), y=np.asarray(y), split=np.asarray(roles),
                  identity=np.asarray(identities), run_id=np.asarray(runs))
    supported, counts = support_gate(arrays["y"], arrays["identity"], arrays["split"])
    patterns={}
    retained={}
    taxonomy=np.unique(sensor['y'])
    for identity in np.unique(arrays['identity']):
        selected=sensor['client']==identity
        activity_counts=[int(np.sum(selected & (sensor['y']==activity))) for activity in taxonomy]
        pattern=tuple(count>0 for count in activity_counts)
        patterns.setdefault(pattern,set()).add(labels[int(identity)])
        retained[int(identity)]=dict(window_counts=activity_counts,activity_labels=taxonomy.tolist())
    missing_explains=(len(patterns)>1 and all(len(value)==1 for value in patterns.values()))
    arrays['acquisition_supported']=np.array(not missing_explains)
    supported &= not missing_explains
    return arrays, dict(status="supported" if supported else "unsupported_attribute", mode=mode,
                        threshold_auxiliary_fit_median=threshold, fractions=fractions,
                        retained_activity_counts=retained,missing_block_pattern_explains_labels=missing_explains,
                        acquisition_gate='binary composition labels must not be completely determined by activity-block absence patterns',
                        class_user_counts=counts, feature_source="actual public models, aggregates, rosters, availability only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON list: trace,run_id,role(0=auxfit,1=auxval,2=target)")
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--mode", choices=["availability", "metadata", "full"], default="full")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sensor = dict(np.load(args.sensor, allow_pickle=False))
    arrays, report = build(manifest, sensor, mode=args.mode, rounds=args.rounds)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "attribute.npz", **arrays)
    (output / "support.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
