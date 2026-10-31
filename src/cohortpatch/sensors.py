"""Prepare real UCI HAR/WISDM inertial windows with subject and segment roles."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


WISDM_ACTIVITIES = "ABCDEFGHIJKLMOPQRS"


def wisdm_windows(root, window=128, stride=64, frequency=20, maximum_gap=3):
    """Window smartphone accelerometer records within uninterrupted acquisition."""
    x, y, subjects, starts, ends, segments = [], [], [], [], [], []
    rejected = Counter()
    files = sorted(Path(root).rglob("*_accel_phone.txt"))
    if not files:
        raise FileNotFoundError("No WISDM *_accel_phone.txt acquisition files")
    for path in files:
        buffer, label, subject, previous, segment = [], None, None, None, 0

        def flush():
            for offset in range(0, len(buffer) - window + 1, stride):
                rows = buffer[offset:offset + window]
                x.append(np.array([row[1:] for row in rows], dtype=np.float32).T)
                y.append(WISDM_ACTIVITIES.index(label))
                subjects.append(subject)
                starts.append(rows[0][0])
                ends.append(rows[-1][0])
                segments.append(path.name + ":" + str(segment))

        with path.open(encoding="utf-8") as stream:
            for row in csv.reader(stream):
                try:
                    if len(row) != 6:
                        raise ValueError("column count")
                    current_subject, current_label = int(row[0]), row[1].strip()
                    timestamp = int(row[2])
                    values = [float(value.rstrip("; ")) for value in row[3:]]
                    if current_label not in WISDM_ACTIVITIES or not np.isfinite(values).all():
                        raise ValueError("activity or nonfinite sensor")
                except ValueError:
                    rejected["corrupt_record"] += 1
                    flush()
                    buffer, previous = [], None
                    segment += 1
                    continue
                boundary = (previous is not None and (
                    timestamp <= previous or timestamp - previous > maximum_gap * 1e9 / frequency
                    or current_label != label or current_subject != subject))
                if boundary:
                    rejected["acquisition_boundary"] += 1
                    flush()
                    buffer = []
                    segment += 1
                label, subject, previous = current_label, current_subject, timestamp
                buffer.append((timestamp, *values))
        flush()
    if not x:
        raise ValueError("No complete valid inertial windows")
    return dict(x=np.stack(x), y=np.array(y), subject=np.array(subjects),
                start=np.array(starts, dtype=np.int64), end=np.array(ends, dtype=np.int64),
                segment=np.array(segments)), dict(rejected)


def assign_roles(subjects, seed, official_test=None, auxiliary_per_role=4):
    """Reserve auxiliary identities before enrollment; never recycle utility test."""
    population = np.unique(subjects)
    rng = np.random.default_rng(seed)
    if official_test is None:
        rng.shuffle(population)
        train_end, validation_end = int(len(population) * 0.6), int(len(population) * 0.8)
        training, validation, test = np.split(population, [train_end, validation_end])
    else:
        test = np.asarray(sorted(official_test))
        training = population[~np.isin(population, test)]
        rng.shuffle(training)
        validation, training = training[:3], training[3:]
    if len(training) <= 2 * auxiliary_per_role:
        raise ValueError("Not enough independent training subjects for requested auxiliary roles")
    fit = training[:auxiliary_per_role]
    auxiliary_validation = training[auxiliary_per_role:2 * auxiliary_per_role]
    enrolled = training[2 * auxiliary_per_role:]
    roles = [enrolled, validation, test, fit, auxiliary_validation]
    mapping, split, offset = {}, {}, 0
    for role, identities in enumerate(roles):
        for identity in identities:
            mapping[int(identity)] = offset
            split[int(identity)] = role
            offset += 1
    return mapping, split, {name: ids.tolist() for name, ids in zip(
        ("enrolled", "utility_validation", "utility_test", "auxiliary_fit", "auxiliary_validation"), roles)}


def har_windows(root, signals="total_acc"):
    """Read official 128-sample HAR inertial windows and preserve its 21/9 split."""
    root = Path(root)
    arrays = []
    test_subjects = None
    for split in ("train", "test"):
        channels = [np.loadtxt(root / split / "Inertial Signals" / f"{signals}_{axis}_{split}.txt",
                               dtype=np.float32, ndmin=2) for axis in "xyz"]
        values = np.stack(channels, axis=1)
        labels = np.loadtxt(root / split / f"y_{split}.txt", dtype=int, ndmin=1) - 1
        subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int, ndmin=1)
        if values.shape != (len(labels), 3, 128) or len(subjects) != len(labels):
            raise ValueError("HAR windows must align as [N,3,128], labels and subjects")
        arrays.append((values, labels, subjects))
        if split == "test":
            test_subjects = set(subjects)
    if set(arrays[0][2]) & test_subjects:
        raise ValueError("Official HAR utility subject split overlaps")
    return dict(x=np.concatenate([row[0] for row in arrays]),
                y=np.concatenate([row[1] for row in arrays]),
                subject=np.concatenate([row[2] for row in arrays])), test_subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["har", "wisdm"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--auxiliary-per-role", type=int, default=4)
    args = parser.parse_args()
    if args.source == "har":
        arrays, official = har_windows(args.root)
        rejected = {}
    else:
        arrays, rejected = wisdm_windows(args.root)
        official = None
    mapping, role, manifest = assign_roles(arrays["subject"], args.seed, official, args.auxiliary_per_role)
    arrays["client"] = np.array([mapping[int(s)] for s in arrays["subject"]])
    arrays["split"] = np.array([role[int(s)] for s in arrays["subject"]])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **arrays)
    manifest.update(source=args.source, seed=args.seed, window=128, stride=64,
                    walking_label=0, subject_to_client=mapping, rejected=rejected,
                    profile_recording_metadata_available=args.source == "wisdm",
                    counts={str(i): int(sum(arrays["split"] == i)) for i in range(5)})
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
