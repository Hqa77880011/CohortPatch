"""Prepare the official subject-disjoint UCI HAR feature files without downloading."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--har-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    root = Path(args.har_root)
    x_train = np.loadtxt(root / 'train/X_train.txt', dtype=np.float32)
    y_train = np.loadtxt(root / 'train/y_train.txt', dtype=np.int64) - 1
    subject_train = np.loadtxt(root / 'train/subject_train.txt', dtype=np.int64)
    x_test = np.loadtxt(root / 'test/X_test.txt', dtype=np.float32)
    y_test = np.loadtxt(root / 'test/y_test.txt', dtype=np.int64) - 1
    subject_test = np.loadtxt(root / 'test/subject_test.txt', dtype=np.int64)
    subjects = np.unique(subject_train)
    if len(subjects) != 21 or set(subjects) & set(subject_test):
        raise ValueError('Expected the official 21/9 disjoint subject assignment')
    np.random.default_rng(args.seed).shuffle(subjects)
    enrolled, validation, auxiliary_fit, auxiliary_validation = np.split(subjects, [12, 15, 18])
    mapping = {int(subject): index for index, subject in enumerate(enrolled)}
    mapping.update({int(subject): 12 + index for index, subject in enumerate(validation)})
    mapping.update({int(subject): 15 + index for index, subject in enumerate(np.unique(subject_test))})
    keep = np.isin(subject_train, np.r_[enrolled, validation])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, x=np.r_[x_train[keep], x_test], y=np.r_[y_train[keep], y_test],
                        client=np.array([mapping[int(s)] for s in np.r_[subject_train[keep], subject_test]]),
                        split=np.r_[np.where(np.isin(subject_train[keep], enrolled), 0, 1),
                                    np.full(len(x_test), 2)])
    roles = dict(seed=args.seed, enrolled=enrolled.tolist(), validation=validation.tolist(),
                 auxiliary_fit=auxiliary_fit.tolist(), auxiliary_validation=auxiliary_validation.tolist(),
                 test=np.unique(subject_test).tolist(), subject_to_client=mapping)
    output.with_suffix('.json').write_text(json.dumps(roles, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
