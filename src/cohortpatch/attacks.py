"""Frozen-update reconstruction or named-profile assignment, kept as distinct endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--updates', required=True, help='NPZ: updates[R,D], averaged equally over each roster')
    parser.add_argument('--reference', required=True, help='NPZ: profiles[N,D], independently obtained named profiles')
    parser.add_argument('--mode', choices=['frozen', 'assignment'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.records).read_text(encoding='utf-8').splitlines() if line.strip()]
    matches = [r for r in rows if r['id'] == args.case_id]
    if len(matches) != 1:
        raise ValueError('Select one unambiguous scheduling record')
    rosters = matches[0]['result']['rosters']
    aggregates = np.load(args.updates, allow_pickle=False)['updates']
    profiles = np.load(args.reference, allow_pickle=False)['profiles']
    matrix = np.zeros((len(rosters), len(profiles)))
    for t, roster in enumerate(rosters):
        matrix[t, roster] = 1
    if aggregates.shape != (len(rosters), profiles.shape[1]):
        raise ValueError('Aggregate and profile dimensions disagree')
    sums = aggregates * matrix.sum(axis=1, keepdims=True)
    if args.mode == 'frozen':
        reconstructed = np.linalg.lstsq(matrix, sums, rcond=None)[0]
        report = dict(frozen_normalized_reconstruction_error=float(np.sum((reconstructed - profiles)**2)
                      / max(float(np.sum(profiles**2)), 1e-12)), matrix_rank=int(np.linalg.matrix_rank(matrix)))
    else:
        # One named reference contribution per client; no test-label fitting.
        column_mass = matrix.sum(axis=0).clip(1)
        observed = (matrix.T @ sums) / column_mass[:, None]
        distances = ((observed[:, None, :] - profiles[None, :, :])**2).sum(axis=-1)
        predicted, named = linear_sum_assignment(distances)
        report = dict(profile_assignment_accuracy=float(np.mean(predicted == named)),
                      endpoint='static independent-profile assignment diagnostic')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
