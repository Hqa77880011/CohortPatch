"""Known-matrix frozen and temporally regularized dynamic reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def reconstruct(matrix, averages, *, alpha=0, dynamic=False, clip=1,
                steps=500, restarts=5, seed=17, known=None, known_ids=()):
    """Subtract declared colluders and reconstruct only the remaining clients."""
    matrix, averages = np.asarray(matrix, float), np.asarray(averages, float)
    if (matrix.ndim != 2 or averages.ndim != 2 or len(matrix) != len(averages)
            or not np.isfinite(matrix).all() or not np.isfinite(averages).all()
            or not np.isin(matrix, [0, 1]).all() or alpha < 0 or clip <= 0):
        raise ValueError("Expected finite binary rosters[R,N] and average updates[R,D]")
    sums = averages * matrix.sum(axis=1, keepdims=True)
    unknown = [i for i in range(matrix.shape[1]) if i not in known_ids]
    if not unknown:
        raise ValueError("At least one unknown client is required")
    if known_ids:
        if known is None or np.shape(known) != (len(matrix), matrix.shape[1], averages.shape[1]):
            raise ValueError("Declared colluders require aligned actual known updates[R,N,D]")
        values=np.asarray(known)[:,list(known_ids)]
        active=matrix[:,list(known_ids)].astype(bool)
        if not np.isfinite(values[active]).all():
            raise ValueError('Active colluders need actual finite known updates')
        values=np.where(active[:,:,None],values,0)
        sums -= np.einsum("rn,rnd->rd",matrix[:,list(known_ids)],values)
    a = matrix[:, unknown]
    if not dynamic:
        estimate = (np.linalg.pinv(a) @ sums if alpha == 0 else
                    np.linalg.solve(a.T @ a + alpha * np.eye(len(unknown)), a.T @ sums))
        return np.broadcast_to(estimate, (len(matrix), *estimate.shape)).copy(), unknown
    if min(steps, restarts) < 1:
        raise ValueError("Positive attack iteration and restart counts are required")
    rng = np.random.default_rng(seed)
    rate = 1 / max(2 * float((a * a).sum(axis=1).max()) + 8 * alpha, 1e-12)
    best = None
    for restart in range(restarts):
        value = (np.broadcast_to(np.linalg.pinv(a) @ sums,
                                (len(a), len(unknown), sums.shape[1])).copy()
                 if restart == 0 else rng.normal(0, clip / np.sqrt(sums.shape[1]),
                                                (len(a), len(unknown), sums.shape[1])))
        for _ in range(steps):
            residual = np.einsum("rn,rnd->rd", a, value) - sums
            gradient = 2 * a[:, :, None] * residual[:, None, :]
            differences = value[1:] - value[:-1]
            gradient[1:] += 2 * alpha * differences
            gradient[:-1] -= 2 * alpha * differences
            value -= rate * gradient
            norms = np.linalg.norm(value, axis=-1, keepdims=True)
            value *= np.minimum(1, clip / np.maximum(norms, 1e-12))
        residual = np.einsum("rn,rnd->rd", a, value) - sums
        objective = float(np.square(residual).sum() + alpha * np.square(np.diff(value, axis=0)).sum())
        if best is None or objective < best[0]:
            best = objective, value.copy()
    return best[1], unknown


def match_profiles(reconstructed, profiles):
    """One-to-one cosine matching of mean updates over the same public prefix."""
    left, right = np.asarray(reconstructed).mean(axis=0), np.asarray(profiles).mean(axis=0)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Profile references and reconstructed trajectories must share [R,N,D]")
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    row, column = linear_sum_assignment(-(left @ right.T))
    return dict(identity_accuracy=float(np.mean(row == column)),
                identity_macro_f1=float(np.mean(row == column)),
                candidates=len(row), assignment=column.tolist())


def reconstruction_error(estimate,truth,mask=None):
    if mask is None:
        mask=np.ones(truth.shape[:2],bool)
    else:
        mask=np.asarray(mask,bool)
    if mask.shape!=truth.shape[:2] or not mask.any() or not np.isfinite(truth[mask]).all():
        raise ValueError('Actual finite ground truth is required on every scored client-round')
    return float(np.square(estimate[mask]-truth[mask]).sum()/max(float(np.square(truth[mask]).sum()),1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ roster[R,N], updates[R,D], optional known[R,N,D]")
    parser.add_argument("--validation", required=True, help="Independent auxiliary NPZ with truth[R,N,D]")
    parser.add_argument('--truth',help='Restricted actual target scorer_updates.npz')
    parser.add_argument('--validation-truth',help='Restricted independent auxiliary scorer_updates.npz')
    parser.add_argument("--profiles", help="Independent named profile updates[R,N,D] NPZ")
    parser.add_argument("--mode", choices=["frozen", "dynamic"], required=True)
    parser.add_argument("--known-ids", nargs="*", type=int, default=[])
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation = dict(np.load(args.validation, allow_pickle=False))
    data = dict(np.load(args.data, allow_pickle=False))
    for record,path in ((data,args.truth),(validation,args.validation_truth)):
        if path:
            scorer=dict(np.load(path,allow_pickle=False))
            if not np.array_equal(record['run_id'],scorer['run_id']):
                raise ValueError('Actual scorer truth belongs to another public utility run')
            record.update(truth=scorer['truth'],truth_mask=scorer['truth_mask'])
        if args.known_ids and 'known' not in record and 'truth' in record:
            known=np.zeros_like(record['truth'])
            known[:,args.known_ids]=record['truth'][:,args.known_ids]
            record['known']=known
    if "run_id" not in data or "run_id" not in validation or set(data["run_id"]) & set(validation["run_id"]):
        raise ValueError("Disjoint complete target/auxiliary validation run_id values are required")
    trials = []
    dynamic = args.mode == "dynamic"
    for alpha in [0, 1e-4, 1e-2, 1]:
        for steps in args.steps if dynamic else [1]:
            estimate, unknown = reconstruct(validation["roster"], validation["updates"], alpha=alpha,
                dynamic=dynamic, clip=args.clip, steps=steps, restarts=args.restarts, seed=args.seed,
                known=validation.get("known"), known_ids=args.known_ids)
            truth = validation["truth"][:, unknown]
            error=reconstruction_error(estimate,truth,validation.get('truth_mask',np.ones(validation['truth'].shape[:2],bool))[:,unknown])
            trials.append((error, alpha, steps))
    _, alpha, steps = min(trials)
    estimate, unknown = reconstruct(data["roster"], data["updates"], alpha=alpha, dynamic=dynamic,
        clip=args.clip, steps=steps, restarts=args.restarts, seed=args.seed,
        known=data.get("known"), known_ids=args.known_ids)
    report = dict(mode=args.mode, alpha=alpha, steps=steps, restarts=args.restarts,
                  seed=args.seed, unknown_ids=unknown, validation_error=min(trials)[0])
    if "truth" in data:
        truth = data["truth"][:, unknown]
        report['normalized_reconstruction_error']=reconstruction_error(estimate,truth,data.get('truth_mask',np.ones(data['truth'].shape[:2],bool))[:,unknown])
    if args.profiles:
        profiles = dict(np.load(args.profiles, allow_pickle=False))
        if "sample_disjoint" not in profiles or not bool(profiles["sample_disjoint"]):
            raise ValueError("Named profiles require verified sample-disjoint recording blocks")
        if not np.array_equal(profiles.get('target_client_ids'),data.get('target_client_ids')):
            raise ValueError('Named profile identities/order differ from the actual target roster population')
        report.update(match_profiles(estimate, profiles["updates"][:, unknown]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reconstruction.npz", updates=estimate, unknown_ids=unknown)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
