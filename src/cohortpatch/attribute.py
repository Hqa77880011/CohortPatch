"""Appendix GRU composition-attribute attack with independent-user support gates."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch
from torch import nn


class AttributeGRU(nn.Module):
    """Two-layer 64-unit GRU and 64-unit ReLU binary head."""

    def __init__(self, features):
        super().__init__()
        self.gru = nn.GRU(features, 64, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, sequence):
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])


def public_features(trace, target, mode="full", projection_seed=17):
    """Build only declared public feature channels; no target update is read."""
    roster = np.asarray(trace["roster"])
    availability = np.asarray(trace["availability"])
    aggregates = np.asarray(trace["updates"])
    models = np.asarray(trace["public_models"])
    if roster.shape != availability.shape or aggregates.shape != models.shape or len(roster) != len(aggregates):
        raise ValueError("Public traces must align by complete round and client")
    signatures = [tuple([int(c)]) for c in trace.get("initial_class", np.zeros(roster.shape[1], int))]
    sizes = []
    for row in roster:
        sizes.append(signatures.count(signatures[target]))
        signatures = [signature + (int(bit),) for signature, bit in zip(signatures, row)]
    debt = np.r_[0, np.cumsum(availability[:-1, target] - roster[:-1, target])]
    metadata = np.column_stack([availability[:, target], roster[:, target], sizes,
                                roster.sum(axis=1), debt])
    if mode in ("availability", "metadata"):
        return metadata[:, [0]] if mode == "availability" else metadata
    moments = []
    offset = 0
    for length in trace["layer_sizes"]:
        for values in (models, aggregates):
            layer = values[:, offset:offset + int(length)]
            moments.extend([layer.mean(axis=1), layer.std(axis=1)])
        offset += int(length)
    if offset != models.shape[1]:
        raise ValueError("Public layer sizes do not cover update coordinates")
    joined = np.concatenate([models, aggregates], axis=1)
    projection = np.random.default_rng(projection_seed).choice([-1, 1], size=(joined.shape[1], 64)) / 8
    return np.column_stack([metadata, *moments, joined @ projection]).astype(np.float32)


def composition_labels(sensor):
    """Freeze the walking-fraction median on independent auxiliary fitting users."""
    ids = np.unique(sensor["client"])
    fractions = {int(client): float(np.mean(sensor["y"][sensor["client"] == client] == 0)) for client in ids}
    auxiliary = np.unique(sensor["client"][sensor["split"] == 3])
    if not len(auxiliary):
        raise ValueError("Composition threshold requires independent auxiliary fitting identities")
    threshold = float(np.median([fractions[int(client)] for client in auxiliary]))
    return {client: int(value > threshold) for client, value in fractions.items()}, threshold, fractions


def support_gate(labels, identities, split):
    """At least two distinct users per class in fitting, validation and targets."""
    counts = {str(role): {str(label): len(set(identities[(split == role) & (labels == label)].tolist()))
                         for label in (0, 1)} for role in (0, 1, 2)}
    return all(count >= 2 for role in counts.values() for count in role.values()), counts


def fit_attack(sequence, labels, split, identities, groups, *, epochs=100,
               rates=(1e-4, 1e-3), seed=17, batch_size=32):
    """Fit on complete auxiliary runs, select by validation NLL, freeze test model."""
    if (sequence.ndim != 3 or len(sequence) != len(labels) or not np.isfinite(sequence).all()
            or not set(np.unique(split)) == {0, 1, 2}):
        raise ValueError("Finite x[episodes,rounds,features] and three aligned roles are required")
    for a in (0, 1, 2):
        for b in range(a + 1, 3):
            if set(groups[split == a]) & set(groups[split == b]) or set(identities[split == a]) & set(identities[split == b]):
                raise ValueError("Attack roles must use disjoint complete runs and identities")
    supported, counts = support_gate(labels, identities, split)
    if not supported:
        return None, dict(status="unsupported_attribute", class_user_counts=counts)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = sequence[split == 0].mean(axis=(0, 1), keepdims=True)
    std = sequence[split == 0].std(axis=(0, 1), keepdims=True).clip(1e-8)
    x = torch.as_tensor((sequence - mean) / std, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    train, validation, test = [np.flatnonzero(split == role) for role in (0, 1, 2)]
    best = None
    for rate in rates:
        torch.manual_seed(seed)
        model = AttributeGRU(x.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        stale = 0
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(train)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Attribute attack has nonfinite loss or gradients")
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(nn.functional.cross_entropy(model(x[validation]), y[validation]))
            if best is None or value < best[0]:
                best = value, copy.deepcopy(model.state_dict()), rate, epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= 10:
                    break
    model = AttributeGRU(x.shape[-1])
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        probabilities = model(x[test]).softmax(-1)[:, 1].numpy()
        validation_probabilities = model(x[validation]).softmax(-1)[:, 1].numpy()
    report = dict(status="evaluated", auc=float(roc_auc_score(labels[test], probabilities)),
                  pr_auc=float(average_precision_score(labels[test], probabilities)),
                  balanced_accuracy=float(balanced_accuracy_score(labels[test], probabilities > 0.5)),
                  prevalence=float(labels[test].mean()), validation_nll=best[0],
                  validation_auc=float(roc_auc_score(labels[validation], validation_probabilities)),
                  learning_rate=best[2], selected_epoch=best[3], class_user_counts=counts,
                  validation_adequacy="inconclusive" if roc_auc_score(labels[validation], validation_probabilities) <= 0.5 else "above_chance")
    checkpoint = dict(state_dict=best[1], features=x.shape[-1], mean=torch.as_tensor(mean),
                      std=torch.as_tensor(std), seed=seed)
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ x[M,R,F], y[M], split[M], identity[M], run_id[M]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rates", type=float, nargs="+", default=[1e-4, 1e-3])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    arrays = dict(np.load(args.data, allow_pickle=False))
    if 'acquisition_supported' in arrays and not bool(arrays['acquisition_supported']):
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'metrics.json').write_text(json.dumps(dict(status='unsupported_attribute',reason='composition labels explained by missing activity acquisition blocks')),encoding='utf-8')
        return
    checkpoint, report = fit_attack(arrays["x"], arrays["y"], arrays["split"], arrays["identity"],
                                    arrays["run_id"], epochs=args.epochs, rates=args.rates, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        torch.save(checkpoint, output / "attribute.pt")
        restored = torch.load(output / "attribute.pt", weights_only=True)
        model = AttributeGRU(restored["features"])
        model.load_state_dict(restored["state_dict"])
        model.eval()
        sample = torch.as_tensor((arrays["x"] - restored["mean"].numpy()) / restored["std"].numpy(), dtype=torch.float32)
        reference = AttributeGRU(checkpoint["features"])
        reference.load_state_dict(checkpoint["state_dict"])
        reference.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(sample), reference(sample))
        report["checkpoint_roundtrip"] = True
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
