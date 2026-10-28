import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from .protocol import RosterLedger


def make_model(features, classes):
    """Create the small equally weighted utility-model implementation."""
    return nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Linear(64, classes))


def load_model(path):
    """Load a locally generated checkpoint with its training-only normalization."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["input_features"], checkpoint["classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Positive steps, batch size, threads and learning rate are required")
    torch.set_num_threads(args.threads)
    arrays = dict(np.load(args.data, allow_pickle=False))
    x = torch.tensor(arrays["x"], dtype=torch.float32)
    y = torch.tensor(arrays["y"], dtype=torch.long)
    client = arrays["client"]
    split = arrays["split"]
    train = split == 0
    if (x.ndim != 2 or not torch.isfinite(x).all() or not train.any()
            or len(x) != len(y) or len(x) != len(split) or len(x) != len(client)):
        raise ValueError("Expected aligned finite x[N,D], y[N], client[N], split[N]")
    if set(client[train]) & set(client[split == 2]):
        raise ValueError("Utility training and test subjects must be disjoint")
    labels = sorted(set(y[train].tolist()))
    if labels != list(range(len(labels))) or not set(y.tolist()) <= set(labels):
        raise ValueError("Labels must be zero-based and every class defined by training data")
    mean = x[train].mean(dim=0)
    std = x[train].std(dim=0, unbiased=False).clamp_min(1e-8)
    x = (x - mean) / std
    with Path(args.records).open(encoding="utf-8") as stream:
        matches = [row for row in map(json.loads, stream) if row["id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError("Select one unambiguous scheduling episode")
    rosters = matches[0]["result"]["rosters"]
    scheduling = matches[0]["result"]
    policy = scheduling["policy"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = RosterLedger(output / 'rosters.sqlite', scheduling['initial_classes'], policy['k'],
                          policy['capacity'], policy['threshold'])
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_model(x.shape[1], len(labels))
    aggregates = []
    history = []
    for round_index, roster in enumerate(rosters):
        local_states = []
        losses, gradients = [], []
        for participant in roster:
            pool = np.flatnonzero((client == participant) & (split == 0))
            if not len(pool):
                raise ValueError("Authorized participant has no local training samples")
            local = copy.deepcopy(model)
            local.train()
            optimizer = torch.optim.SGD(local.parameters(), lr=args.learning_rate)
            for _ in range(args.steps):
                selected = rng.choice(pool, min(args.batch_size, len(pool)), replace=False)
                optimizer.zero_grad()
                value = F.cross_entropy(local(x[selected]), y[selected])
                value.backward()
                grad_norm = torch.sqrt(sum((p.grad.square().sum() for p in local.parameters() if p.grad is not None)))
                if not torch.isfinite(value) or not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite local loss or gradient")
                optimizer.step()
                losses.append(float(value.detach()))
                gradients.append(float(grad_norm))
            local_states.append(
                {key: value.detach().clone() for key, value in local.state_dict().items()}
            )
        if local_states:
            session = args.case_id + ':' + str(round_index)
            ledger.reserve(session, roster, scheduling['uploaded'][round_index])
            ledger.freeze(session, roster)
            vectors = {client_id: np.concatenate([state[key].numpy().ravel() for key in state])
                       for client_id, state in zip(roster, local_states)}
            released = ledger.release(session, vectors) / len(roster)
            offset = 0
            averaged = {}
            for key, tensor in local_states[0].items():
                length = tensor.numel()
                averaged[key] = torch.as_tensor(released[offset:offset+length].reshape(tensor.shape), dtype=tensor.dtype)
                offset += length
            before = torch.cat([p.detach().flatten() for p in model.parameters()])
            model.load_state_dict(averaged)
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            aggregates.append((after - before).numpy())
        else:
            aggregates.append(
                np.zeros(sum(p.numel() for p in model.parameters()), dtype=np.float32)
            )
        history.append({"round": round_index + 1, "included": len(roster),
                        "loss": float(np.mean(losses)) if losses else None,
                        "gradient_norm": float(np.mean(gradients)) if gradients else None})
    test = split == 2
    if not test.any():
        raise ValueError("An independent test split is required")
    model.eval()
    with torch.no_grad():
        predicted = model(x[test]).argmax(-1).numpy()
    report = {
        "utility_macro_f1": float(
            f1_score(y[test].numpy(), predicted, average="macro", zero_division=0)
        ),
        "utility_accuracy": float(accuracy_score(y[test].numpy(), predicted)),
        "included_updates": sum(map(len, rosters)),
        "rounds": len(rosters),
    }
    ledger.close()
    checkpoint = dict(state_dict=model.state_dict(), input_features=x.shape[1], classes=len(labels),
                      mean=mean, std=std, seed=args.seed, learning_rate=args.learning_rate,
                      local_steps=args.steps, batch_size=args.batch_size)
    torch.save(checkpoint, output / "activity.pt")
    restored, _ = load_model(output / "activity.pt")
    with torch.no_grad():
        torch.testing.assert_close(model(x[test]), restored(x[test]), rtol=1e-6, atol=1e-7)
    report["checkpoint_roundtrip"] = True
    np.savez_compressed(output / "server_aggregates.npz", updates=np.stack(aggregates))
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "rounds.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
