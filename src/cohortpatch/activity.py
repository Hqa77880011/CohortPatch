import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from .protocol import RosterLedger


def make_model(features, classes, architecture="mlp"):
    """Construct the manuscript 32/64-channel CNN or the legacy feature MLP."""
    if architecture == "cnn":
        return nn.Sequential(nn.Conv1d(features, 32, 5, padding=2), nn.ReLU(),
                             nn.MaxPool1d(2), nn.Conv1d(32, 64, 3, padding=1),
                             nn.ReLU(), nn.MaxPool1d(2), nn.AdaptiveAvgPool1d(1),
                             nn.Flatten(), nn.Linear(64, classes))
    return nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Linear(64, classes))


def load_model(path):
    """Load a locally generated checkpoint with its training-only normalization."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["input_features"], checkpoint["classes"], checkpoint.get("architecture", "mlp"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, help="Fixed local minibatch count for functional checks")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--architecture", choices=["auto", "mlp", "cnn"], default="auto")
    parser.add_argument("--optimizer", choices=["auto", "sgd", "adam"], default="auto")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument('--run-id',help='Independent complete utility/attack run identifier')
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--training-role", type=int, choices=[0, 3, 4], default=0)
    parser.add_argument("--client-ids", type=int, nargs="+", help="Sensor client IDs corresponding to local roster IDs")
    parser.add_argument("--backend", choices=["ideal", "secaggplus"], default="ideal")
    parser.add_argument("--secagg-python", help="Python in a dedicated flwr==1.23.0 environment")
    args = parser.parse_args()
    if args.backend == "secaggplus" and not args.secagg_python:
        parser.error("SecAgg+ requires --secagg-python with its pinned isolated environment")
    if (min(args.epochs, args.batch_size, args.threads) < 1 or args.clip_norm <= 0
            or (args.steps is not None and args.steps < 1)
            or (args.learning_rate is not None and args.learning_rate <= 0)):
        parser.error("Positive steps, batch size, threads and learning rate are required")
    torch.set_num_threads(args.threads)
    arrays = dict(np.load(args.data, allow_pickle=False))
    x = torch.tensor(arrays["x"], dtype=torch.float32)
    y = torch.tensor(arrays["y"], dtype=torch.long)
    client = arrays["client"]
    split = arrays["split"]
    train = split == args.training_role
    if args.client_ids:
        train &= np.isin(client, args.client_ids)
    architecture = ("cnn" if x.ndim == 3 else "mlp") if args.architecture == "auto" else args.architecture
    if (x.ndim != (3 if architecture == "cnn" else 2) or not torch.isfinite(x).all() or not train.any()
            or len(x) != len(y) or len(x) != len(split) or len(x) != len(client)):
        raise ValueError("Expected aligned finite x[N,D], y[N], client[N], split[N]")
    if any(set(client[split == a]) & set(client[split == b])
           for a in set(split) for b in set(split) if a < b):
        raise ValueError("Utility training and test subjects must be disjoint")
    labels = sorted(set(y[train].tolist()))
    utility = split <= 2
    if labels != list(range(len(labels))) or not set(y[utility].tolist()) <= set(labels):
        raise ValueError("Labels must be zero-based and every class defined by training data")
    axes = (0, 2) if architecture == "cnn" else (0,)
    mean = x[train].mean(dim=axes, keepdim=True)
    std = x[train].std(dim=axes, unbiased=False, keepdim=True).clamp_min(1e-8)
    x = (x - mean) / std
    with Path(args.records).open(encoding="utf-8") as stream:
        matches = [row for row in map(json.loads, stream) if row["id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError("Select one unambiguous scheduling episode")
    rosters = matches[0]["result"]["rosters"]
    scheduling = matches[0]["result"]
    policy = scheduling["policy"]
    method=matches[0].get('method','cohortpatch')
    historical_k=1 if method in ('no_history','no_complement') else policy['k']
    client_ids = args.client_ids or sorted(set(client[train].tolist()))
    if len(client_ids) != policy["clients"]:
        raise ValueError("Training sensor clients must match the enrolled roster population")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = RosterLedger(output / 'rosters.sqlite',scheduling['initial_classes'],historical_k,
                          policy['capacity'], policy['threshold'])
    if args.backend == "secaggplus":
        (output / "native_policy.json").write_text(json.dumps(dict(
            initial_classes=scheduling["initial_classes"],k=historical_k,
            capacity=policy["capacity"], threshold=policy["threshold"])), encoding="utf-8")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    learning_rate = args.learning_rate or (0.001 if architecture == "cnn" else 0.03)
    optimizer_name = ("adam" if architecture == "cnn" else "sgd") if args.optimizer == "auto" else args.optimizer
    model = make_model(x.shape[1], len(labels), architecture)
    aggregates = []
    scorer_updates=[]
    public_models = []
    history = []
    for round_index, roster in enumerate(rosters):
        public_models.append(torch.cat([p.detach().flatten() for p in model.parameters()]).numpy().copy())
        local_states = []
        losses, gradients = [], []
        for participant in roster:
            pool = np.flatnonzero((client == client_ids[participant]) & train)
            if not len(pool):
                raise ValueError("Authorized participant has no local training samples")
            local = copy.deepcopy(model)
            local.train()
            optimizer = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer_name](local.parameters(), lr=learning_rate)
            batches = ([rng.choice(pool, min(args.batch_size, len(pool)), replace=False) for _ in range(args.steps)]
                       if args.steps is not None else [order[start:start + args.batch_size]
                       for _ in range(args.epochs) for order in [rng.permutation(pool)]
                       for start in range(0, len(pool), args.batch_size)])
            for selected in batches:
                optimizer.zero_grad()
                value = F.cross_entropy(local(x[selected]), y[selected])
                value.backward()
                grad_norm = torch.sqrt(sum((p.grad.square().sum() for p in local.parameters() if p.grad is not None)))
                if not torch.isfinite(value) or not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite local loss or gradient")
                optimizer.step()
                losses.append(float(value.detach()))
                gradients.append(float(grad_norm))
            with torch.no_grad():
                norm = torch.sqrt(sum((a - b).square().sum() for a, b in zip(local.parameters(), model.parameters())))
                factor = min(1.0, args.clip_norm / max(float(norm), 1e-12))
                for a, b in zip(local.parameters(), model.parameters()):
                    a.copy_(b + factor * (a - b))
            local_states.append(
                {key: value.detach().clone() for key, value in local.state_dict().items()}
            )
        if local_states:
            session = args.case_id + ':' + str(round_index)
            ledger.reserve(session, roster, scheduling['uploaded'][round_index])
            base=np.concatenate([tensor.detach().numpy().ravel() for tensor in model.state_dict().values()])
            vectors = {client_id: np.concatenate([state[key].numpy().ravel() for key in state])-base
                       for client_id, state in zip(roster, local_states)}
            truth=np.full((policy['clients'],len(base)),np.nan)
            for client_id,vector in vectors.items():
                truth[client_id]=vector
            scorer_updates.append(truth)
            if args.backend == "ideal":
                ledger.freeze(session, roster)
                released = ledger.release(session, vectors) / len(roster)
                protocol = dict(backend="ideal local sum")
            else:
                client_path = output / f"native_clients_{round_index}.npz"
                response_path = output / f"native_response_{round_index}.json"
                np.savez_compressed(client_path, client_ids=np.asarray(roster),
                                    vectors=np.stack([vectors[i] for i in roster]))
                environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
                subprocess.run([args.secagg_python, "-m", "cohortpatch.secaggplus", "--data", str(client_path),
                    "--policy", str(output / "native_policy.json"), "--ledger", str(output / "rosters.sqlite"),
                    "--session", session,'--clipping-range',str(args.clip_norm), "--output", str(response_path)], env=environment, check=True)
                protocol = json.loads(response_path.read_text(encoding="utf-8"))
                released = np.asarray(protocol.pop("answer")) / len(roster)
            offset = 0
            averaged = {}
            for key, tensor in local_states[0].items():
                length = tensor.numel()
                averaged[key] = torch.as_tensor((base[offset:offset+length]+released[offset:offset+length]).reshape(tensor.shape), dtype=tensor.dtype)
                offset += length
            before = torch.cat([p.detach().flatten() for p in model.parameters()])
            model.load_state_dict(averaged)
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            aggregates.append((after - before).numpy())
        else:
            protocol = dict(backend=args.backend, status="empty roster")
            aggregates.append(
                np.zeros(sum(p.numel() for p in model.parameters()), dtype=np.float32)
            )
            scorer_updates.append(np.full((policy['clients'],sum(p.numel() for p in model.parameters())),np.nan))
        history.append({"round": round_index + 1, "included": len(roster),
                        "loss": float(np.mean(losses)) if losses else None,
                        "gradient_norm": float(np.mean(gradients)) if gradients else None,
                        "protocol": protocol})
    evaluation_role=2 if args.training_role==0 else (4 if args.training_role==3 else 3)
    test=split==evaluation_role
    if not test.any():
        raise ValueError("An independent test split is required")
    model.eval()
    with torch.no_grad():
        predicted = model(x[test]).argmax(-1).numpy()
    prefix='utility' if args.training_role==0 else 'auxiliary'
    report = {
        'method':method,'historical_k_enforced':historical_k,'crypto_threshold':policy['threshold'],
        prefix+"_macro_f1": float(
            f1_score(y[test].numpy(), predicted, average="macro", zero_division=0)
        ),
        prefix+"_accuracy": float(accuracy_score(y[test].numpy(), predicted)),
        "included_updates": sum(map(len, rosters)),
        "rounds": len(rosters),
    }
    ledger.close()
    checkpoint = dict(state_dict=model.state_dict(), input_features=x.shape[1], classes=len(labels),
                      mean=mean, std=std, seed=args.seed, learning_rate=learning_rate,
                      local_steps=args.steps, local_epochs=args.epochs, optimizer=optimizer_name,
                      architecture=architecture, input_shape=list(x.shape[1:]),
                      clip_norm=args.clip_norm, batch_size=args.batch_size, backend=args.backend)
    if {"start", "end"} <= set(arrays):
        checkpoint["training_ranges"] = {str(int(identity)): dict(
            start_ns=int(arrays["start"][(client == identity) & train].min()),
            end_ns=int(arrays["end"][(client == identity) & train].max())) for identity in client_ids}
    torch.save(checkpoint, output / "activity.pt")
    restored, _ = load_model(output / "activity.pt")
    with torch.no_grad():
        torch.testing.assert_close(model(x[test]), restored(x[test]), rtol=1e-6, atol=1e-7)
    report["checkpoint_roundtrip"] = True
    np.savez_compressed(output / "server_aggregates.npz", updates=np.stack(aggregates),
                        run_id=np.array([args.run_id or f'{args.case_id}:role{args.training_role}:seed{args.seed}']),
                        public_models=np.stack(public_models),
                        roster=np.array([[int(i in row) for i in range(policy["clients"])] for row in rosters]),
                        layer_sizes=np.array([p.numel() for p in model.parameters()]),
                        availability=np.array([[int(i in row) for i in range(policy["clients"])] for row in scheduling["uploaded"]]),
                        target_client_ids=np.asarray(client_ids),
                        initial_class=np.array([next(j for j, group in enumerate(scheduling["initial_classes"]) if i in group)
                                                for i in range(policy["clients"])]))
    np.savez_compressed(output/'scorer_updates.npz',truth=np.stack(scorer_updates),
                        truth_mask=np.array([[int(i in row) for i in range(policy['clients'])] for row in rosters]),
                        run_id=np.array([args.run_id or f'{args.case_id}:role{args.training_role}:seed{args.seed}']))
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "rounds.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
