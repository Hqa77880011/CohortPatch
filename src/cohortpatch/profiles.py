"""Chronologically disjoint named profiles evaluated at each actual public model."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .activity import load_model


def split_blocks(sensor, fraction=0.5):
    """Separate enrolled WISDM acquisition blocks with a full-window time gap.

    HAR lacks verifiable per-window acquisition timestamps and therefore cannot
    support this stronger-prior condition through fabricated chronological IDs.
    """
    required = {"start", "end", "client", "split"}
    if not required <= set(sensor):
        return None, None, dict(status="unsupported_profile_prior", reason="No verifiable per-window acquisition times")
    if not 0 < fraction < 1:
        raise ValueError("Public chronological cut fraction must lie in (0,1)")
    victim = sensor["split"] != 0
    profile = np.zeros(len(sensor["client"]), bool)
    blocks = []
    for client in np.unique(sensor["client"][sensor["split"] == 0]):
        indices = np.flatnonzero((sensor["client"] == client) & (sensor["split"] == 0))
        starts, ends = sensor["start"][indices], sensor["end"][indices]
        gap = int(np.max(ends - starts))
        cut = int(starts.min() + fraction * (ends.max() - starts.min()))
        left = indices[ends <= cut - gap // 2]
        right = indices[starts >= cut + (gap - gap // 2)]
        if not len(left) or not len(right):
            return None, None, dict(status="unsupported_profile_prior", reason=f"Insufficient independent recording blocks for client {client}")
        victim[left], profile[right] = True, True
        blocks.append(dict(client=int(client), victim_end_ns=int(sensor["end"][left].max()),
                           profile_start_ns=int(sensor["start"][right].min()), minimum_gap_ns=gap))
    subset = lambda mask: {key: value[mask] if np.ndim(value) and len(value) == len(mask) else value for key, value in sensor.items()}
    return subset(victim), subset(profile), dict(status="supported", sample_disjoint=True, blocks=blocks,
                                                rule="public chronological cut with gap >= one complete sensor window")


def reference_updates(sensor, trace, checkpoint, manifest, *, epochs=1, steps=None, batch_size=32, seed=17):
    """Run stateless identical local optimization on independently held prior data."""
    if manifest.get("status") != "supported" or not manifest.get("sample_disjoint"):
        raise ValueError("Named prior requires a supported independent acquisition-block manifest")
    if not {"start", "end"} <= set(sensor):
        raise ValueError("Profile acquisition timestamps are missing")
    torch.set_num_threads(1)
    model, metadata = load_model(checkpoint)
    ids = trace["target_client_ids"]
    blocks = {int(block["client"]): block for block in manifest["blocks"]}
    x = (torch.tensor(sensor["x"], dtype=torch.float32) - metadata["mean"]) / metadata["std"]
    y = torch.tensor(sensor["y"], dtype=torch.long)
    rng, references = np.random.default_rng(seed), []
    optimizer = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[metadata["optimizer"]]
    for public in trace["public_models"]:
        offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                length = parameter.numel()
                parameter.copy_(torch.tensor(public[offset:offset + length].reshape(parameter.shape)))
                offset += length
        round_updates = []
        for identity in ids:
            pool = np.flatnonzero(sensor["client"] == identity)
            block = blocks[int(identity)]
            if (not len(pool) or sensor["start"][pool].min() < block["profile_start_ns"]
                    or block["profile_start_ns"] - block["victim_end_ns"] < block["minimum_gap_ns"]):
                raise ValueError("Profile samples overlap or violate their declared recording-block gap")
            actual = metadata.get("training_ranges", {}).get(str(int(identity)))
            if actual is None or actual["end_ns"] > block["victim_end_ns"]:
                raise ValueError("Actual victim checkpoint does not establish the claimed independent recording block")
            local = copy.deepcopy(model); local.train()
            trainer = optimizer(local.parameters(), lr=metadata["learning_rate"])
            batches = ([rng.choice(pool, min(batch_size, len(pool)), replace=False) for _ in range(steps)] if steps else
                       [order[start:start + batch_size] for _ in range(epochs) for order in [rng.permutation(pool)] for start in range(0, len(pool), batch_size)])
            for indices in batches:
                trainer.zero_grad()
                loss = nn.functional.cross_entropy(local(x[indices]), y[indices]); loss.backward()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite profile loss")
                trainer.step()
            update = torch.cat([(a.detach() - b.detach()).flatten() for a, b in zip(local.parameters(), model.parameters())])
            update *= min(1.0, metadata["clip_norm"] / max(float(update.norm()), 1e-12))
            round_updates.append(update.numpy())
        references.append(round_updates)
    return np.asarray(references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("--sensor", required=True); split.add_argument("--fraction", type=float, default=0.5)
    build = sub.add_parser("build")
    for name in ("sensor", "trace", "checkpoint", "manifest"):
        build.add_argument("--" + name, required=True)
    for item in (split, build):
        item.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sensor = dict(np.load(args.sensor, allow_pickle=False))
    if args.command == "split":
        victim, prior, manifest = split_blocks(sensor, args.fraction)
        if victim is not None:
            np.savez_compressed(output / "victim.npz", **victim)
            np.savez_compressed(output / "prior.npz", **prior)
        (output / "blocks.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        trace = dict(np.load(args.trace, allow_pickle=False))
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        metadata = torch.load(args.checkpoint, weights_only=True)
        updates = reference_updates(sensor, trace, args.checkpoint, manifest,
                                    epochs=metadata["local_epochs"], steps=metadata["local_steps"], batch_size=metadata["batch_size"])
        np.savez_compressed(output / "profiles.npz", updates=updates, sample_disjoint=np.array(True),
                            target_client_ids=trace["target_client_ids"])


if __name__ == "__main__":
    main()
