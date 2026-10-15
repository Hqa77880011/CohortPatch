import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    arrays = dict(np.load(args.data, allow_pickle=False))
    x = torch.tensor(arrays["x"], dtype=torch.float32)
    y = torch.tensor(arrays["y"], dtype=torch.long)
    client = arrays["client"]
    split = arrays["split"]
    with Path(args.records).open(encoding="utf-8") as stream:
        matches = [row for row in map(json.loads, stream) if row["id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError("Select one unambiguous scheduling episode")
    rosters = matches[0]["result"]["rosters"]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, int(y.max()) + 1))
    aggregates = []
    history = []
    for round_index, roster in enumerate(rosters):
        local_states = []
        for participant in roster:
            pool = np.flatnonzero((client == participant) & (split == 0))
            if not len(pool):
                raise ValueError("Authorized participant has no local training samples")
            local = copy.deepcopy(model)
            optimizer = torch.optim.SGD(local.parameters(), lr=0.03)
            for _ in range(args.steps):
                selected = rng.choice(pool, min(args.batch_size, len(pool)), replace=False)
                optimizer.zero_grad()
                value = F.cross_entropy(local(x[selected]), y[selected])
                value.backward()
                optimizer.step()
            local_states.append(
                {key: value.detach().clone() for key, value in local.state_dict().items()}
            )
        if local_states:
            averaged = {
                key: torch.stack([state[key] for state in local_states]).mean(0)
                for key in local_states[0]
            }
            before = torch.cat([p.detach().flatten() for p in model.parameters()])
            model.load_state_dict(averaged)
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            aggregates.append((after - before).numpy())
        else:
            aggregates.append(
                np.zeros(sum(p.numel() for p in model.parameters()), dtype=np.float32)
            )
        history.append({"round": round_index + 1, "included": len(roster)})
    test = split == 2
    if not test.any():
        raise ValueError("An independent test split is required")
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "activity.pt")
    np.savez_compressed(output / "server_aggregates.npz", updates=np.stack(aggregates))
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "rounds.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
