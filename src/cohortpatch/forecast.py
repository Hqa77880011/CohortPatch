"""Fit public Markov availability only on training traces and sample continuations."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train, cases = read(args.train), read(args.data)
    if (not train or not cases or args.horizon < 1 or args.samples < 1
            or {case['id'] for case in train} & {case['id'] for case in cases}):
        raise ValueError('Positive horizon/samples and independent nonempty train/data traces are required')
    clients = train[0]['clients']
    counts = np.ones((clients, 2, 2))  # Public Laplace smoothing, fixed before test access.
    for case in train:
        if case['clients'] != clients:
            raise ValueError('Forecast fitting requires a common enrolled population')
        states = np.array([[i in row for i in range(clients)] for row in case['availability']], dtype=int)
        for previous, following in zip(states, states[1:]):
            counts[np.arange(clients), previous, following] += 1
    probabilities = counts[:, :, 1] / counts.sum(axis=2)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open('w', encoding='utf-8') as stream:
        for case in cases:
            if case['clients'] != clients:
                raise ValueError('Forecast target population differs from training')
            forecasts = []
            for usable in case['availability']:
                samples = []
                for _ in range(args.samples):
                    current = np.array([i in usable for i in range(clients)], dtype=int)
                    for _ in range(args.horizon):
                        current = (rng.random(clients) < probabilities[np.arange(clients), current]).astype(int)
                        samples.append(np.flatnonzero(current).tolist())
                forecasts.append(samples)
            stream.write(json.dumps({**case, 'forecasts': forecasts, 'forecast_horizon': args.horizon,
                                     'forecast_seed': args.seed}) + '\n')
    output.with_suffix('.model.json').write_text(json.dumps(dict(probability_on=probabilities.tolist(),
                     training_case_ids=[case['id'] for case in train], seed=args.seed)), encoding='utf-8')


if __name__ == '__main__':
    main()
