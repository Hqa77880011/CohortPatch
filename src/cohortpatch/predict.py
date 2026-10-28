"""Evaluate a saved activity model with its original training normalization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .activity import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint)
    with np.load(args.data, allow_pickle=False) as data:
        x = torch.as_tensor(data['x'], dtype=torch.float32)
    with torch.inference_mode():
        logits = model((x - checkpoint['mean']) / checkpoint['std'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(predictions=logits.argmax(-1).tolist(),
                                      logits=logits.tolist())), encoding='utf-8')


if __name__ == '__main__':
    main()
