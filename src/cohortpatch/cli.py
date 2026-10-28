import argparse
import csv
import json
import random
import platform
import sys
from importlib.metadata import version
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from . import core


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected nonempty JSONL with unique case IDs')
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def summarize(rows):
    report = {}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row["metrics"].items()
            if value is None or isinstance(value, (int, float, bool))
        }
    )
    for key in keys:
        values = np.array(
            [float(row["metrics"][key]) for row in rows if row["metrics"].get(key) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if not len(values):
            report[key] = {"count": 0, "mean": None}
            continue
        report[key] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "p50": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }
    elapsed = sum(row["metrics"]["latency_ms"] for row in rows) / 1000
    legal = sum(row["metrics"].get("legal_effects", 0) for row in rows)
    report["legal_throughput_per_second"] = legal / elapsed if elapsed else None
    report.update(core.aggregate(rows))
    return report


def execute(cases, method, output):
    rows = []
    process = psutil.Process()
    for case in cases:
        start_cpu = time.process_time()
        start = time.perf_counter()
        result = core.run_case(case, method)
        elapsed = time.perf_counter() - start
        cpu = time.process_time() - start_cpu
        metrics = core.score(case, result)
        metrics.update(
            latency_ms=elapsed * 1000,
            cpu_seconds=cpu,
            rss_mib=process.memory_info().rss / 2**20,
            output_bytes=len(json.dumps(result).encode()),
        )
        rows.append(
            {
                "id": case["id"],
                "family": case.get("family", case["id"]),
                "method": method,
                "cluster": case.get("cluster", case["id"]),
                "seed": case.get("seed"),
                "group": case.get("group"),
                "result": result,
                "metrics": metrics,
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "records.jsonl", rows)
    (output / "metrics.json").write_text(
        json.dumps(summarize(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = dict(python=platform.python_version(), command=sys.argv,
                    dependencies={name: version(name) for name in ('numpy', 'scipy', 'matplotlib', 'psutil', 'scikit-learn')},
                    case_ids=[case['id'] for case in cases], source_type='finite-model execution')
    (output / 'run_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return rows


def plot(paths, output, metrics=None):
    records = [row for path in paths for row in read_jsonl(Path(path) / "records.jsonl")]
    methods = sorted({row["method"] for row in records})
    summaries = {method: summarize([row for row in records if row['method'] == method]) for method in methods}
    keys = metrics or sorted({key for summary in summaries.values() for key, value in summary.items() if isinstance(value, dict)})
    if any(not any(key in summary for summary in summaries.values()) for key in keys):
        raise ValueError('Unknown plot metric')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "n", "mean", "std", "p50", "p95", "p99"])
        for method in methods:
            summary = summaries[method]
            for key, value in summary.items():
                if isinstance(value, dict):
                    writer.writerow(
                        [
                            method,
                            key,
                            value["count"],
                            value["mean"],
                            value.get("std"),
                            value.get("p50"),
                            value.get("p95"),
                            value.get("p99"),
                        ]
                    )
    for key in keys:
        fig, ax = plt.subplots(figsize=(max(6, len(methods)), 4))
        means = [summaries[method].get(key, {}).get('mean') for method in methods]
        means = [value if value is not None else np.nan for value in means]
        ax.bar(methods, means, color="#527c91")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output / (key + ".png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=200)
    prepare.add_argument("--seed", type=int, default=17)
    for name in ("run", "suite"):
        item = sub.add_parser(name)
        item.add_argument("--data", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--config", help="JSON public-parameter overrides; keep input data unchanged")
        item.add_argument(
            "--methods",
            nargs="+",
            choices=core.METHODS,
            default=[core.METHODS[0]] if name == "run" else core.METHODS,
        )
    item = sub.add_parser("plot")
    item.add_argument("--inputs", nargs="+", required=True)
    item.add_argument("--metrics", nargs="+", help="Metric names; defaults to all available aggregates")
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 1:
            parser.error("--count must be positive")
        rows = list(core.generate(random.Random(args.seed), args.count))
        for row in rows:
            row['family'] = str(args.seed) + ':' + row.get('family', row['id'])
            row["id"] = str(args.seed) + ":" + row["id"]
        write_jsonl(args.output, rows)
    elif args.command in ("run", "suite"):
        cases = read_jsonl(args.data)
        if args.config:
            overrides = json.loads(Path(args.config).read_text(encoding='utf-8'))
            for case in cases:
                for key, value in overrides.items():
                    if isinstance(value, dict) and isinstance(case.get(key), dict):
                        case[key] = {**case[key], **value}
                    else:
                        case[key] = value
        for method in args.methods:
            execute(cases, method, Path(args.output) / method)
    else:
        plot(args.inputs, args.output, args.metrics)


if __name__ == "__main__":
    main()
