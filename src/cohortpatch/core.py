import numpy as np


METHODS = [
    "cohortpatch",
    "fixed_groups",
    "greedy_safe",
    "no_history",
    "no_complement",
    "no_fragmentation",
    "no_fairness",
]


def legal_size(m, v, k):
    result = [0]
    if v == m:
        result.append(m)
    if m >= 2 * k:
        result.extend(range(k, min(v, m - k) + 1))
    return sorted(set(result))


def largest(m, v, k):
    return max(legal_size(m, v, k), default=0)


def refine(classes, selected):
    result = []
    for group in classes:
        for child in (group & selected, group - selected):
            if child:
                result.append(child)
    return result


def choose(classes, usable, debt, forecasts, capacity, k, threshold, method):
    options = []
    for group in classes:
        available = sorted(group & usable, key=lambda i: (-debt[i], i))
        sizes = legal_size(len(group), len(available), k)
        if method == "fixed_groups":
            sizes = [0, len(group)] if group <= usable else [0]
        if method == "no_complement":
            sizes = [0] + list(range(k, len(available) + 1))
        candidates = []
        for size in sizes:
            subset = set(available[:size])
            fragmentation = 0.0
            if forecasts and method not in ("no_fragmentation", "greedy_safe"):
                for future in forecasts:
                    before = largest(len(group), len(group & future), k)
                    after = largest(len(subset), len(subset & future), k)
                    rest = group - subset
                    after += largest(len(rest), len(rest & future), k)
                    fragmentation += max(0, before - after) / len(forecasts)
            fairness = (
                0
                if method in ("no_fairness", "greedy_safe")
                else 0.25 * sum(debt[i] for i in subset)
            )
            candidates.append((subset, len(subset) + fairness - fragmentation))
        options.append(candidates)
    table = {0: (0.0, set())}
    for candidates in options:
        following = {}
        for used, (value, selected) in table.items():
            for subset, score in candidates:
                new_size = used + len(subset)
                if new_size <= capacity and (
                    new_size not in following or following[new_size][0] < value + score
                ):
                    following[new_size] = (value + score, selected | subset)
        table = following
    feasible = [
        (value, sorted(selected))
        for used, (value, selected) in table.items()
        if used == 0 or used >= threshold
    ]
    return set(max(feasible, default=(0.0, []), key=lambda item: (item[0], item[1]))[1])


def generate(rng, count):
    for index in range(count):
        clients = 12
        availability = [[i for i in range(clients) if rng.random() < 0.8] for _ in range(12)]
        forecasts = [
            [[i for i in range(clients) if rng.random() < 0.8] for _ in range(3)]
            for _ in availability
        ]
        yield {
            "id": str(index),
            "family": "independent-availability",
            "clients": clients,
            "k": 3,
            "threshold": 3,
            "capacity": 9,
            "availability": availability,
            "forecasts": forecasts,
        }


def run_case(case, method):
    n, k = case["clients"], case["k"]
    if n < k:
        raise ValueError("Enrolled population smaller than anonymity threshold")
    if method == "fixed_groups":
        classes = [set(range(i, min(i + k, n))) for i in range(0, n, k)]
        if any(len(group) < k for group in classes):
            remainder = classes.pop()
            classes[-1] |= remainder
    else:
        classes = [set(range(n))]
    debt = [0] * n
    rosters = []
    minimum = n
    violations = 0
    for t, indices in enumerate(case["availability"]):
        usable = set(indices)
        if method == "no_history":
            selected = set(sorted(usable)[: case["capacity"]])
        else:
            selected = choose(
                classes,
                usable,
                debt,
                [set(v) for v in case["forecasts"][t]],
                case["capacity"],
                k,
                case["threshold"],
                method,
            )
        if selected and len(selected) < case["threshold"]:
            selected = set()
        classes = refine(classes, selected)
        minimum = min(minimum, min(map(len, classes)))
        violations += sum(len(group) < k for group in classes)
        for client in usable:
            debt[client] += client not in selected
        rosters.append(sorted(selected))
    counts = [sum(i in roster for roster in rosters) for i in range(n)]
    weights = np.asarray(counts, dtype=float)
    jain = float(weights.sum() ** 2 / (n * (weights**2).sum())) if weights.any() else None
    return {
        "rosters": rosters,
        "minimum_class_size": minimum,
        "violations": violations,
        "final_classes": len(classes),
        "fairness": jain,
        "inclusion_counts": counts,
    }


def score(case, result):
    included = sum(map(len, result["rosters"]))
    possible = sum(map(len, case["availability"]))
    history = [tuple(i in roster for roster in result["rosters"]) for i in range(case["clients"])]
    sizes = [history.count(value) for value in history]
    attribution_bound = sum(1 / size for size in sizes) / len(sizes)
    return {
        "minimum_class_size": result["minimum_class_size"],
        "history_violations": result["violations"],
        "included_fraction": included / possible if possible else None,
        "partition_classes": result["final_classes"],
        "jain_fairness": result["fairness"],
        "history_only_identity_accuracy": attribution_bound,
        "successful_rounds": sum(bool(r) for r in result["rosters"]),
        "legal_effects": included if not result["violations"] else 0,
    }
