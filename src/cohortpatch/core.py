import numpy as np


METHODS = [
    "cohortpatch",
    "fixed_groups",
    "greedy_safe",
    "no_history",
    "no_complement",
    "no_fragmentation",
    "no_fairness",
    "mr_fixed_k",
    "mr_plus",
    "no_refinement",
    "class_count",
    "no_retry",
    "future_oracle",
]


def legal_size(m, v, k):
    """Return sizes preserving both nonempty historical children at least k."""
    result = [0]
    if v == m and m >= k:
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


def choose(classes, usable, debt, forecasts, capacity, k, threshold, method,
           fairness_weight=0.1, fragmentation_weight=1.0, retry_costs=None, retry_weight=1.0,
           forecast_horizon=1):
    options = []
    for group in classes:
        available = sorted(group & usable, key=lambda i: (
            0 if method in ("no_fairness", "greedy_safe") else -debt[i],
            -sum(i in future for future in forecasts), i))
        sizes = legal_size(len(group), len(available), k)
        if method in ("fixed_groups", "no_refinement"):
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
                    fragmentation += forecast_horizon * max(0, before - after) / len(forecasts)
            if method == "class_count":
                fragmentation = float(bool(subset) and bool(group - subset))
            fairness = (
                0
                if method in ("no_fairness", "greedy_safe")
                else fairness_weight * sum(debt[i] for i in subset)
            )
            retry = sum((retry_costs or {}).get(str(i), 0) for i in subset)
            candidates.append((subset, len(subset) + fairness - fragmentation_weight * fragmentation
                               - (0 if method == "no_retry" else retry_weight) * retry))
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
    if min(n, k, case["threshold"], case["capacity"]) < 1 or n < k:
        raise ValueError("Positive policy values and enrolled population >= k are required")
    if case["threshold"] > case["capacity"]:
        raise ValueError("Cryptographic threshold exceeds capacity")
    enrolled = set(range(n))
    if len(case["forecasts"]) != len(case["availability"]):
        raise ValueError("Forecasts must align with availability attempts")
    for indices in case["availability"]:
        if len(set(indices)) != len(indices) or not set(indices) <= enrolled:
            raise ValueError("Availability must contain unique enrolled client IDs")
    if any(len(set(row)) != len(row) or not set(row) <= enrolled
           for forecast in case["forecasts"] for row in forecast):
        raise ValueError("Forecasts must contain unique enrolled client IDs")
    if "initial_classes" in case:
        classes = [set(group) for group in case["initial_classes"]]
        if (sum(map(len, classes)) != n or set().union(*classes) != set(range(n))
                or any(len(group) < k for group in classes)):
            raise ValueError("Initial classes must partition enrolled clients into k-safe histories")
    elif method == "fixed_groups" and not case.get("warm_start_rosters"):
        classes = [set(range(i, min(i + k, n))) for i in range(0, n, k)]
        if any(len(group) < k for group in classes):
            remainder = classes.pop()
            classes[-1] |= remainder
    else:
        classes = [set(range(n))]
    initial_partition = [sorted(group) for group in classes]
    history_classes = [set(group) for group in case.get("initial_classes", [list(range(n))])]
    debt = [0] * n
    rosters = []
    warm = case.get("warm_start_rosters", [])
    for roster in warm:
        selected = set(roster)
        if (not selected <= set(range(n)) or len(selected) != len(roster)
                or not case["threshold"] <= len(selected) <= case["capacity"]):
            raise ValueError("Warm-start release violates enrolled roster or resource bounds")
        classes = refine(classes, selected)
        history_classes = refine(history_classes, selected)
        if any(len(group) < k for group in classes):
            raise ValueError("Warm-start release violates historical safety")
        rosters.append(sorted(selected))
        for client in range(n):
            debt[client] += client not in selected
    minimum = min(map(len, history_classes))
    violations = 0
    from .baselines import batches, batch_select, clairvoyant_path
    batch_groups = batches(n, case.get("batch_size", k), method == "mr_fixed_k") if method in ("mr_fixed_k", "mr_plus") else None
    if batch_groups and any(not any(group <= history for history in history_classes)
                            for group in batch_groups):
        raise ValueError("Fixed MR batches must respect retained warm-start histories")
    rng = np.random.default_rng(case.get("seed", 17))
    if method == "future_oracle" and (n > 12 or len(case["availability"]) > 6):
        raise ValueError("Exact oracle domain is N <= 12 and horizon <= 6")
    oracle = clairvoyant_path(classes, case["availability"], case["capacity"],
                             case["threshold"], k) if method == "future_oracle" else None
    for t, indices in enumerate(case["availability"]):
        usable = set(indices)
        if oracle is not None:
            selected = set(oracle[t])
        elif batch_groups is not None:
            counts = [sum(i in roster for roster in rosters) for i in range(n)]
            selected = batch_select(batch_groups, usable, counts, case["capacity"],
                                    case["threshold"], rng, fixed_k=method == "mr_fixed_k",
                                    heterogeneous=case.get("heterogeneous_availability", True))
        elif method == "no_history":
            # Keep the same public debt/retry objective after removing history.
            selected=choose([enrolled],usable,debt,[],case['capacity'],1,case['threshold'],'no_complement',
                            fairness_weight=case.get('fairness_weight',.1),fragmentation_weight=0,
                            retry_costs=case.get('retry_costs'),retry_weight=case.get('retry_weight',1))
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
                case.get("fairness_weight", 0.1),
                case.get("fragmentation_weight", 1.0),
                case.get("retry_costs"),
                case.get("retry_weight", 1.0),
                case.get("forecast_horizon", 1),
            )
        if selected and len(selected) < case["threshold"]:
            selected = set()
        classes = refine(classes, selected)
        history_classes = refine(history_classes, selected)
        minimum = min(minimum, min(map(len, history_classes)))
        violations += sum(len(group) < k for group in history_classes)
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
        "final_classes": len(history_classes),
        "protected_partition_classes": len(classes),
        "fairness": jain,
        "inclusion_counts": counts,
        "included": sum(counts),
        "usable": sum(map(len, case["availability"])) + n * len(warm),
        "attempted_rounds": len(rosters),
        "policy": {name: case[name] for name in ("clients", "k", "capacity", "threshold")},
        "initial_classes": initial_partition,
        "uploaded": [list(range(n)) for _ in warm] + case["availability"],
        "warm_start_rounds": len(warm),
    }


def score(case, result):
    included = sum(map(len, result["rosters"]))
    possible = result["usable"]
    initial = case.get("initial_classes", [list(range(case["clients"]))])
    history = [(next(j for j, group in enumerate(initial) if i in group),)
               + tuple(i in roster for roster in result["rosters"]) for i in range(case["clients"])]
    sizes = [history.count(value) for value in history]
    attribution_bound = sum(1 / size for size in sizes) / len(sizes)
    opportunities = np.array([sum(i in row for row in result["uploaded"])
                              for i in range(case["clients"])])
    inclusion = np.asarray(result["inclusion_counts"])[opportunities > 0] / opportunities[opportunities > 0]
    return {
        "minimum_class_size": result["minimum_class_size"],
        "history_violations": result["violations"],
        "included_fraction": included / possible if possible else None,
        "partition_classes": result["final_classes"],
        "jain_fairness": result["fairness"],
        "p10_inclusion_fraction": float(np.quantile(inclusion, 0.1)) if len(inclusion) else None,
        "p10_inclusion_count": float(np.quantile(result["inclusion_counts"], 0.1)),
        "history_only_identity_accuracy": attribution_bound,
        "successful_rounds": sum(bool(r) for r in result["rosters"]),
        "legal_effects": included if not result["violations"] else 0,
    }


def aggregate(rows):
    """Use common original usable client-rounds as the inclusion denominator."""
    possible = sum(row["result"]["usable"] for row in rows)
    included = sum(row["result"]["included"] for row in rows)
    return {"included_fraction": dict(count=possible, mean=included/possible if possible else None),
            "minimum_class_size": dict(count=len(rows), mean=min(row["result"]["minimum_class_size"] for row in rows))}
