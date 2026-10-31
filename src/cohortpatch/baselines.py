"""Batch-partitioning Algorithms 1--2 of So et al. (AAAI 2023).

The fixed-K algorithm and the manuscript's variable-capacity MR+ extension
are separate policies. Neither policy refines or merges its fixed batches.
"""

import itertools


def batches(clients, size, strict=False):
    """Create the public fixed partition; enlarge the MR+ remainder batch."""
    if size < 1 or clients < size or (strict and clients % size):
        raise ValueError("Original MR requires N/T to be integral and N >= T")
    groups = [set(range(start, min(start + size, clients)))
              for start in range(0, clients, size)]
    if len(groups[-1]) < size:
        remainder = groups.pop()
        groups[-1] |= remainder
    return groups


def batch_select(groups, usable, counts, capacity, threshold, rng,
                 *, fixed_k=False, heterogeneous=True):
    """Uniformly select a legal batch union with the published fairness rule."""
    if fixed_k and (len({len(group) for group in groups}) != 1
                    or capacity % len(groups[0])):
        raise ValueError("Original MR requires K/T to be integral")
    available = [group for group in groups if group <= usable]
    candidates = []
    for count in range(1, len(available) + 1):
        for subset in itertools.combinations(available, count):
            roster = set().union(*subset)
            if threshold <= len(roster) <= capacity and (not fixed_k or len(roster) == capacity):
                candidates.append(roster)
    if not candidates:
        return set()
    if heterogeneous:
        target = min(usable, key=lambda client: (counts[client], client))
        eligible = [roster for roster in candidates if target in roster]
        if not eligible and fixed_k:
            # Algorithm 2's least-available-user restriction is infeasible.
            return set()
        if not eligible:
            legal_members = set().union(*candidates)
            target = min(legal_members, key=lambda client: (counts[client], client))
            eligible = [roster for roster in candidates if target in roster]
        candidates = eligible
    if not fixed_k:
        largest = max(map(len, candidates))
        candidates = [roster for roster in candidates if len(roster) == largest]
    return set(candidates[int(rng.integers(len(candidates)))])


def exact_rosters(classes, usable, capacity, threshold, k):
    """Enumerate the actual two-sided safe family, without prefix restriction."""
    names = sorted(usable)
    for count in range(threshold, min(capacity, len(names)) + 1):
        for subset in itertools.combinations(names, count):
            selected = set(subset)
            if all((not group & selected or len(group & selected) >= k)
                   and (not group - selected or len(group - selected) >= k)
                   for group in classes):
                yield selected


def clairvoyant_path(classes, availability, capacity, threshold, k):
    """Exact finite-horizon inclusion oracle; explicitly observes future usability."""
    from functools import lru_cache

    def encode(partition):
        return tuple(sorted(tuple(sorted(group)) for group in partition))

    @lru_cache(None)
    def solve(t, partition):
        if t == len(availability):
            return 0, ()
        groups = [set(group) for group in partition]
        best = None
        for selected in itertools.chain([set()], exact_rosters(
                groups, set(availability[t]), capacity, threshold, k)):
            following = [child for group in groups for child in
                         (group & selected, group - selected) if child]
            score, suffix = solve(t + 1, encode(following))
            candidate = (len(selected) + score, (tuple(sorted(selected)),) + suffix)
            if best is None or candidate > best:
                best = candidate
        return best

    return [list(roster) for roster in solve(0, encode(classes))[1]]
