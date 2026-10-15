"""Why the hybrid failed: GrowthBook's outcome-level tree against a tree that splits on the contrasts.

The design, hypotheses and failure criteria are in PREREG_contextual_uplift.md, written before
this script was run. In `ctxrobust.py` OR-TS refitted on GrowthBook's tree leaves was no better
than GrowthBook (H3 failed in all four L24 scenarios). GrowthBook's tree splits on how much a
segment converts; this script swaps in a tree that splits on whether the arms' log-odds contrasts
differ between the two sides (a likelihood-ratio test of the arm-by-side interaction), with the
same input, leaf cap and leaf minimum, and refits OR-TS on its leaves the same way. It also runs
the contextual model with one variance component per attribute on the same scenarios.

    python ctxuplift.py --driver /path/to/driver.cjs --reps 20 --json ctxuplift.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings

import numpy as np

import ctxrobust
from ctxrobust import (ARMS, DRAWS, FLOOR, K, L24, MAX_LEAVES, Engine, Layout, gb_contextual, hybrid,
                       orts_contextual, run)
from orts import ContextualLogisticBandit, allocate_cells_from_rows

warnings.filterwarnings("ignore")

MIN_USERS_PER_LEAF = 100  # GrowthBook's minUsersPerLeaf in ctxrobust.py


def _loglik(n, s, p):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return float(np.sum(s * np.log(p) + (n - s) * np.log1p(-p)))


def interaction_g2(n, s):
    """Likelihood-ratio statistic for arm-by-side interaction in a (sides, arms) table of counts.

    Saturated: every (side, arm) has its own rate. Null: logit p = alpha[side] + theta[arm].
    """
    seen = n > 0
    saturated = _loglik(n[seen], s[seen], s[seen] / n[seen])
    rows, cols = n.shape
    # null fit by Newton; a tiny ridge keeps a side or arm with no events finite
    w = np.zeros(rows + cols - 1)
    X = []
    for r in range(rows):
        for c in range(cols):
            x = np.zeros(rows + cols - 1)
            x[r] = 1.0
            if c < cols - 1:
                x[rows + c] = 1.0
            X.append(x)
    X, nn, ss = np.array(X), n.ravel(), s.ravel()
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (ss - nn * p) - 1e-8 * w
        hess = (X * (nn * p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    null = _loglik(nn[nn > 0], ss[nn > 0], (1 / (1 + np.exp(-(X @ w))))[nn > 0])
    return max(0.0, 2.0 * (saturated - null))


def contrast_tree(layout: Layout, n, s, max_leaves=MAX_LEAVES, min_users=MIN_USERS_PER_LEAF):
    """Leaf id per cell from pooled counts n, s of shape (cells, arms).

    Candidate splits of a leaf: for each attribute, every bipartition of its levels present in the
    leaf, each side holding at least ``min_users`` users. A split's score is the interaction G^2 of
    the leaf's (side, arm) table. Grown best-first, always splitting the leaf whose best split
    scores highest, until ``max_leaves`` leaves or no admissible split. Ties go to the earlier
    attribute and the earlier bipartition.
    """
    cells = np.array(layout.cells)

    def best_split(members):
        best = None
        for j in range(len(layout.names)):
            levels = sorted(set(cells[members, j]))
            if len(levels) < 2:
                continue
            first, rest = levels[0], levels[1:]
            # bipartitions with the first level on the left, so each appears once
            for r in range(len(rest)):
                for extra in itertools.combinations(rest, r):
                    left_levels = {first, *extra}
                    left = [c for c in members if cells[c, j] in left_levels]
                    right = [c for c in members if cells[c, j] not in left_levels]
                    nl, nr = n[left].sum(0), n[right].sum(0)
                    if nl.sum() < min_users or nr.sum() < min_users:
                        continue
                    g2 = interaction_g2(np.array([nl, nr]), np.array([s[left].sum(0), s[right].sum(0)]))
                    if best is None or g2 > best[0]:
                        best = (g2, left, right)
        return best

    leaves = [list(range(layout.C))]
    splits = [best_split(leaves[0])]
    while len(leaves) < max_leaves:
        candidates = [i for i, b in enumerate(splits) if b is not None]
        if not candidates:
            break
        i = max(candidates, key=lambda i: splits[i][0])
        _, left, right = splits[i]
        leaves[i:i + 1] = [left, right]
        splits[i:i + 1] = [best_split(left), best_split(right)]
    leaf = np.empty(layout.C, dtype=int)
    for i, members in enumerate(leaves):
        leaf[members] = i
    return leaf


def contrast_hybrid(layout):
    """The contrast tree's current leaves as cells; OR-TS refitted from the whole history every period."""
    history = []
    n, s = np.zeros((layout.C, K)), np.zeros((layout.C, K))

    def step(n_t, s_t, t):
        nonlocal n, s
        n, s = n + n_t, s + s_t
        history.append((n_t, s_t))
        leaf = contrast_tree(layout, n, s)
        step.leaves = len(set(leaf))
        rows = [{"period": p, "leaf": int(leaf[c]), "arm": a, "exposures": h_n[c, k], "events": h_s[c, k]}
                for p, (h_n, h_s) in enumerate(history) for c in range(layout.C)
                for k, a in enumerate(ARMS) if h_n[c, k] > 0]
        out = allocate_cells_from_rows(rows, cell="leaf", arms=ARMS, draw=DRAWS, floor=FLOOR, seed=1000 + t)
        return np.array([[out[int(leaf[c])].shares[a] for a in ARMS] if int(leaf[c]) in out
                         else np.full(K, 1 / K) for c in range(layout.C)])

    step.leaves = 1
    return step


def orts_attributes(layout):
    """Contextual OR-TS over the full cross with one variance component per attribute."""
    cells = [layout.cells[c] for c in range(layout.C)]
    bandit = ContextualLogisticBandit(ARMS, cells, attributes=layout.names)
    rng = np.random.default_rng(1)

    def step(n_t, s_t, t):
        bandit.update({cells[c]: {a: [n_t[c, i], s_t[c, i]] for i, a in enumerate(ARMS)}
                       for c in range(layout.C)})
        out = bandit.allocate(draw=DRAWS, floor=FLOOR, rng=rng)
        step.components = bandit.variance_components
        return np.array([[out[cells[c]].shares[a] for a in ARMS] for c in range(layout.C)])

    return step


REFERENCE = "Contextual OR-TS"
POLICIES = {
    REFERENCE: orts_contextual,
    "Contextual OR-TS, per-attribute components": orts_attributes,
    "GrowthBook contextual": gb_contextual,
    "Hybrid, GrowthBook leaves": hybrid,
    "Hybrid, contrast-tree leaves": contrast_hybrid,
}
SCENARIOS = [
    ("B", L24, "additive", 0.0), ("B", L24, "additive", 0.3),
    ("B", L24, "residual", 0.0), ("B", L24, "residual", 0.3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", help="write every run's numbers here")
    args = ap.parse_args()
    ctxrobust.DRIVER = args.driver
    record = {}
    for part, layout, structure, sigma in SCENARIOS:
        label = f"{part} {layout.C} cells, {structure}, sigma {sigma}"
        res = {k: np.array([run(f, layout, structure, sigma, s) for s in range(args.reps)])
               for k, f in POLICIES.items()}
        record[label] = {k: v.tolist() for k, v in res.items()}
        print(f"\n{label}: {args.reps} paired runs", flush=True)
        for k, r in res.items():
            line = f"  {k:44} regret {r[:, 0].mean():8.1f}"
            if not np.isnan(r[:, 1]).all():
                line += f"   leaves {r[:, 1].mean():.1f}"
            print(line, flush=True)
        pairs = [("GrowthBook contextual", "Hybrid, contrast-tree leaves"),
                 ("Hybrid, GrowthBook leaves", "Hybrid, contrast-tree leaves"),
                 ("Hybrid, contrast-tree leaves", REFERENCE),
                 (REFERENCE, "Contextual OR-TS, per-attribute components"),
                 ("GrowthBook contextual", "Hybrid, GrowthBook leaves")]
        for hi, lo in pairs:
            d = res[hi][:, 0] - res[lo][:, 0]
            se = d.std(ddof=1) / np.sqrt(args.reps)
            print(f"  {hi} minus {lo}: {d.mean():+8.1f} ± {se:6.1f} ({d.mean() / se:+.1f} SE), "
                  f"higher in {int((d > 0).sum())}/{args.reps}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(record, f)


if __name__ == "__main__":
    main()
