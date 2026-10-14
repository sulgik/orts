"""Contextual robustness: a non-additive environment, and GrowthBook's tree leaves as OR-TS cells.

The design, hypotheses and failure criteria are in PREREG_contextual_robustness.md, written before
this script was run. In short: `ctxbench.py` compared contextual OR-TS with GrowthBook's contextual
engine on six cells whose contrasts were additive in two attributes. This script adds cells whose
contrasts do not decompose by attribute ("residual", "checkerboard"), a 24-cell layout where the
full cross is thin, and a hybrid that refits OR-TS each period with GrowthBook's current tree
leaves as its cells.

GrowthBook's engine is run, not reimplemented: `computeContextualBanditWeights` bundled from
`packages/stats-ts` and driven over stdin by `gb_contextual_driver.ts`, as in `ctxbench.py`.

    python ctxrobust.py --driver /path/to/driver.cjs --reps 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np

from orts import ContextualLogisticBandit, LogisticBandit, allocate_cells_from_rows

warnings.filterwarnings("ignore")

MAIN = np.array([0.0, 0.05, 0.10, 0.15])  # log-odds against arm 0; arm 3 is best on average
K = len(MAIN)
ARMS = [f"a{i}" for i in range(K)]
BASE = np.log(0.03 / 0.97)
BASE_SD = 0.3  # spread of the cells' base log-odds
H = 0.2  # sd of an attribute's effect on the contrasts
N, T = 20_000, 40
FLOOR = 0.01
MAX_LEAVES = 8
DRAWS = 20_000
DRIVER = ""


@dataclass
class Layout:
    names: List[str]
    levels: List[List[str]]
    shares: List[np.ndarray]

    def __post_init__(self):
        self.cells = list(itertools.product(*[range(len(l)) for l in self.levels]))
        self.C = len(self.cells)
        self.share = np.array([np.prod([s[i] for s, i in zip(self.shares, c)]) for c in self.cells])

    def context(self, c):
        return {n: self.levels[j][i] for j, (n, i) in enumerate(zip(self.names, self.cells[c]))}

    def index(self, context):
        return self.cells.index(tuple(self.levels[j].index(context[n]) for j, n in enumerate(self.names)))


L6 = Layout(["region", "device"], [["r0", "r1", "r2"], ["d0", "d1"]],
            [np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.3])])
L24 = Layout(["region", "device", "plan"], [["r0", "r1", "r2", "r3"], ["d0", "d1"], ["p0", "p1", "p2"]],
             [np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.7, 0.3]), np.array([0.6, 0.3, 0.1])])


def contrasts(layout: Layout, structure: str, env: np.random.Generator) -> np.ndarray:
    """Each cell's log-odds contrasts, shape (cells, arms)."""
    m = len(layout.names)
    if structure == "additive":
        effects = [env.normal(0, H, (len(l), K)) for l in layout.levels]
        return np.array([MAIN + sum(e[i] for e, i in zip(effects, c)) for c in layout.cells])
    if structure == "residual":
        return MAIN + env.normal(0, H * np.sqrt(m), (layout.C, K))
    if structure == "checkerboard":
        s = np.array([1.0 if sum(c) % 2 == 0 else -1.0 for c in layout.cells])
        out = np.tile(MAIN, (layout.C, 1))
        out[:, 0] += H * s
        out[:, 3] -= H * s
        return out
    raise ValueError(structure)


class Engine:
    """GrowthBook's contextual engine, one node process per run."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.proc = subprocess.Popen(["node", DRIVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def __call__(self, n, s):
        """Weights and leaf id per cell, from the run's pooled counts n, s of shape (cells, arms)."""
        lay = self.layout
        observations = [
            {"variationIndex": k, "context": lay.context(c),
             "arm": {"n": n[c, k], "main_sum": s[c, k], "main_sum_squares": s[c, k],
                     "denominator_sum": 0, "denominator_sum_squares": 0,
                     "main_denominator_sum_product": 0, "covariate_sum": 0,
                     "covariate_sum_squares": 0, "main_covariate_sum_product": 0}}
            for c in range(lay.C) for k in range(K) if n[c, k] > 0]
        self.proc.stdin.write(json.dumps({
            "varIds": ARMS, "attributes": lay.names, "maxLeaves": MAX_LEAVES, "minUsersPerLeaf": 100,
            "metricSettings": {"id": "m", "name": "m", "inverse": False, "statistic_type": "mean",
                               "main_metric_type": "binomial"},
            "analysisWeights": [1 / K] * K, "observations": observations}) + "\n")
        self.proc.stdin.flush()
        weights, leaf = np.full((lay.C, K), 1 / K), np.arange(lay.C) + 1000  # unseen cells: own leaf
        for r in json.loads(self.proc.stdout.readline())["responses"]:
            c = lay.index(r["context"])
            weights[c], leaf[c] = r["weights"], r["leafId"]
        return weights, leaf

    def close(self):
        self.proc.kill()


def gb_contextual(layout):
    engine, n, s = Engine(layout), np.zeros((layout.C, K)), np.zeros((layout.C, K))

    def step(n_t, s_t, t):
        nonlocal n, s
        n, s = n + n_t, s + s_t
        weights, leaf = engine(n, s)
        step.leaves = len(set(leaf))
        return weights

    step.close, step.leaves = engine.close, 1
    return step


def hybrid(layout):
    """GrowthBook's current leaves as cells; OR-TS refitted from the whole history every period."""
    engine, history = Engine(layout), []
    n, s = np.zeros((layout.C, K)), np.zeros((layout.C, K))

    def step(n_t, s_t, t):
        nonlocal n, s
        n, s = n + n_t, s + s_t
        history.append((n_t, s_t))
        _, leaf = engine(n, s)
        step.leaves = len(set(leaf))
        rows = [{"period": p, "leaf": int(leaf[c]), "arm": a, "exposures": h_n[c, k], "events": h_s[c, k]}
                for p, (h_n, h_s) in enumerate(history) for c in range(layout.C)
                for k, a in enumerate(ARMS) if h_n[c, k] > 0]
        out = allocate_cells_from_rows(rows, cell="leaf", arms=ARMS, draw=DRAWS, floor=FLOOR, seed=1000 + t)
        return np.array([[out[int(leaf[c])].shares[a] for a in ARMS] if int(leaf[c]) in out
                         else np.full(K, 1 / K) for c in range(layout.C)])

    step.close, step.leaves = engine.close, 1
    return step


def orts_contextual(layout):
    bandit, rng = ContextualLogisticBandit(ARMS, list(range(layout.C))), np.random.default_rng(1)

    def step(n_t, s_t, t):
        bandit.update({c: {a: [n_t[c, i], s_t[c, i]] for i, a in enumerate(ARMS)} for c in range(layout.C)})
        out = bandit.allocate(draw=DRAWS, floor=FLOOR, rng=rng)
        return np.array([[out[c].shares[a] for a in ARMS] for c in range(layout.C)])

    return step


def orts_per_cell(layout):
    bandits, rng = [LogisticBandit() for _ in range(layout.C)], np.random.default_rng(1)

    def step(n_t, s_t, t):
        weights = np.full((layout.C, K), 1 / K)
        for c, b in enumerate(bandits):
            obs = {a: [n_t[c, i], s_t[c, i]] for i, a in enumerate(ARMS) if n_t[c, i] > 0}
            if len(obs) > 1 and 0 < s_t[c].sum() < n_t[c].sum():
                b.update(obs)
            if b.fitted:
                sh = b.allocate(ARMS, draw=DRAWS, floor=FLOOR, rng=rng).shares
                weights[c] = [sh[a] for a in ARMS]
        return weights

    return step


def orts_pooled(layout):
    bandit, rng = LogisticBandit(), np.random.default_rng(1)

    def step(n_t, s_t, t):
        nk, sk = n_t.sum(0), s_t.sum(0)
        bandit.update({a: [nk[i], sk[i]] for i, a in enumerate(ARMS) if nk[i] > 0})
        sh = bandit.allocate(ARMS, draw=DRAWS, floor=FLOOR, rng=rng).shares
        return np.tile([sh[a] for a in ARMS], (layout.C, 1))

    return step


REFERENCE = "Contextual OR-TS"
POLICIES = {
    REFERENCE: orts_contextual,
    "GrowthBook contextual": gb_contextual,
    "Hybrid (GrowthBook leaves as OR-TS cells)": hybrid,
    "OR-TS, one per cell": orts_per_cell,
    "OR-TS, context ignored": orts_pooled,
}
SCENARIOS = [
    ("A", L6, "additive", 0.0), ("A", L6, "additive", 0.3),
    ("A", L6, "residual", 0.0), ("A", L6, "residual", 0.3),
    ("A", L6, "checkerboard", 0.0), ("A", L6, "checkerboard", 0.3),
    ("B", L24, "additive", 0.0), ("B", L24, "additive", 0.3),
    ("B", L24, "residual", 0.0), ("B", L24, "residual", 0.3),
]


def run(make, layout, structure, sigma, seed):
    env = np.random.default_rng(seed)
    contrast = contrasts(layout, structure, env)
    base = BASE + env.normal(0, BASE_SD, layout.C)
    step, weights = make(layout), np.full((layout.C, K), 1 / K)
    regret = 0.0
    for t in range(T):
        p = 1 / (1 + np.exp(-(base[:, None] + env.normal(0, sigma) + contrast)))
        users = env.multinomial(N, layout.share)
        n_t = np.array([env.multinomial(users[c], weights[c]) for c in range(layout.C)])
        s_t = env.binomial(n_t, p)
        regret += float((n_t * (p.max(1, keepdims=True) - p)).sum())
        weights = step(n_t.astype(float), s_t.astype(float), t)
    leaves = getattr(step, "leaves", np.nan)
    if hasattr(step, "close"):
        step.close()
    return regret, leaves


def main():
    global DRIVER
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", help="write every run's numbers here")
    args = ap.parse_args()
    DRIVER = args.driver
    record = {}
    for part, layout, structure, sigma in SCENARIOS:
        label = f"{part} {layout.C} cells, {structure}, sigma {sigma}"
        res = {k: np.array([run(f, layout, structure, sigma, s) for s in range(args.reps)])
               for k, f in POLICIES.items()}
        record[label] = {k: v.tolist() for k, v in res.items()}
        ref = res[REFERENCE][:, 0]
        print(f"\n{label}: {args.reps} paired runs", flush=True)
        for k, r in res.items():
            line = f"  {k:44} regret {r[:, 0].mean():8.1f}"
            if k != REFERENCE:
                d = r[:, 0] - ref
                se = d.std(ddof=1) / np.sqrt(args.reps)
                line += f"   minus {REFERENCE} {d.mean():+8.1f} ± {se:6.1f} ({d.mean() / se:+.1f} SE), higher in {int((d > 0).sum())}/{args.reps}"
            if not np.isnan(r[:, 1]).all():
                line += f"   leaves {r[:, 1].mean():.1f}"
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(record, f)


if __name__ == "__main__":
    main()
