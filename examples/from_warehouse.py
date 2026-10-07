"""From a warehouse table to the next allocation, end to end.

Builds a small experiment in SQLite (a stand-in for your warehouse), runs
examples/sql/period_counts.sql on it, and hands the rows to orts.  The base rate
moves a lot from day to day while arm C is better than A and B throughout, which is
the situation OR-TS is for.

    python examples/from_warehouse.py
"""
import pathlib
import sqlite3
from datetime import datetime, timedelta

import numpy as np

from orts import allocate_from_rows

SQL = pathlib.Path(__file__).with_name("sql") / "period_counts.sql"
ARMS, CONTRAST = ["A", "B", "C"], [0.0, 0.05, 0.25]  # log-odds against arm A
DAYS, USERS_PER_DAY, SHIFT_SD = 14, 6000, 0.35


def build(db: sqlite3.Connection, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    db.execute("CREATE TABLE exposures (experiment_id, user_id, variation, exposed_at)")
    db.execute("CREATE TABLE conversions (user_id, converted_at)")
    start, uid = datetime(2026, 10, 1), 0
    for day in range(DAYS):
        base = np.log(0.04 / 0.96) + rng.normal(0, SHIFT_SD)  # the common shift of the day
        for arm, contrast in zip(ARMS, CONTRAST):
            n = USERS_PER_DAY // len(ARMS)
            p = 1 / (1 + np.exp(-(base + contrast)))
            for converted in rng.random(n) < p:
                uid += 1
                at = start + timedelta(days=day, seconds=int(rng.integers(0, 86400)))
                db.execute("INSERT INTO exposures VALUES ('exp1', ?, ?, ?)",
                           (uid, arm, at.strftime("%Y-%m-%d %H:%M:%S")))
                if converted:
                    later = at + timedelta(hours=int(rng.integers(1, 100)))
                    db.execute("INSERT INTO conversions VALUES (?, ?)",
                               (uid, later.strftime("%Y-%m-%d %H:%M:%S")))


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    build(db)
    as_of = (datetime(2026, 10, 1) + timedelta(days=DAYS + 7)).strftime("%Y-%m-%d %H:%M:%S")
    rows = db.execute(SQL.read_text(), {"experiment_id": "exp1", "as_of": as_of}).fetchall()
    print(f"{len(rows)} rows, one per (period, arm); the first three:")
    for r in rows[:3]:
        print("  ", dict(r))

    q = allocate_from_rows(rows, floor=0.01, seed=0)
    print("\nnext allocation:", {a: round(s, 3) for a, s in q.shares.items()})
    print("P(best):        ", {a: round(p, 3) for a, p in q.p_best.items()})
    print("leader:", q.leader)


if __name__ == "__main__":
    main()
