"""Warehouse rows to allocations: the helpers must equal the manual fold, and the SQL must count right."""

import pathlib
import random
import runpy
import sqlite3

import numpy as np
import pytest

from orts import (ContextualLogisticBandit, LogisticBandit, allocate_cells_from_rows,
                  allocate_from_rows, batches_from_rows, replay)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SQL = (ROOT / "examples" / "sql" / "period_counts.sql").read_text()

ROWS = [
    {"period": "2026-10-02", "arm": "A", "exposures": 12000, "events": 150},
    {"period": "2026-10-01", "arm": "A", "exposures": 30000, "events": 300},
    {"period": "2026-10-01", "arm": "B", "exposures": 30000, "events": 330},
    {"period": "2026-10-02", "arm": "B", "exposures": 18000, "events": 240},
    {"period": "2026-10-02", "arm": "C", "exposures": 6000, "events": 70},
    {"period": "2026-10-01", "arm": "C", "exposures": 30000, "events": 290},
]
MANUAL = [{"A": [30000, 300], "B": [30000, 330], "C": [30000, 290]},
          {"A": [12000, 150], "B": [18000, 240], "C": [6000, 70]}]


def shares(allocation):
    return {a: round(s, 12) for a, s in allocation.shares.items()}


def test_batches_are_in_period_order_and_duplicates_are_summed():
    rows = ROWS + [{"period": "2026-10-01", "arm": "A", "exposures": 10, "events": 1}]
    batches = batches_from_rows(rows)
    assert batches[0] == {"A": [30010, 301], "B": [30000, 330], "C": [30000, 290]}
    assert batches[1] == MANUAL[1]


def test_allocation_equals_the_manual_fold():
    manual = LogisticBandit()
    for batch in MANUAL:
        manual.update(batch)
    expected = manual.allocate(["A", "B", "C"], draw=20000, floor=0.01, rng=np.random.default_rng(7))
    got = allocate_from_rows(ROWS, draw=20000, floor=0.01, seed=7)
    assert shares(got) == shares(expected)


def test_row_order_does_not_matter():
    shuffled = ROWS[:]
    random.Random(3).shuffle(shuffled)
    assert shares(allocate_from_rows(shuffled, draw=5000, seed=1)) == \
        shares(allocate_from_rows(ROWS, draw=5000, seed=1))


def test_cell_row_order_does_not_matter():
    shuffled = CELL_ROWS[:]
    random.Random(5).shuffle(shuffled)
    one = allocate_cells_from_rows(shuffled, draw=3000, seed=1)
    two = allocate_cells_from_rows(CELL_ROWS, draw=3000, seed=1)
    assert {c: shares(a) for c, a in one.items()} == {c: shares(a) for c, a in two.items()}


def test_seed_makes_a_run_reproducible():
    one = allocate_from_rows(ROWS, draw=3000, seed=11)
    two = allocate_from_rows(ROWS, draw=3000, seed=11)
    assert shares(one) == shares(two)


def test_other_table_types_work():
    class Frame:  # the one method a pandas DataFrame is read through
        def to_dict(self, orient):
            assert orient == "records"
            return ROWS

    class Polars:
        def to_dicts(self):
            return ROWS

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE t (period, arm, exposures, events)")
    db.executemany("INSERT INTO t VALUES (:period, :arm, :exposures, :events)", ROWS)
    sqlite_rows = db.execute("SELECT * FROM t").fetchall()
    reference = shares(allocate_from_rows(ROWS, draw=4000, seed=2))
    for table in (Frame(), Polars(), sqlite_rows, iter(ROWS)):
        assert shares(allocate_from_rows(table, draw=4000, seed=2)) == reference


def test_column_names_and_extra_columns():
    renamed = [{"day": r["period"], "variation": r["arm"], "users": r["exposures"],
                "conversions": r["events"], "note": "ignored"} for r in ROWS]
    got = allocate_from_rows(renamed, period="day", arm="variation", exposures="users",
                             events="conversions", draw=4000, seed=2)
    assert shares(got) == shares(allocate_from_rows(ROWS, draw=4000, seed=2))


def test_numeric_and_decimal_periods_order_numerically():
    from decimal import Decimal
    rows = [{"period": p, "arm": a, "exposures": Decimal(n), "events": Decimal(s)}
            for p, batch in zip((10, 9, 2), (MANUAL[1], MANUAL[0], MANUAL[0])) for a, (n, s) in batch.items()]
    assert [b["A"][0] for b in batches_from_rows(rows)] == [30000, 30000, 12000]


def test_arms_default_to_the_ones_seen_and_can_be_restricted():
    full = allocate_from_rows(ROWS, draw=4000, seed=0)
    assert set(full.shares) == {"A", "B", "C"}
    live = allocate_from_rows(ROWS, arms=["B", "C"], draw=4000, seed=0)
    assert set(live.shares) == {"B", "C"} and sum(live.shares.values()) == pytest.approx(1.0)


def test_decay_is_passed_through():
    manual = replay(MANUAL, decay=0.5)
    expected = manual.allocate(["A", "B", "C"], draw=4000, rng=np.random.default_rng(5))
    assert shares(allocate_from_rows(ROWS, decay=0.5, draw=4000, seed=5)) == shares(expected)
    assert shares(allocate_from_rows(ROWS, decay=0.5, draw=4000, seed=5)) != \
        shares(allocate_from_rows(ROWS, draw=4000, seed=5))


def test_history_with_nothing_to_learn_from_starts_uniform():
    rows = [{"period": 1, "arm": a, "exposures": 100, "events": 0} for a in "ABC"]
    assert shares(allocate_from_rows(rows)) == {a: pytest.approx(1 / 3) for a in "ABC"}


@pytest.mark.parametrize("mutate, message", [
    (lambda r: [], "empty"),
    (lambda r: [{k: v for k, v in x.items() if k != "events"} for x in r], "no column 'events'"),
    (lambda r: [{**r[0], "events": r[0]["exposures"] + 1}] + r[1:], "exceed"),
    (lambda r: [{**r[0], "exposures": -5}] + r[1:], "non-negative"),
    (lambda r: [{**r[0], "events": float("nan")}] + r[1:], "non-negative"),
    (lambda r: [{**r[0], "events": None}] + r[1:], "must be a number"),
    (lambda r: [{**r[0], "period": None}] + r[1:], "cannot be ordered"),
])
def test_bad_tables_are_rejected_with_a_reason(mutate, message):
    with pytest.raises(ValueError, match=message):
        allocate_from_rows(mutate([dict(r) for r in ROWS]))


CELL_ROWS = [{**r, "cell": c, "exposures": r["exposures"] // 2, "events": r["events"] // 2}
             for r in ROWS for c in ("mobile", "desktop")]


def test_cells_equal_the_manual_contextual_fold():
    bandit = ContextualLogisticBandit(["A", "B", "C"], ["desktop", "mobile"])  # name order, as the helper uses
    for batch in batches_from_rows(CELL_ROWS, cell="cell"):
        bandit.update(batch)
    expected = bandit.allocate(draw=3000, floor=0.01, rng=np.random.default_rng(4))
    got = allocate_cells_from_rows(CELL_ROWS, draw=3000, floor=0.01, seed=4)
    assert set(got) == {"mobile", "desktop"}
    for c in got:
        assert shares(got[c]) == shares(expected[c])


def conn():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE exposures (experiment_id, user_id, variation, exposed_at)")
    db.execute("CREATE TABLE conversions (user_id, converted_at)")
    return db


def test_sql_counts_each_user_once_inside_a_closed_window():
    db = conn()
    exposures = [
        ("e", 1, "A", "2026-10-01 09:00:00"),   # converts inside the window
        ("e", 1, "A", "2026-10-02 09:00:00"),   # a later exposure of the same user: ignored
        ("e", 2, "A", "2026-10-01 10:00:00"),   # converts twice: still one event
        ("e", 3, "A", "2026-10-01 11:00:00"),   # converted before being exposed: not an event
        ("e", 4, "A", "2026-10-01 12:00:00"),   # converts after the 7-day window: not an event
        ("e", 5, "B", "2026-10-01 13:00:00"),   # seen in two variations: dropped
        ("e", 5, "A", "2026-10-01 14:00:00"),
        ("e", 6, "B", "2026-10-01 15:00:00"),   # never converts
        ("e", 7, "B", "2026-10-03 09:00:00"),   # window still open at :as_of: left out
        ("other", 8, "A", "2026-10-01 09:00:00"),  # another experiment
        ("e", 9, "B", "2026-10-02 09:00:00"),   # converts on the day the window opens
    ]
    db.executemany("INSERT INTO exposures VALUES (?, ?, ?, ?)", exposures)
    db.executemany("INSERT INTO conversions VALUES (?, ?)", [
        (1, "2026-10-01 20:00:00"), (2, "2026-10-01 10:30:00"), (2, "2026-10-04 10:30:00"),
        (3, "2026-09-30 11:00:00"), (4, "2026-10-08 12:00:01"), (7, "2026-10-04 09:00:00"),
        (9, "2026-10-02 09:00:00"), (8, "2026-10-01 10:00:00"),
    ])
    rows = db.execute(SQL, {"experiment_id": "e", "as_of": "2026-10-09 10:00:00"}).fetchall()
    assert [tuple(r) for r in rows] == [
        ("2026-10-01", "A", 4, 2),
        ("2026-10-01", "B", 1, 0),
        ("2026-10-02", "B", 1, 1),
    ]


def test_the_warehouse_example_runs_and_finds_the_best_arm(capsys):
    runpy.run_path(str(ROOT / "examples" / "from_warehouse.py"), run_name="__main__")
    assert "leader: C" in capsys.readouterr().out
