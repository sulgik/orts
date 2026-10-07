"""From a warehouse table to the next allocation, with no state kept between runs.

OR-TS is a fold over batches, but a scheduled job usually has no memory: each run
reads the experiment's history from a table and has to answer "what share should
each arm get now?".  This module does exactly that.  The input is one row per
(period, arm), optionally per cell, with the number of users exposed and the
number who converted; the output is the same ``Allocation`` that
``LogisticBandit.allocate`` returns.

    rows = [
        {"period": "2026-10-01", "arm": "A", "exposures": 30000, "events": 300},
        {"period": "2026-10-01", "arm": "B", "exposures": 30000, "events": 330},
        {"period": "2026-10-02", "arm": "A", "exposures": 12000, "events": 150},
        ...
    ]
    q = allocate_from_rows(rows, floor=0.01, seed=0)
    q.shares          # {'A': 0.2, 'B': 0.8}

The result depends only on the rows: the periods are folded in sorted order, so
the same table gives the same state however the rows arrive.  The state is a
fold over the batch history, so counts must be *per period*, not cumulative, and
every period that has been served belongs in the table.  ``examples/sql/``
has the query that produces such rows.

Rows may be dicts, ``sqlite3.Row`` objects, or anything with ``to_dict("records")``
(a pandas DataFrame) or ``to_dicts()`` (a polars DataFrame).
"""

import math
from collections.abc import Mapping
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence

import numpy as np

from .contextual import ContextualLogisticBandit
from .logisticbandit import Allocation, LogisticBandit

Batch = Dict[str, List[float]]
CellBatch = Dict[Hashable, Batch]


def _records(rows: Any) -> List[Any]:
    """The rows as a list, whatever table type they came in."""
    if not isinstance(rows, Mapping):
        if hasattr(rows, "to_dict"):
            return list(rows.to_dict("records"))  # pandas
        if hasattr(rows, "to_dicts"):
            return list(rows.to_dicts())  # polars
    return list(rows)


def _field(row: Any, column: str, index: int) -> Any:
    try:
        return row[column]
    except (KeyError, IndexError):
        raise ValueError(f"row {index} has no column {column!r}: {row!r}") from None


def _count(row: Any, column: str, index: int) -> float:
    value = _field(row, column, index)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"row {index}: {column!r} must be a number, got {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"row {index}: {column!r} must be a non-negative number, got {value!r}")
    return number


def _group(rows: Any, period: str, arm: str, exposures: str, events: str,
           cell: Optional[str]) -> List[Any]:
    """Sum the counts of each (period, cell, arm), and return the periods in order.

    The result is a list of ``(period, batch)``; a batch is ``{arm: [n, s]}``, or
    ``{cell: {arm: [n, s]}}`` when ``cell`` names a column.
    """
    records = _records(rows)
    if not records:
        raise ValueError("rows is empty: there is no history to fold")
    table: Dict[Any, Any] = {}
    for i, row in enumerate(records):
        p = _field(row, period, i)
        a = str(_field(row, arm, i))
        n, s = _count(row, exposures, i), _count(row, events, i)
        slot = table.setdefault(p, {})
        if cell is not None:
            slot = slot.setdefault(_field(row, cell, i), {})
        pair = slot.setdefault(a, [0.0, 0.0])
        pair[0] += n
        pair[1] += s
    for p, slot in table.items():
        groups = slot.values() if cell is not None else [slot]
        for group in groups:
            for a, (n, s) in group.items():
                if s > n:
                    raise ValueError(
                        f"period {p!r}, arm {a!r}: {s:g} events exceed {n:g} exposures")
    try:
        order = sorted(table)
    except TypeError:
        raise ValueError(
            "the periods cannot be ordered against one another; use one type for the "
            "period column (dates, ISO date strings, or numbers)") from None
    # Arms and cells in name order, so the table alone fixes the coordinates of the
    # fit and the Monte Carlo draws: the same rows give the same answer in any row order.
    def by_name(d):
        return dict(sorted(d.items(), key=lambda kv: str(kv[0])))

    def tidy(slot):
        if cell is None:
            return by_name(slot)
        return {c: by_name(group) for c, group in by_name(slot).items()}

    return [(p, tidy(table[p])) for p in order]


def batches_from_rows(rows: Iterable[Any], *, period: str = "period", arm: str = "arm",
                      exposures: str = "exposures", events: str = "events",
                      cell: Optional[str] = None) -> List[Any]:
    """Rows of ``(period, arm, exposures, events)`` as the list of batches ``update`` takes.

    Batches come back in period order.  Rows sharing a period and arm are summed.
    With ``cell`` naming a column, each batch is ``{cell: {arm: [exposures, events]}}``
    for ``ContextualLogisticBandit.update``; otherwise ``{arm: [exposures, events]}``.
    String periods sort as text, which is right for ISO dates and hours.
    """
    return [batch for _, batch in _group(rows, period, arm, exposures, events, cell)]


def replay(batches: Iterable[Batch], *, decay: float = 0.0, **bandit_kwargs: Any) -> LogisticBandit:
    """A fresh ``LogisticBandit`` that has absorbed ``batches`` in order.

    ``decay`` is passed to every ``update``; the remaining keyword arguments go to
    the ``LogisticBandit`` constructor (``contrast_prior``, ``arm_effect_prior_sd``, ...).
    A batch the model cannot learn from, such as one with no events, is skipped
    exactly as ``update`` skips it.
    """
    bandit = LogisticBandit(**bandit_kwargs)
    for batch in batches:
        bandit.update(batch, decay=decay)
    return bandit


def allocate_from_rows(rows: Iterable[Any], arms: Optional[Sequence[str]] = None, *,
                       period: str = "period", arm: str = "arm",
                       exposures: str = "exposures", events: str = "events",
                       decay: float = 0.0, draw: int = 100_000, floor: float = 0.0,
                       aggressive: float = 1.0, seed: Optional[int] = None,
                       **bandit_kwargs: Any) -> Allocation:
    """The next allocation, from the whole history in one call.

    Parameters
    ----------
    rows
        One row per (period, arm) with the users exposed in that period and how
        many of them had the event.  Counts are per period, not cumulative.
    arms
        The arms to allocate over.  Default: every arm that appears in ``rows``, in name order.
        Name only the live arms to drop an arm that was stopped.
    period, arm, exposures, events
        Column names.
    decay, draw, floor, aggressive
        As in ``LogisticBandit.update`` and ``allocate``.
    seed
        Seed for the Monte Carlo draws, so a scheduled job can be reproduced.
        ``None`` draws fresh numbers each run.
    bandit_kwargs
        Passed to ``LogisticBandit`` (``contrast_prior``, ``arm_effect_prior_sd``, ...).

    If no period has events and non-events in it, there is nothing to learn from
    yet and every arm gets the uniform start-up share, as ``allocate`` does.
    """
    grouped = _group(rows, period, arm, exposures, events, None)
    bandit = replay([batch for _, batch in grouped], decay=decay, **bandit_kwargs)
    seen = sorted({a for _, batch in grouped for a in batch})
    return bandit.allocate(list(arms) if arms is not None else seen, draw=draw, floor=floor,
                           aggressive=aggressive, rng=np.random.default_rng(seed))


def allocate_cells_from_rows(rows: Iterable[Any], *, cell: str = "cell",
                             arms: Optional[Sequence[str]] = None,
                             period: str = "period", arm: str = "arm",
                             exposures: str = "exposures", events: str = "events",
                             draw: int = 100_000, floor: float = 0.0, aggressive: float = 1.0,
                             seed: Optional[int] = None,
                             **bandit_kwargs: Any) -> Dict[Hashable, Allocation]:
    """Each cell's next allocation from rows of ``(period, cell, arm, exposures, events)``.

    The model is ``ContextualLogisticBandit`` (experimental): one logistic fit per
    period with a fresh intercept per cell, and a hierarchical prior over the
    arm-by-cell contrasts.  ``arms`` defaults to every arm in ``rows``, ordered by
    name, and the cells are the values of the ``cell`` column.  The remaining
    keyword arguments (``interaction_sd``, ``arm_effect_prior_sd``) go to the
    ``ContextualLogisticBandit`` constructor.  Returns ``{cell: Allocation}``.
    """
    grouped = _group(rows, period, arm, exposures, events, cell)
    seen_arms = sorted({a for _, batch in grouped for group in batch.values() for a in group})
    cells = sorted({c for _, batch in grouped for c in batch}, key=str)
    bandit = ContextualLogisticBandit(list(arms) if arms is not None else seen_arms, cells,
                                      **bandit_kwargs)
    for _, batch in grouped:
        bandit.update(batch)
    return bandit.allocate(draw=draw, floor=floor, aggressive=aggressive,
                           rng=np.random.default_rng(seed))
