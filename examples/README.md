# Examples

| file | what it shows |
|---|---|
| `basic_usage.py` | Algorithm 1 one boundary at a time: update, allocate, a level shift, the two controls, stopping quantities |
| `comparison.py` | OR-TS against Beta-TS and Full-TS under a common shock redrawn every batch |
| `ab_testing.py` | Migrating from a Beta-Bernoulli service with a warm start, then the default stopping and dropping rule |
| `from_warehouse.py` | A scheduled job's view: run `sql/period_counts.sql` on exposure and conversion logs (SQLite stands in for the warehouse), then `allocate_from_rows` |
| `make_readme_figures.py` | Regenerates the two figures in the README (needs matplotlib) |

Run any of them from the repository root after `pip install -e .`.
