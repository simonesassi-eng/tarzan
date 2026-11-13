# Working on Tarzan

- **Backtests and every simulated history follow `docs/BACKTEST_METHOD.md`.**
  There is one reconstruction, `tarzan.backtest.engine.instrument_base_returns`.
  Never copy it, and never monkeypatch the engine from an analysis script. If a
  treatment is missing, add it to the engine with a test and document it there.
- Run tests with `.venv/bin/python -m pytest` (pinned dependencies).
- **This repository is public.** Do not commit holdings, quantities, values,
  allocation weights, the book's own returns, ISIN lists of a real book,
  personal links or identifiers, including in code comments, tests and commit
  messages. Never quote a live run to explain a change: use a synthetic figure
  and write "synthetic" on the line. `input/` is git-ignored except
  `instrument_taxonomy.csv`. The `.githooks` (money amounts, plus the real
  target weights and returns read from `input/`) block most slips; they are a
  net, not the rule.
- Before merging a change to returns or data: verify it against real prices, run
  the full pipeline (`python -m tarzan.main`), then run the full suite.
