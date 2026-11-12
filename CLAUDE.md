# Working on Tarzan

- **Backtests and every simulated history follow `docs/BACKTEST_METHOD.md`.**
  There is one reconstruction, `tarzan.backtest.engine.instrument_base_returns`.
  Never copy it, and never monkeypatch the engine from an analysis script. If a
  treatment is missing, add it to the engine with a test and document it there.
- Run tests with `.venv/bin/python -m pytest` (pinned dependencies).
- **This repository is public.** Do not commit holdings, quantities, values,
  ISIN lists of a real book, personal links or identifiers, including in code
  comments and commit messages. `input/` is git-ignored except
  `instrument_taxonomy.csv`.
- Before merging a change to returns or data: verify it against real prices, run
  the full pipeline (`python -m tarzan.main`), then run the full suite.
