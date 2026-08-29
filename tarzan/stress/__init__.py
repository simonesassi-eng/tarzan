"""A reusable structural bench for Tarzan.

Not a unit-test suite. It drives the REAL pipeline over seeded synthetic books,
across pinned market instants and run modes, and judges the output by invariants
and differential comparison rather than by expected values — because on a random
book there is no known-good number to compare against.

Read ``PLAN.md`` for the matrix, the checks and their pre-registered tolerances.
Nothing here reads ``input/`` or writes outside a temporary directory.
"""
