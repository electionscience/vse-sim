# AGENTS.md

## Project

This repository runs Monte Carlo simulations of Voter Satisfaction Efficiency
(VSE) for voting methods under different electorate and strategy models. The
published explanation and results live in `docs/`.

The code currently uses a flat module layout. Run commands from the repository
root; do not assume the project is installed as a Python package.

## Environment and validation

- Supported Python: 3.10 through 3.12; local and CI default to Python 3.12.
- Dependency manager: `uv`; keep `uv.lock` in sync with `pyproject.toml`.
- Install dependencies with `uv sync --locked`.
- Run the test suite with `uv run python -m pytest`.
- Run repository lint and security checks with `trunk check`.
- Do not commit generated `SimResults*.csv` or ad hoc simulation dumps.

Pytest is configured with `--doctest-modules`, so examples in module docstrings
are tests. Add focused pytest tests for regressions that are awkward to express
as doctests. Keep random tests deterministic and seed both Python's `random`
module and NumPy.

## Code map

- `vse.py`: simulation orchestration, method presets, and CSV output.
- `dataClasses.py`: core method API, tallies, ballot caching, and VSE rows.
- `methods.py`: voting method and ballot implementations.
- `voterModels.py`: voter, electorate, and spatial/clustered voter models.
- `stratFunctions.py`: strategic ballot choosers and media models.
- `mydecorators.py`: local decorators used throughout the simulation.
- `scripts/recalculate_irv_pages.py`: reproducible, parallel IRV calculations.
- `scripts/regenerate_pages_images.py`: generated HTML and chart updates.
- `docs/`: GitHub Pages source plus committed generated charts.
- `sodaTest.py`: experimental legacy code; do not make production code depend
  on it.

## Change guidance

### Voting methods

Voting methods derive from `dataClasses.Method`. Preserve the existing ballot
and result conventions unless a deliberate migration updates all callers:

- candidate results are index-aligned sequences;
- the winning candidate is selected through `Method.winner`;
- ballot functions are memoized on voter objects by method class name;
- chooser names and tally fields are serialized into CSV and may be consumed by
  scripts or published-data tooling.

Add tests for ties, identical utilities, empty or minimal profiles, and cyclic
profiles as applicable. Do not infer correctness from one happy-path doctest.

### Simulation state and randomness

Election metadata is held in the method instance's `ElectionContext`, and Mav
cutoffs are captured by the election's ballot function. Keep this state
election-scoped:

- reset it before each independent election;
- do not parallelize elections that share method classes unless state has first
  been isolated;
- do not introduce new class-level mutable simulation state;
- extend `ElectionContext` instead of adding implicit cross-phase state.

For reproducible runners, derive and set both Python and NumPy seeds. Prefer
local RNG objects in new code over adding more process-global RNG use.

### Numerical behavior

VSE normalizes by `best - rand`, and score ballots normalize by each voter's
utility range. Handle zero ranges explicitly. Define and test the intended
result rather than allowing `ZeroDivisionError`, NaN, or infinity.

Avoid private NumPy import paths such as `numpy.core.*`; use public `numpy`
APIs. NumPy 2.x remains unsupported until behavioral compatibility has been
validated and the dependency bounds are deliberately updated.

### Published results

Changes to voter generation, strategies, tabulation, tie-breaking, seeding, or
VSE normalization can alter published numbers. When such behavior changes:

1. Add a small deterministic regression test.
2. Run an appropriately sized smoke calculation with
   `scripts/recalculate_irv_pages.py`.
3. If the change is intended to update published results, regenerate the site
   artifacts and explain the changed assumptions in the same change.
4. Do not hand-edit generated HTML or PNG output.

The full published run can be expensive. Use a small election count while
developing, then use the documented seed and full command before publishing.

## Refactoring priorities

When touching nearby code, prefer small staged changes in this order:

1. Protect correctness with regression tests, especially Schulze cycles,
   normalization edge cases, and strategy chooser behavior.
2. Extend the explicit election context instead of introducing shared state.
3. Use `retain_rows=False` for large CSV batches and preserve the streaming
   path when changing persistence.
4. Introduce a package layout only as a deliberate migration; update scripts,
   doctests, CI, and imports together.

Do not combine algorithm changes with broad formatting or module moves. Voting
method changes should remain reviewable against the prior mathematical
behavior.

## Repository hygiene

- Preserve unrelated working-tree changes.
- Keep runtime dependencies minimal; charting and test tools belong in the dev
  dependency group.
- Update `README.md` when setup or common commands change.
- If generated files change, identify the generating command in the change
  description.
