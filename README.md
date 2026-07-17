# Voter Satisfaction Efficiency

This repository runs Voter Satisfaction Efficiency (VSE) simulations for
different voting systems, electorate models, and strategic behaviors.

See the [VSE FAQ](https://electionscience.github.io/vse-sim/) for an explanation
of the methods and published results.

See the [chart reproduction guide](docs/chart-reproduction.md) to generate the
full simulation CSV used by the chart analysis.

## Setup

The project supports Python 3.10 through 3.12 and uses
[uv](https://docs.astral.sh/uv/) with a committed lockfile.

```sh
uv sync --locked
```

The repository has a flat module layout, so run commands from its root.

## Validation

Doctests are part of the pytest suite:

```sh
uv run python -m pytest
trunk check
```

## Running simulations

```python
from voterModels import PolyaModel
from vse import CsvBatch, Mav, Score, baseRuns, medianRuns

batch = CsvBatch(
    PolyaModel(),
    [[Score(), baseRuns], [Mav(), medianRuns]],
    nvot=5,
    ncand=4,
    niter=3,
)
batch.saveFile()
```

This writes the next available `SimResultsN.csv`.

Large runs can write rows directly instead of retaining every row in memory:

```python
CsvBatch(
    PolyaModel(),
    [[Score(), baseRuns]],
    nvot=40,
    ncand=6,
    niter=15_000,
    baseName="SimResults",
    retain_rows=False,
)
```

## Reproducing published IRV results

Use a small deterministic run while developing:

```sh
uv run python scripts/recalculate_irv_pages.py \
  --elections 50 \
  --workers 1 \
  --seed smoke
```

The full published configuration and seed are the script defaults:

```sh
uv run python scripts/recalculate_irv_pages.py
uv run python scripts/regenerate_pages_images.py
```

Changes to voter generation, strategies, tabulation, tie-breaking, random
seeding, or VSE normalization can change published results. See `AGENTS.md` for
the required regeneration workflow.
