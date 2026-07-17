"""Generate the complete simulation CSV consumed by the chart analysis.

Example:

    uv run python scripts/generate_published_results.py --elections 15000 \
        --output artifacts/published-results
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debugDump import setDebug
from vse import CsvBatch, KSModel, allSystems, fuzzyMediaFor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elections", type=int, default=15_000)
    parser.add_argument("--seed", default="target15000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/published-results"),
        help="Filename prefix; the runner appends a numeric CSV suffix.",
    )
    args = parser.parse_args()

    setDebug(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch = CsvBatch(
        KSModel(dcdecay=(1, 3), wcdecay=(1.5, 3), dccut=.2, wcalpha=1.5),
        allSystems,
        nvot=40,
        ncand=6,
        niter=args.elections,
        baseName=str(args.output),
        media=fuzzyMediaFor(),
        seed=args.seed,
        force=True,
        retain_rows=False,
    )
    print(batch.output_file)


if __name__ == "__main__":
    main()
