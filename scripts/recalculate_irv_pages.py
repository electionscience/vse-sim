"""Recalculate the IRV/RCV VSE values cited by the GitHub Pages site.

This reproduces the historical run configuration documented in vse.py without
retaining every election result in memory.  For the published 15,000-election
run, use:

    uv run python scripts/recalculate_irv_pages.py
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debugDump import setDebug
from methods import Irv
from voterModels import KSModel
from vse import baseRuns, fuzzyMediaFor


def recalculate(elections, seed):
    """Return mean VSE by IRV ballot strategy for the historical configuration."""
    setDebug(False)
    random.seed(seed)
    model = KSModel(dcdecay=(1, 3), wcdecay=(1.5, 3), dccut=.2, wcalpha=1.5)
    method = Irv()
    media = fuzzyMediaFor()
    summary = defaultdict(lambda: [0, 0.0])

    for election in range(elections):
        electorate = model(40, 6)
        for row in method.resultsTable(
            election, str(model), 6, electorate, baseRuns, media=media
        ):
            count_and_total = summary[row["chooser"]]
            count_and_total[0] += 1
            count_and_total[1] += row["vse"]

    return {
        chooser: total / count
        for chooser, (count, total) in summary.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elections", type=int, default=15_000)
    parser.add_argument("--seed", default="target15000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = recalculate(args.elections, args.seed)
    rows = [
        (chooser, args.elections, value, 100 * value)
        for chooser, value in sorted(results.items())
    ]
    if args.output:
        with args.output.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["chooser", "elections", "mean_vse", "percent_vse"])
            writer.writerows(rows)
    else:
        for chooser, _elections, _value, percent in rows:
            print(f"{chooser:30} {percent:.6f}%")


if __name__ == "__main__":
    main()
