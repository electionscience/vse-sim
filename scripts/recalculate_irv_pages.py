"""Recalculate the IRV/RCV VSE values cited by the GitHub Pages site.

This reproduces the historical run configuration documented in vse.py without
retaining every election result in memory.  For the published 15,000-election
run, use:

    uv run python scripts/recalculate_irv_pages.py
"""

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debugDump import setDebug
from methods import Irv
from voterModels import KSModel
from vse import baseRuns, fuzzyMediaFor


def recalculate(elections, seed):
    """Return IRV VSE and one-sided strategic-voting outcomes by ballot strategy."""
    setDebug(False)
    random.seed(seed)
    numpy_seed = int.from_bytes(
        hashlib.sha256(str(seed).encode()).digest()[:4], byteorder="little"
    )
    np.random.seed(numpy_seed)
    model = KSModel(dcdecay=(1, 3), wcdecay=(1.5, 3), dccut=.2, wcalpha=1.5)
    method = Irv()
    media = fuzzyMediaFor()
    summary = defaultdict(lambda: [0, 0.0])
    one_sided_strategy = {"attempts": 0, "successes": 0, "backfires": 0}

    for election in range(elections):
        electorate = model(40, 6)
        for row in method.resultsTable(
            election, str(model), 6, electorate, baseRuns, media=media
        ):
            count_and_total = summary[row["chooser"]]
            count_and_total[0] += 1
            count_and_total[1] += row["vse"]
            if row["chooser"] == "Oss.hon_strat.":
                one_sided_strategy["attempts"] += 1
                for index in range(4):
                    if row.get(f"tallyName{index}") == "worked":
                        worked = int(row[f"tallyVal{index}"])
                        one_sided_strategy["successes"] += worked == 1
                        one_sided_strategy["backfires"] += worked == -1

    return (
        {chooser: total / count for chooser, (count, total) in summary.items()},
        one_sided_strategy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elections", type=int, default=15_000)
    parser.add_argument("--seed", default="target15000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results, one_sided_strategy = recalculate(args.elections, args.seed)
    rows = [
        (chooser, args.elections, value, 100 * value)
        for chooser, value in sorted(results.items())
    ]
    if args.output:
        with args.output.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["chooser", "elections", "mean_vse", "percent_vse"])
            writer.writerows(rows)
            writer.writerow([])
            writer.writerow(["strategy", "attempts", "success_rate", "backfire_rate"])
            attempts = one_sided_strategy["attempts"]
            writer.writerow([
                "Oss.hon_strat.",
                attempts,
                f"{one_sided_strategy['successes'] / attempts:.12f}",
                f"{one_sided_strategy['backfires'] / attempts:.12f}",
            ])
    else:
        for chooser, _elections, _value, percent in rows:
            print(f"{chooser:30} {percent:.6f}%")
        attempts = one_sided_strategy["attempts"]
        print(f"one-sided strategic success: {one_sided_strategy['successes'] / attempts:.6%}")
        print(f"one-sided strategic backfire: {one_sided_strategy['backfires'] / attempts:.6%}")


if __name__ == "__main__":
    main()
