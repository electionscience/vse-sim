"""Generate the IRV/RCV VSE values cited by the GitHub Pages site.

This reproduces the historical run configuration in
``vse_sim.simulation`` without retaining every election result in memory. For
the published 15,000-election run, use:

    uv run python scripts/recalculate_irv_pages.py
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from vse_sim.diagnostics import setDebug
from vse_sim.methods import Irv, Schulze
from vse_sim.simulation import baseRuns, fuzzyMediaFor, seedRandomGenerators
from vse_sim.voter_models import KSModel

DEFAULT_WORKERS = 10


def _recalculate_chunk(elections, seed):
    """Simulate one independently seeded chunk of elections."""
    setDebug(False)
    seedRandomGenerators(seed)
    model = KSModel(dcdecay=(1, 3), wcdecay=(1.5, 3), dccut=.2, wcalpha=1.5)
    method = Irv()
    scenario_method = Schulze()
    media = fuzzyMediaFor()
    summary = defaultdict(lambda: [0, 0.0, 0.0])
    scenario_summary = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    scenario_outcomes = defaultdict(
        lambda: {"attempts": 0, "successes": 0, "backfires": 0}
    )
    one_sided_strategy = {"attempts": 0, "successes": 0, "backfires": 0}

    for election in range(elections):
        electorate = model(40, 6)
        scenario_method.resultsFor(
            electorate,
            scenario_method.honBallotFor(electorate),
            isHonest=True,
        )
        scenario = scenario_method.extraEvents["scenario"]
        for row in method.resultsTable(
            election, str(model), 6, electorate, baseRuns, media=media
        ):
            count_and_total = summary[row["chooser"]]
            count_and_total[0] += 1
            count_and_total[1] += row["vse"]
            count_and_total[2] += row["vse"] ** 2
            scenario_count_and_total = scenario_summary[scenario][row["chooser"]]
            scenario_count_and_total[0] += 1
            scenario_count_and_total[1] += row["vse"]
            scenario_count_and_total[2] += row["vse"] ** 2
            if row["chooser"] == "Oss.hon_strat.":
                one_sided_strategy["attempts"] += 1
                scenario_outcomes[scenario]["attempts"] += 1
                for index in range(4):
                    if row.get(f"tallyName{index}") == "worked":
                        worked = int(row[f"tallyVal{index}"])
                        one_sided_strategy["successes"] += worked == 1
                        one_sided_strategy["backfires"] += worked == -1
                        scenario_outcomes[scenario]["successes"] += worked == 1
                        scenario_outcomes[scenario]["backfires"] += worked == -1

    return (
        dict(summary),
        one_sided_strategy,
        {scenario: dict(strategies) for scenario, strategies in scenario_summary.items()},
        dict(scenario_outcomes),
    )


def recalculate(elections, seed, workers=DEFAULT_WORKERS):
    """Return IRV results overall and by honest-ballot scenario type.

    Each worker receives a deterministic seed derived from ``seed`` and its
    chunk number.  This makes a parallel run reproducible regardless of task
    completion order.
    """
    chunks = min(workers, elections)
    chunk_sizes = [elections // chunks + (index < elections % chunks)
                   for index in range(chunks)]
    chunk_seeds = [f"{seed}:chunk:{index}" for index in range(chunks)]
    if chunks == 1:
        chunks = [_recalculate_chunk(chunk_sizes[0], chunk_seeds[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, chunks)) as executor:
            chunks = list(executor.map(_recalculate_chunk, chunk_sizes, chunk_seeds))

    summary = defaultdict(lambda: [0, 0.0, 0.0])
    scenario_summary = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    one_sided_strategy = {"attempts": 0, "successes": 0, "backfires": 0}
    scenario_outcomes = defaultdict(
        lambda: {"attempts": 0, "successes": 0, "backfires": 0}
    )
    for chunk_summary, chunk_outcomes, chunk_scenarios, chunk_scenario_outcomes in chunks:
        for chooser, (count, total, total_squared) in chunk_summary.items():
            summary[chooser][0] += count
            summary[chooser][1] += total
            summary[chooser][2] += total_squared
        for key in one_sided_strategy:
            one_sided_strategy[key] += chunk_outcomes[key]
        for scenario, strategies in chunk_scenarios.items():
            for chooser, (count, total, total_squared) in strategies.items():
                scenario_summary[scenario][chooser][0] += count
                scenario_summary[scenario][chooser][1] += total
                scenario_summary[scenario][chooser][2] += total_squared
        for scenario, outcomes in chunk_scenario_outcomes.items():
            for key in outcomes:
                scenario_outcomes[scenario][key] += outcomes[key]

    results = {chooser: total / count for chooser, (count, total, _sum_sq) in summary.items()}
    confidence_intervals = {}
    for chooser, (count, total, total_squared) in summary.items():
        if count < 2:
            confidence_intervals[chooser] = 0.0
            continue
        variance = max(0.0, (total_squared - total ** 2 / count) / (count - 1))
        confidence_intervals[chooser] = 1.96 * math.sqrt(variance / count)

    return (
        results,
        one_sided_strategy,
        {
            scenario: {
                chooser: total / count
                for chooser, (count, total, _sum_sq) in strategy_results.items()
            }
            for scenario, strategy_results in scenario_summary.items()
        },
        scenario_outcomes,
        confidence_intervals,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elections", type=int, default=15_000)
    parser.add_argument("--seed", default="target15000")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Deterministic simulation chunks (default: 10).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results, one_sided_strategy, scenario_results, scenario_outcomes, confidence_intervals = recalculate(
        args.elections, args.seed, args.workers
    )
    rows = [
        (chooser, args.elections, value, confidence_intervals[chooser], 100 * value,
         100 * (value - confidence_intervals[chooser]),
         100 * (value + confidence_intervals[chooser]))
        for chooser, value in sorted(results.items())
    ]
    if args.output:
        with args.output.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["chooser", "elections", "mean_vse", "ci95_half_width", "percent_vse", "ci95_low_percent", "ci95_high_percent"])
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
            writer.writerow([])
            writer.writerow(["scenario", "chooser", "mean_vse", "percent_vse"])
            for scenario in sorted(scenario_results):
                for chooser, value in sorted(scenario_results[scenario].items()):
                    writer.writerow([scenario, chooser, value, 100 * value])
            writer.writerow([])
            writer.writerow(["scenario", "attempts", "success_rate", "backfire_rate"])
            for scenario, outcome in sorted(scenario_outcomes.items()):
                attempts = outcome["attempts"]
                writer.writerow([
                    scenario,
                    attempts,
                    f"{outcome['successes'] / attempts:.12f}",
                    f"{outcome['backfires'] / attempts:.12f}",
                ])
    else:
        for chooser, _elections, _value, percent in rows:
            print(f"{chooser:30} {percent:.6f}%")
        attempts = one_sided_strategy["attempts"]
        print(f"one-sided strategic success: {one_sided_strategy['successes'] / attempts:.6%}")
        print(f"one-sided strategic backfire: {one_sided_strategy['backfires'] / attempts:.6%}")


if __name__ == "__main__":
    main()
