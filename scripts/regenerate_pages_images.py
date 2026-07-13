"""Regenerate the GitHub Pages charts from the IRV simulation results.

The non-IRV points are retained from the published chart data. The IRV points,
including the scenario breakdowns, are generated with the seeded configuration
in ``recalculate_irv_pages.py``.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.recalculate_irv_pages import DEFAULT_WORKERS, recalculate  # noqa: E402

STRATEGIES = [
    "a.100% honest",
    "b.50% 1-sided strategy",
    "c.50% strategic",
    "d.Smart 1-sided strat.",
    "e.100% 1-sided strategy",
    "f.100% strategic",
]
COLORS = {
    "a.100% honest": "#1f77b4",
    "b.50% 1-sided strategy": "#9467bd",
    "c.50% strategic": "#8c564b",
    "d.Smart 1-sided strat.": "#ff7f0e",
    "e.100% 1-sided strategy": "#2ca02c",
    "f.100% strategic": "#d62728",
}
IRV_CHOOSERS = {
    "a.100% honest": "honBallot",
    "b.50% 1-sided strategy": "Oss.hon_Prob.strat50_hon50..",
    "c.50% strategic": "Prob.strat50_hon50.",
    "d.Smart 1-sided strat.": "smartOss",
    "e.100% 1-sided strategy": "Oss.hon_strat.",
    "f.100% strategic": "stratBallot",
}
SMALL_METHODS = {
    " 1. Plurality": "Plurality",
    " 5. IRV/RCV": "IRV/RCV",
    " 9. IdealApproval": "Approval",
    "13. Star0to10": "STAR",
    "15. V321": "3-2-1 voting",
}


def embedded_data(path):
    """Load the first HTML widget data object from a legacy static chart."""
    text = path.read_text()
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', text, re.DOTALL
    )
    if match is None:
        raise ValueError(f"No chart data found in {path}")
    return json.loads(match.group(1))["x"]["data"]


def replace_embedded_data(path, data):
    """Replace the first widget's data while preserving its HTML wrapper."""
    text = path.read_text()
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', text, re.DOTALL
    )
    if match is None:
        raise ValueError(f"No chart data found in {path}")
    payload = json.loads(match.group(1))
    payload["x"]["data"] = data
    replacement = json.dumps(payload, separators=(",", ":"))
    updated, replacements = re.subn(
        r'(<script type="application/json"[^>]*>).*?(</script>)',
        lambda match: f"{match.group(1)}{replacement}{match.group(2)}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError(f"No chart data found in {path}")
    path.write_text(updated)


def method_number(label):
    return int(label.strip().split(".", maxsplit=1)[0])


def refreshed_vse_data(results):
    """Load VSE points and replace the complete IRV strategy series."""
    data = embedded_data(ROOT / "docs" / "vse-graph.html")
    for index, (method, strategy) in enumerate(zip(data["y"], data["col_var"], strict=False)):
        if "IRV/RCV" in method:
            data["x"][index] = results[IRV_CHOOSERS[strategy]]
    return data


def scenario_for_label(label):
    labels = {
        "1.Cond.": "cycle",
        "2.Easy": "easy",
        "3.Spoiler": "spoiler",
        "4.Ctr.": "squeeze",
        "5.Chicken": "chicken",
        "6.Other": "other",
    }
    return next(scenario for prefix, scenario in labels.items() if label.startswith(prefix))


def refresh_interactive_charts(results, outcomes, scenario_results, scenario_outcomes):
    """Update the IRV points stored in the four interactive chart files."""
    docs = ROOT / "docs"

    vse = refreshed_vse_data(results)
    replace_embedded_data(docs / "vse-graph.html", vse)

    breakdown_path = docs / "vsebreakdown.html"
    breakdown = embedded_data(breakdown_path)
    for index, method in enumerate(breakdown["y"]):
        if "IRV/RCV" in method:
            scenario = scenario_for_label(breakdown["symbol_var"][index])
            breakdown["x"][index] = scenario_results[scenario][
                IRV_CHOOSERS[breakdown["col_var"][index]]
            ]
    replace_embedded_data(breakdown_path, breakdown)

    strategy_path = docs / "stratstuff.html"
    strategy = embedded_data(strategy_path)
    index = strategy["lab"].index("IRV/RCV")
    strategy["x"][index] = outcomes["successes"] / outcomes["attempts"]
    strategy["y"][index] = outcomes["backfires"] / outcomes["attempts"]
    replace_embedded_data(strategy_path, strategy)

    strategy_breakdown_path = docs / "stratbreakdown.html"
    strategy_breakdown = embedded_data(strategy_breakdown_path)
    for index, method in enumerate(strategy_breakdown["col_var"]):
        if method == "IRV/RCV":
            scenario = scenario_for_label(strategy_breakdown["symbol_var"][index])
            outcome = scenario_outcomes[scenario]
            strategy_breakdown["x"][index] = outcome["successes"] / outcome["attempts"]
            strategy_breakdown["y"][index] = outcome["backfires"] / outcome["attempts"]
    replace_embedded_data(strategy_breakdown_path, strategy_breakdown)


def render_vse(data, output, size, selected=False):
    points = list(zip(data["x"], data["y"], data["col_var"], strict=False))
    if selected:
        points = [point for point in points if point[1] in SMALL_METHODS]
        labels = list(SMALL_METHODS)
        display_labels = list(SMALL_METHODS.values())
    else:
        labels = sorted({point[1] for point in points}, key=method_number)
        display_labels = [label.strip() for label in labels]

    positions = {label: index + 1 for index, label in enumerate(labels)}
    figure, axis = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    for value, method, strategy in points:
        axis.scatter(value, positions[method], color=COLORS[strategy], s=34, zorder=3)

    axis.set_xlim(0.55 if selected else -0.2, 1.0)
    axis.set_ylim(0.5, len(labels) + 0.9)
    axis.set_yticks(list(positions.values()), display_labels)
    axis.set_xlabel("vse", loc="right")
    axis.set_ylabel("method", loc="top")
    axis.grid(color="#d9d9d9", linewidth=0.8)
    if not selected:
        axis.axvline(0, color="#555555", linestyle=(0, (3, 3)), linewidth=1)
    axis.legend(
        [Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS[key], markersize=10)
         for key in STRATEGIES],
        STRATEGIES,
        title="strategy",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.14, right=0.80, bottom=0.12, top=0.96)
    figure.savefig(output, dpi=100)
    plt.close(figure)


def render_strategy(results, outcomes, output):
    data = embedded_data(ROOT / "docs" / "stratstuff.html")
    index = data["lab"].index("IRV/RCV")
    data["x"][index] = outcomes["successes"] / outcomes["attempts"]
    data["y"][index] = outcomes["backfires"] / outcomes["attempts"]

    figure, axis = plt.subplots(figsize=(9.5, 6.77), dpi=100)
    palette = plt.get_cmap("tab20")
    for index, (x, y, label) in enumerate(zip(data["x"], data["y"], data["lab"], strict=False)):
        axis.scatter(x, y, s=65, color=palette(index), zorder=3)
        axis.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="stratWorks", ylabel="stratBackfire")
    axis.grid(color="#d9d9d9", linewidth=0.8)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98)
    figure.savefig(output, dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elections",
        type=int,
        default=15_000,
        help="Number of elections to simulate (use a smaller value while developing charts).",
    )
    parser.add_argument("--seed", default="target15000")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Deterministic simulation chunks (default: 10).",
    )
    args = parser.parse_args()
    results, outcomes, scenario_results, scenario_outcomes = recalculate(
        args.elections, args.seed, args.workers
    )
    data = refreshed_vse_data(results)
    docs = ROOT / "docs"
    refresh_interactive_charts(results, outcomes, scenario_results, scenario_outcomes)
    render_vse(data, docs / "vse.png", (970, 681))
    render_vse(data, docs / "5vse.png", (952, 567), selected=True)
    render_vse(data, docs / "5vse_small.png", (656, 271), selected=True)
    render_strategy(results, outcomes, docs / "vsestrat.png")


if __name__ == "__main__":
    main()
