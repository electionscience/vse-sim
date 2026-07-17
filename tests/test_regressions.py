import csv
import random
from pathlib import Path

import numpy as np
import pytest

from scripts.recalculate_irv_pages import recalculate
from vse_sim.core import SideTally
from vse_sim.diagnostics import TRACE, setDebug, trace
from vse_sim.methods import Irv, Mav, Schulze, Score
from vse_sim.simulation import CsvBatch, seedRandomGenerators
from vse_sim.strategies import ProbChooser, beHon, beStrat
from vse_sim.voter_models import Electorate, Voter


def test_schulze_uses_independent_strongest_path_rows():
    margins = [
        [0, -3, 1],
        [3, 0, -1],
        [-1, 1, 0],
    ]

    assert Schulze().resolveCycle(margins, 3) == [1, 2, 0]


def test_schulze_metadata_is_scoped_to_method_instance():
    cycle_method = Schulze()
    easy_method = Schulze()

    cycle_method.results(
        [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        isHonest=True,
    )
    easy_method.results([[0, 1, 2]], isHonest=True)

    assert cycle_method.extraEvents == {"scenario": "cycle"}
    assert easy_method.extraEvents == {"scenario": "easy"}


def test_score_and_vse_handle_identical_utilities():
    voters = Electorate([Voter([1, 1, 1]), Voter([1, 1, 1])])
    method = Score()

    assert method.honBallot(method.__class__, voters[0]) == [10, 10, 10]
    assert all(row["vse"] == 0.0 for row in method.resultsTable(
        "equal", "equal", 3, voters
    ))


def test_vse_on_returns_every_simulation_run():
    voters = Electorate([Voter([0, 1]), Voter([0, 1])])

    result = Score().vseOn(voters)

    assert {run.strat for run in result.results} == {
        "honBallot",
        "stratBallot",
        "Oss.hon_strat.",
        "smartOss",
    }
    assert all(run.result == [1.0] for run in result.results)
    assert result.extraEvents == {}


def test_mav_cutoffs_are_scoped_to_generated_ballot_function():
    method = Mav()
    low_electorate = Electorate([Voter([-2, -1]), Voter([-2, -1])])
    high_electorate = Electorate([Voter([1, 2]), Voter([1, 2])])
    low_ballot = method.honBallotFor(low_electorate)
    expected = low_ballot(Mav, Voter([-2, -1]), SideTally())

    method.honBallotFor(high_electorate)

    assert low_ballot(Mav, Voter([-2, -1]), SideTally()) == expected


def test_irv_results_keep_simulator_score_contract_for_strategy():
    method = Irv()
    voters = Electorate(
        [Voter([0, 1, 2])] * 4
        + [Voter([2, 1, 0])] * 3
        + [Voter([1, 2, 0])] * 2
    )

    results = method.resultsFor(voters, method.honBallot)["results"]
    polls = sorted(enumerate(results), key=lambda candidate_result: -candidate_result[1])

    assert results == [2, 0, 1]
    assert method.winner(results) == 0
    assert polls[0][0] == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [(-0.1, beHon), (1.1, beStrat)],
        [(0.25, beHon), (0.25, beStrat)],
    ],
)
def test_prob_chooser_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        ProbChooser(probabilities)


def test_prob_chooser_falls_back_to_last_choice(monkeypatch):
    chooser = ProbChooser([(0.5, beHon), (0.5, beStrat)])
    monkeypatch.setattr(random, "random", lambda: 1.0)

    assert chooser(object, object(), SideTally()) == "strat"


def test_prob_chooser_selects_both_choices_and_tracks_non_default_choice():
    seedRandomGenerators("prob-chooser")
    chooser = ProbChooser([(0.3, beHon), (0.7, beStrat)])
    tally = SideTally()

    choices = [chooser(object, object(), tally) for _ in range(500)]

    assert set(choices) == {"hon", "strat"}
    assert tally[f"{chooser.getName()}_strat"] == choices.count("strat")


def test_seed_random_generators_is_reproducible():
    seedRandomGenerators("same-seed")
    first = (random.random(), np.random.random())
    seedRandomGenerators("same-seed")

    assert (random.random(), np.random.random()) == first

    seedRandomGenerators("seed-a")
    sequence_a = (random.random(), np.random.random())
    seedRandomGenerators("seed-b")
    sequence_b = (random.random(), np.random.random())

    assert sequence_a != sequence_b


def test_csv_batch_can_stream_without_retaining_rows(tmp_path):
    output_base = str(tmp_path / "results")
    batch = CsvBatch(
        _NumpyModel(),
        [[Score(), []]],
        nvot=3,
        ncand=2,
        niter=2,
        baseName=output_base,
        seed="stream-test",
        force=True,
        retain_rows=False,
    )

    assert batch.rows == []
    output_path = Path(batch.output_file)
    assert output_path.exists()

    with output_path.open(newline="") as output:
        assert output.readline().startswith("# {")
        rows = list(csv.DictReader(output))

    expected_choosers = {
        "honBallot",
        "stratBallot",
        "Oss.hon_strat.",
        "smartOss",
    }
    assert len(rows) == batch.niter * len(expected_choosers)
    assert {row["chooser"] for row in rows} == expected_choosers
    assert {"eid", "util", "vse"} <= rows[0].keys()


def test_irv_recalculation_smoke():
    results, outcomes, scenarios, scenario_outcomes, intervals = recalculate(
        elections=2,
        seed="test-irv",
        workers=1,
    )

    assert "honBallot" in results
    assert outcomes["attempts"] == 2
    assert intervals["honBallot"] >= 0
    assert sum(data["attempts"] for data in scenario_outcomes.values()) == 2
    assert set(scenarios).issubset(
        {"cycle", "easy", "spoiler", "squeeze", "chicken", "other"}
    )


class _NumpyModel:
    def __call__(self, nvot, ncand):
        return Electorate(
            Voter(np.random.normal(size=ncand))
            for _ in range(nvot)
        )


def test_trace_diagnostics_use_logging(caplog):
    setDebug(True)
    try:
        with caplog.at_level(TRACE, logger="vse_sim"):
            trace("election", 7)
        assert "election 7" in caplog.text
    finally:
        setDebug(False)
