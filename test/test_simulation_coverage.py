import sys
from types import SimpleNamespace

from methods import (
    IRNR,
    V321,
    Borda,
    BulletyApprovalWith,
    Irv,
    IrvPrime,
    Plurality,
    Rp,
    Schulze,
    Score,
)
from voterModels import DimVoter, KSElectorate, KSModel, Voter
from vse import CsvBatch, uniquify


def test_csv_batch_save_file_and_repo_metadata(tmp_path, monkeypatch):
    class FakeCommit:
        hexsha = "abc123"

    class FakeHead:
        commit = FakeCommit()

    class FakeRepo:
        head = FakeHead()

        def __init__(self, cwd):
            self.cwd = cwd

        def is_dirty(self):
            return False

    class Model:
        def __str__(self):
            return "Model"

        def __call__(self, nvot, ncand):
            return [[0] * ncand for _ in range(nvot)]

    class FakeMethod:
        def resultsTable(self, eid, emodel, cands, voters, chooserFuns, media):
            return [{"vse": 1, "method": "fake", "chooser": "hon", "extra": "value"}]

    monkeypatch.setitem(sys.modules, "git", SimpleNamespace(Repo=FakeRepo))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coverage-results1.csv").write_text("old")

    batch = CsvBatch(
        Model(),
        [[FakeMethod(), []]],
        nvot=2,
        ncand=3,
        niter=1,
        baseName="coverage-results",
        media=lambda results, tally=None: results,
    )

    assert batch.seed == "coverage-results1"
    assert batch.repo_version == "abc123"
    assert batch.rows == [{"vse": 1, "method": "fake", "chooser": "hon", "extra": "value"}]
    assert (tmp_path / "coverage-results2.csv").exists()
    assert uniquify(["vse", "method", "vse", "chooser"]) == ["vse", "method", "chooser"]

    explicit_seed = CsvBatch(
        Model(),
        [[FakeMethod(), []]],
        nvot=2,
        ncand=3,
        niter=0,
        seed="manual-seed",
        force=True,
    )
    assert explicit_seed.seed == "manual-seed"
    assert explicit_seed.rows == []


def test_irv_helpers_and_generator_inputs():
    irv = Irv()
    schedule = irv.buildPreferenceSchedule([[0, 1, 2], [0, 2, 1], [2, 1, 0]])

    assert schedule == {(0, 1, 2): 1, (0, 2, 1): 1, (2, 1, 0): 1}
    assert irv.eliminateCandidate(schedule, "not-a-candidate") is schedule
    ranked = irv.candidateVotes({(0, 1): 3, (2, 1): 2})
    assert [candidate.candidate for candidate in ranked] == [0, 2, 1]
    assert irv.getLeast(ranked, keep={1}).candidate == 2
    assert irv.getLeast(ranked, keep={0, 1, 2}) is None
    assert irv.results(iter([[0, 1, 2], [2, 1, 0]]))[1] == 0

    ballot = Irv().stratBallotFor([3, 2, 1, 0])(Irv, Voter([6, 3, 5, 2]))
    assert sorted(ballot) == [0, 1, 2, 3]

    assert IrvPrime().results(iter([[0, 1], [1, 0]]))[0] in {0, 1}


def test_v321_honest_results_and_strategy_branches():
    ballots = [[2, 1, 0, 3], [3, 2, 1, 0], [0, 3, 2, 1], [1, 0, 3, 2]]
    V321.extraEvents = {}
    results = V321().results(ballots, isHonest=True)

    assert len(results) == 4
    assert set(V321.extraEvents) >= {"3beats1", "3beats2", "4beats1"}

    V321.extraEvents = {"3beats1": True, "4beats1": False}
    ballot2 = V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([4, 3, 2, 1]))
    assert sorted(ballot2) == [0, 0, 1, 2]

    V321.extraEvents = {"3beats1": False, "4beats1": True}
    ballot3 = V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([1, 2, 3, 4]))
    assert len(ballot3) == 4

    V321.extraEvents = {"3beats1": False, "4beats1": False}
    normal = V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([4, 3, 2, 1]))
    assert len(normal) == 4


def test_additional_method_branch_coverage(monkeypatch):
    ballot = [None] * 4
    Borda.fillPrefOrder(
        Voter([1, 4, 3, 2]),
        ballot,
        whichCands=[1, 2],
        lowSlot=5,
        nSlots=1,
        remainderScore=0,
    )
    assert ballot == [None, 5, 0, None]

    assert Borda().stratBallotFor([4, 3, 2])(Borda, Voter([4, 2, 1])) == [2, 0, 1]
    assert Plurality.oneVote([0, 0, 0], 1) == [0, 1, 0]
    assert str(Score(1)) == "IdealApproval"
    score = Score()
    assert score.stratBallotFor([1, 0, 2])(score.__class__, Voter([6, 7, 6])) == [10, 10, 10]
    assert str(BulletyApprovalWith(0.5)) == "BulletyApproval50"

    monkeypatch.setattr("methods.random.random", lambda: 0.9)
    assert BulletyApprovalWith(0.5).honBallot(
        BulletyApprovalWith(0.5).__class__, Voter([1, 2])
    ) == [
        0.0,
        1.0,
    ]
    monkeypatch.setattr("methods.random.random", lambda: 0.1)
    assert BulletyApprovalWith(0.5).honBallot(
        BulletyApprovalWith(0.5).__class__, Voter([1, 2])
    ) == [
        0,
        1,
    ]

    V321.extraEvents = {}
    assert len(V321().results([[0, 1, 2], [2, 1, 0]], isHonest=True)) == 3
    V321.extraEvents = {"3beats1": False, "4beats1": False}
    assert V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([4, 3, 1, 0])) == [2, 1, 0, 0]
    assert V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([5, 3, 4, 0])) == [2, 0, 0, 0]
    assert V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([3, 5, 0, 4])) == [1, 2, 0, 1]
    V321.extraEvents = {"3beats1": True, "4beats1": False}
    assert V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([2, 1, 4, 0]))[2] == 2
    V321.extraEvents = {"3beats1": False, "4beats1": True}
    assert V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([1, 2, 3, 4]))[3] == 2
    assert len(V321().stratBallotFor([4, 3, 2, 1])(V321, Voter([1, 4, 3, 2]))) == 4
    assert V321().stratBallotFor([3, 2, 1, 0])(V321, Voter([4, 1, 2, 3]))[0] == 2

    assert len(Schulze().results([[0, 1, 2], [1, 2, 0]])) == 3
    assert len(Schulze().stratBallotFor([3, 2, 1])(Schulze, Voter([1, 3, 2]))) == 3
    assert len(Schulze().stratBallotFor([3, 2, 1])(Schulze, Voter([3, 1, 2]))) == 3
    assert len(Rp().resolveCycle([[0, 2, -1], [-2, 0, 3], [1, -3, 0]], 3)) == 3

    irnr = IRNR()
    assert len(irnr.results([[10, 0], [0, 10], [0, 0]])) == 2
    assert len(irnr.results([[10, 0, 0], [0, 10, 0], [0, 0, 10]])) == 3
    assert IRNR.honBallot(IRNR, Voter([1, 2])) == (1, 2)
    assert len(irnr.stratBallotFor([3, 2, 1])(IRNR, Voter([3, 1, 2]))) == 3
    assert len(irnr.stratBallotFor([3, 2, 1])(IRNR, Voter([1, 3, 2]))) == 3


def test_ks_electorate_and_model_paths(monkeypatch):
    electorate = KSElectorate()
    electorate.numClusters = 1
    electorate.numSubclusters = [0]
    electorate.dcs = [1]

    random_values = iter([0.99, 0.0])
    monkeypatch.setattr("voterModels.random.random", lambda: next(random_values))
    monkeypatch.setattr("voterModels.random.gauss", lambda mu, sigma: mu + sigma)
    electorate.chooseClusters(2, alpha=1, caring=lambda: 1)

    assert electorate.clusters == [[0], [0]]
    dim_voter = electorate.asDims(Voter([0]), 0)
    assert dim_voter.cares == [1]

    electorate.dimWeights = [1]
    electorate.cands = [[0]]
    electorate.totWeight = 1
    electorate.fromDims([dim_voter], DimVoter)
    assert len(electorate) == 1

    monkeypatch.setattr("voterModels.beta.rvs", lambda *args: 0.1)
    monkeypatch.setattr("voterModels.random.random", lambda: 0.99)
    model = KSModel(dccut=0.2, wccut=0.2)
    assert str(model).startswith("KSModel_")
    built = model(1, 1)
    assert len(built) == 1
    assert built.dcs == [1]
