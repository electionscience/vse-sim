from collections import defaultdict
from types import SimpleNamespace

import pytest

import stratFunctions
import voterModels
from compat import as_builtin_scalar, ceil, floor, isnum, mean, median, sqrt, std
from dataClasses import (
    CandidateWithCount,
    Method,
    SideTally,
    Tallies,
    VseMethodRun,
    VseOneRun,
    rememberBallot,
    rememberBallots,
)
from mydecorators import (
    autoassign,
    cached_property,
    curried,
    decorator,
    memoized,
    setdefaultattr,
    timeit,
)
from stratFunctions import (
    Chooser,
    LazyChooser,
    OssChooser,
    ProbChooser,
    beHon,
    beStrat,
    biasedMediaFor,
    biaserAround,
    fuzzyMediaFor,
    orderOf,
    skewedMediaFor,
    topNMediaFor,
    truth,
)
from voterModels import (
    DeterministicModel,
    DimElectorate,
    DimModel,
    DimVoter,
    Electorate,
    PersonalityVoter,
    PolyaModel,
    ReverseModel,
)


def test_compat_scalar_helpers_return_builtin_values():
    assert as_builtin_scalar(mean([1, 2, 3])) == 2.0
    assert as_builtin_scalar("plain") == "plain"
    assert ceil(1.2) == 2.0
    assert floor(1.8) == 1.0
    assert median([3, 1, 2]) == 2.0
    assert sqrt(4) == 2.0
    assert round(std([1, 2, 3]), 5) == 0.8165
    assert isnum(1 + 2j)
    assert not isnum("1")


def test_decorator_helper_wraps_function_metadata():
    def shout(fn):
        def wrapper(value):
            return fn(value).upper()

        return wrapper

    @decorator(shout)
    def greet(value):
        """greeting"""
        return f"hello {value}"

    assert greet("vse") == "HELLO VSE"
    assert greet.__name__ == "greet"
    assert greet.__doc__ == "greeting"


def test_setdefaultattr_returns_existing_or_default():
    obj = SimpleNamespace(existing="kept")
    assert setdefaultattr(obj, "existing", "new") == "kept"
    assert setdefaultattr(obj, "missing", "created") == "created"
    assert obj.missing == "created"


def test_autoassign_assigns_args_kwargs_and_defaults():
    class Assigned:
        @autoassign
        def __init__(self, foo, bar="bar", baz="baz"):
            pass

    assigned = Assigned("foo", baz="custom")
    assert assigned.foo == "foo"
    assert assigned.bar == "bar"
    assert assigned.baz == "custom"

    class Included:
        @autoassign("bar", "baz")
        def __init__(self, foo, bar, baz="baz"):
            pass

    included = Included("foo", "bar")
    assert not hasattr(included, "foo")
    assert included.bar == "bar"
    assert included.baz == "baz"

    class Excluded:
        @autoassign(exclude=("secret",))
        def __init__(self, visible, secret="hidden"):
            pass

    excluded = Excluded("shown")
    assert excluded.visible == "shown"
    assert not hasattr(excluded, "secret")


def test_memoized_caches_plain_and_bound_calls():
    calls = []

    @memoized
    def add(left, right):
        """add docs"""
        calls.append((left, right))
        return left + right

    assert add(2, 3) == 5
    assert add(2, 3) == 5
    assert calls == [(2, 3)]
    assert repr(add) == "add docs"

    assert add([1], [2]) == [1, 2]
    assert add([1], [2]) == [1, 2]

    class Calculator:
        @memoized
        def double(self, value):
            return value * 2

    calc = Calculator()
    assert calc.double(4) == 8
    assert calc.double(4) == 8


def test_cached_property_caches_and_exposes_descriptor():
    class Counter:
        def __init__(self):
            self.calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return 42

    assert hasattr(Counter.value, "__get__")
    counter = Counter()
    assert counter.value == 42
    assert counter.value == 42
    assert Counter.value.__get__(counter, Counter) == 42
    assert counter.calls == 1


def test_curried_waits_until_all_arguments_arrive():
    @curried
    def add3(a, b, c):
        return a + b + c

    assert add3(1, 2, 3) == 6
    assert add3(1)(2)(3) == 6


def test_timeit_returns_result_and_prints_timing(capsys):
    @timeit
    def identity(value):
        return value

    assert identity("ok") == "ok"
    assert "identity" in capsys.readouterr().out


def test_tally_containers_and_serialization_paths():
    class KeyProvider:
        def allTallyKeys(self):
            return ["a", "b"]

    tally = SideTally()
    tally["a"] = 2
    tally["c"] = 5
    assert tally.serialize() == []
    assert sorted(tally.itemList()) == [("a", 2), ("c", 5)]

    tally.initKeys(KeyProvider())
    assert tally.serialize() == [2, 0]
    assert tally.fullSerialize() == [2, 0]
    assert tally.itemList() == [("a", 2), ("b", 0), ("c", 5)]

    list_tally = SideTally()
    list_tally["x"] = 7
    list_tally.initKeys(["x"])
    assert list_tally.serialize() == [7]

    empty_tally = SideTally()
    empty_tally["ignored"] = 1
    empty_tally.initKeys(object())
    assert empty_tally.serialize() == []


def test_tallies_generate_once_then_iterate_like_list():
    tallies = Tallies()
    for tally, value in zip(tallies, [5, 4, 3]):
        tally[value] += value

    assert [t.fullSerialize() for t in tallies] == [[5], [4], [3], []]
    assert tallies == tallies
    assert tallies != Tallies()
    assert tallies == list(tallies)


class SumMethod(Method):
    candScore = staticmethod(sum)

    @staticmethod
    def honBallot(cls, utils):
        return list(utils)

    @staticmethod
    def fillStratBallot(
        voter,
        polls,
        places,
        n,
        stratGap,
        ballot,
        frontId,
        frontResult,
        targId,
        targResult,
    ):
        ballot[targId] = 1
        return {"extra": frontId + targId}


def test_method_base_helpers_and_ballot_memoizers():
    method = SumMethod()

    assert str(method) == "SumMethod"
    assert method.results(iter([[1, 2], [3, 4]])) == [4, 6]
    with pytest.raises(NotImplementedError):
        Method.honBallot(Method, [1, 2])

    dummy = method.dummyBallotFor([10, 9])
    assert dummy(SumMethod, ["u"], SideTally()) == ["u"]

    def chooser(cls, voter, tally):
        tally["seen"] += 1
        return voter

    chooser.__name__ = "chooser"
    chooser.allTallyKeys = lambda: ["seen"]
    result = method.resultsFor([[1, 2], [3, 4]], chooser)
    assert result["results"] == [4, 6]
    assert result["chooser"] == "chooser"
    assert result["tally"].serialize() == [2]

    @rememberBallot
    def honBallot(cls, voter):
        return ["honest"]

    voter = SimpleNamespace()
    assert honBallot(SumMethod, voter) == ["honest"]
    assert voter.SumMethod_hon == ["honest"]
    assert honBallot.allTallyKeys() == []

    @rememberBallots
    def stratBallot(cls, voter):
        return {"strat": ["strategic"], "isStrat": True}

    assert stratBallot(SumMethod, voter) == ["strategic"]
    assert voter.SumMethod_strat == ["strategic"]
    assert voter.SumMethod_isStrat is True

    assert method.stratTarget2([(0, 5), (1, 4)]) == (0, 5, 1, 4)
    assert method.stratTarget3([(0, 5), (1, 4), (2, 3)]) == (0, 5, 2, 3)
    strategic = method.stratBallotFor([4, 5])

    class UtilityVoter:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    strategic_voter = UtilityVoter([2, 4])
    assert strategic(SumMethod, strategic_voter) == [1, 0]
    assert strategic_voter.SumMethod_extra == 1


def test_method_vse_on_builds_vse_run_from_multi_results():
    method = SumMethod()

    def fake_multi_results(voters, chooserFuns=(), **args):
        return [
            [([0, 10], "hon", [("event", 1)])],
            [("scenario", "fake")],
        ]

    voters = SimpleNamespace(socUtils=[0, 10])
    method.multiResults = fake_multi_results

    result = method.vseOn(voters, chooserFuns=["chooser"])
    assert result.method is SumMethod
    assert result.choosers == ["chooser"]
    assert result.extraEvents == [("scenario", "fake")]
    assert result.results[0].result == [1.0]
    assert result.results[0].strat == "hon"


def test_candidate_and_vse_run_data_holders():
    run = VseOneRun(result=[1], tallyItems=[("x", 2)], strat="hon")
    method_run = VseMethodRun(method="m", choosers=["c"], results=[run])
    candidate = CandidateWithCount(["A"], 3)

    assert run.result == [1]
    assert method_run.results == [run]
    assert candidate.candidate == ["A"]
    assert candidate.votes == 3


class DummyMethod:
    pass


def test_choosers_track_names_keys_and_selection(monkeypatch):
    base = Chooser("hon")
    assert base.getName() == "hon"
    assert base(DummyMethod, object(), {}) == "hon"
    assert base.__name__ == "Chooser"

    lazy = LazyChooser()
    tally = defaultdict(int)
    voter = SimpleNamespace(DummyMethod_hon="same", DummyMethod_strat="same")
    assert lazy(DummyMethod, voter, tally) == "hon"
    assert tally[lazy.myKeys[0]] == 0

    voter.DummyMethod_strat = "different"
    assert lazy(DummyMethod, voter, tally) == "extraStrat"
    assert tally[lazy.myKeys[0]] == 1

    oss = OssChooser()
    voter.DummyMethod_isStrat = False
    assert oss(DummyMethod, voter, tally) == "hon"
    voter.DummyMethod_isStrat = True
    voter.DummyMethod_stratGap = 2
    assert oss(DummyMethod, voter, tally) == "strat"
    assert tally[oss.myKeys[0]] == 1
    assert tally[oss.myKeys[1]] == 2
    assert "Oss.hon_strat." in oss.getName()

    pc = ProbChooser([(0.25, beHon), (0.25, beStrat)])
    monkeypatch.setattr(stratFunctions.random, "random", lambda: 0.1)
    assert pc(DummyMethod, voter, tally) == "hon"
    monkeypatch.setattr(stratFunctions.random, "random", lambda: 0.3)
    assert pc(DummyMethod, voter, tally) == "strat"
    monkeypatch.setattr(stratFunctions.random, "random", lambda: 0.9)
    assert pc(DummyMethod, voter, tally) == "strat"
    assert "Prob.hon25_strat25." in pc.getName()

    keyed = Chooser(subChoosers=[oss])
    keyed.addTallyKeys(tally)
    assert oss.myKeys[0] in tally


def test_media_helpers_and_tally_updates(monkeypatch):
    standings = [5, 4, 3, 2]
    assert truth(standings) is standings
    assert topNMediaFor(2)(standings) == [5, 4, 2, 2]
    assert biaserAround(2)(standings) > 0
    assert orderOf([1, 3, 2]) == [1, 2, 0]

    fuzzy_tally = defaultdict(int)
    shifts = iter([-10, 10, 0, 0])
    monkeypatch.setattr(stratFunctions.random, "gauss", lambda mu, sigma: next(shifts))
    assert fuzzyMediaFor(biaser=1)(standings, fuzzy_tally) == [-5, 14, 3, 2]
    assert fuzzy_tally["changed"] == 1

    stable_tally = defaultdict(int)
    assert biasedMediaFor(biaser=1, numerator=1)(standings, stable_tally) == [
        5,
        4,
        2.5,
        1.3333333333333333,
    ]
    assert stable_tally["changed"] == 0

    skewed_tally = defaultdict(int)
    assert skewedMediaFor(3)(standings, skewed_tally) == [5, 3.0, 1.0, -1.0]
    assert skewed_tally["changed"] == 0

    monkeypatch.setattr(stratFunctions.random, "gauss", lambda mu, sigma: 0)
    assert fuzzyMediaFor(biaser=1)(standings) == standings
    assert biasedMediaFor(biaser=1)(standings) == [5, 4, 2.5, 1.3333333333333333]
    assert skewedMediaFor(0)(standings) == standings


def test_voter_model_edge_cases_and_dimensional_helpers(monkeypatch):
    with pytest.raises(ValueError):
        ReverseModel()(3, 2)

    monkeypatch.setattr(voterModels.random, "randrange", lambda stop: stop - 1)
    polya = PolyaModel(seedVoters=1, alpha=1, seedModel=DeterministicModel(2))
    assert len(polya(2, 2)) == 2

    base_voter = PersonalityVoter([1])
    elec = SimpleNamespace(totWeight=1, dimWeights=[1], cands=[[0], [2]])
    dim_voter = DimVoter.fromDims(base_voter, elec)
    assert list(dim_voter) == [-1.0, -1.0]
    assert dim_voter.dims is base_voter
    assert dim_voter.elec is elec
    assert dim_voter.cluster == base_voter.cluster

    caring_voter = PersonalityVoter([1])
    caring_dim_voter = DimVoter.fromDims(caring_voter, elec, caring=[0.5])
    assert len(caring_dim_voter) == 2

    dim_electorate = DimElectorate()
    dim_electorate.dimWeights = [1, 2]
    assert dim_electorate.asDims("voter") == "voter"
    dim_electorate.calcTotWeight()
    assert dim_electorate.totWeight == 5

    dim_electorate.dimWeights = [1]
    dim_electorate.cands = [[0]]
    dim_electorate.totWeight = 1
    dim_electorate.fromDims([PersonalityVoter([0])], DimVoter)
    assert len(dim_electorate) == 1

    model = DimModel(2, dimWeights=[1, 1], baseElectorate=DeterministicModel(2))
    built = model(1, 1)
    assert isinstance(built, Electorate)
    assert len(built) == 1
    with pytest.raises(AssertionError):
        DimModel(2, dimWeights=[1])
