from types import SimpleNamespace

import numpy as np

import sodaTest
from sodaTest import ElectionCounts, autoargs, cached_property


def test_autoargs_assigns_defaults_varargs_kwargs_and_excludes():
    class Assigned:
        @autoargs()
        def __init__(self, first, second="default", *extra, **named):
            self.finished = True

    assigned = Assigned("one", "two", "three", label="four")
    assert assigned.first == "one"
    assert assigned.second == "two"
    assert assigned.extra == ("three",)
    assert assigned.label == "four"
    assert assigned.finished is True

    class Excluded:
        @autoargs(exclude=("secret",))
        def __init__(self, visible, secret="hidden"):
            pass

    excluded = Excluded("shown")
    assert excluded.visible == "shown"
    assert not hasattr(excluded, "secret")

    class Sparse:
        @autoargs("first")
        def __init__(self, first, second, *extra, **named):
            pass

    sparse = Sparse("kept", "ignored", "extra", second_kw="hidden")
    assert sparse.first == "kept"
    assert not hasattr(sparse, "second")
    assert not hasattr(sparse, "extra")
    assert not hasattr(sparse, "second_kw")


def test_cached_property_descriptor_caches_value():
    class Counter:
        def __init__(self):
            self.calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return 10

    assert Counter.value.__get__(None, Counter) is Counter.value
    counter = Counter()
    assert counter.value == 10
    assert counter.value == 10
    assert counter.calls == 1


def test_election_counts_matrices_and_smith_helpers():
    election = sodaTest.myEc

    assert repr(election).startswith("ElectionCounts(")
    np.testing.assert_array_equal(
        election.appMatrix([1, 2, 3, 4]),
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
    )
    np.testing.assert_array_equal(
        election.oneMatrix([0, 1, 2, 3], size=2),
        [
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 2, 0, 0],
            [2, 2, 2, 0],
        ],
    )
    assert election.matrix[0, 2] == 5

    candidates = [0, 2, 3]
    minwin = [(999,)]
    rival = [(0,)]
    assert list(election.beaters(1, candidates, minwin, rival, private=True)) == [0]
    assert candidates == [2, 3]
    assert minwin[0][1:] == (1, 0)

    assert election.oneWinner(election.matrix) == 1
    assert election.majSmith == [1, 0, 2]
    assert election.minWin[1:] == (0, 2)
    assert election.rival == (0,)


def test_election_counts_debug_and_array_approval_paths(monkeypatch):
    array_approval = ElectionCounts(
        [0, 0],
        np.array([1, 2]),
        [[0, 1], [1, 0]],
        [],
    )
    assert type(array_approval.appr) is sodaTest.arrayType

    monkeypatch.setattr(sodaTest, "DEBUG", False)
    unchecked = ElectionCounts(
        [1],
        np.array([0]),
        [[0]],
        [],
    )
    assert unchecked.n == 1
    assert sodaTest.myEc.delegated([4, 4, 0, 0]).order == [1, 2, 3]


def test_beaters_rival_updates_private_removal_and_default_growth():
    election = ElectionCounts(
        [0, 0, 0],
        [0, 0, 0],
        [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        [],
    )
    election.matrix = np.matrix([[0, 0, 10], [1, 0, 0], [5, 7, 0]])

    candidates = [1]
    rival = [(0,)]
    assert list(election.beaters(0, candidates, [(99,)], rival, private=True)) == [1]
    assert candidates == []
    assert rival[0] == (7, 2, 0, 1)

    assert list(election.beaters(0, [1], [(99,)], [None], private=True)) == [1]
    stale_rival = [(99,)]
    assert list(election.beaters(0, [1], [(99,)], stale_rival, private=True)) == [1]
    assert stale_rival[0] == (99,)
    assert list(election.beaters(0, [1], [(99,)], [(0,)], private=False)) == [1]

    plant = [0]
    election.growFrom(0, plant, [])
    assert plant == [0]


def test_election_counts_delegation_winner_scores_and_shortcuts():
    election = sodaTest.myEc

    delegated = election.delegated([4, 4, 0, 0])
    assert delegated.order == [1, 2, 3]
    assert delegated.winner() == 1
    assert [delegation.tolist() for delegation in election.possibleDelegations(2, 0)] == [
        [4.0, 0.0, 0.0, 0.0],
        [4.0, 4.0, 0.0, 0.0],
        [4.0, 4.0, 4.0, 0.0],
        [4.0, 2.1, 0.0, 0.0],
    ]
    np.testing.assert_array_equal(election.scores(), [18.0, 18.0, 15.0, 3.0])

    leaf = ElectionCounts(
        [0, 0],
        [1, 2],
        [[0, 1], [1, 0]],
        [],
    )
    assert leaf.winner() == 1

    blocked = ElectionCounts(
        election.delg,
        election.appr,
        election.prefs,
        election.order,
        cantWin={0, 1, 2},
    )
    assert blocked.winner() is None


def test_election_counts_verbose_winner_paths(capsys):
    leaf = ElectionCounts(
        [0, 0],
        [1, 2],
        [[0, 1], [1, 0]],
        [],
    )
    assert leaf.winner(verbose=3) == 1
    assert "leafed out" in capsys.readouterr().out

    delegated = sodaTest.myEc.delegated([4, 4, 0, 0])
    assert delegated.winner(verbose=3) == 1
    assert "crystal ball" in capsys.readouterr().out

    expanded = ElectionCounts(
        sodaTest.myEc.delg,
        sodaTest.myEc.appr,
        sodaTest.myEc.prefs,
        sodaTest.myEc.order,
        oldSmith=[1],
    )
    expanded.winner(verbose=1)
    assert "Smith set expanded!" in capsys.readouterr().out

    blocked = ElectionCounts(
        sodaTest.myEc.delg,
        sodaTest.myEc.appr,
        sodaTest.myEc.prefs,
        sodaTest.myEc.order,
        cantWin={0, 1, 2},
    )
    assert blocked.winner(verbose=3) is None
    assert "giving up" in capsys.readouterr().out


def test_winner_loop_verbose_updates_and_continue_paths(capsys):
    class Decision:
        def __init__(self, winner):
            self._winner = winner
            self.matrix = np.eye(3)

        def winner(self, verbose=0):
            return self._winner

    outcomes = iter([None, 1, 0])

    def delegated(amounts, cant_win):
        return Decision(next(outcomes))

    fake = SimpleNamespace(
        order=[0, 1, 2],
        n=3,
        prefs=[[0, 1, 2]],
        majSmith=[0, 1],
        oldSmith=None,
        cantWin=set(),
        possibleDelegations=lambda worst, ideal: [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
        ],
        delegated=delegated,
    )

    assert ElectionCounts.winner(fake, verbose=3) == 0
    output = capsys.readouterr().out
    assert "amounts" in output
    assert "updating" in output
    assert "love it" in output


def test_possible_delegations_without_minwin():
    fake = SimpleNamespace(
        prefs=[[0, 1, 2]],
        order=[0],
        delg=[4, 0, 0],
        n=3,
        minWin=None,
    )

    assert [
        delegation.tolist() for delegation in ElectionCounts.possibleDelegations(fake, 2, 0)
    ] == [
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [4.0, 4.0, 4.0],
    ]


def test_module_level_random_helpers(monkeypatch, capsys):
    monkeypatch.setattr(sodaTest.random, "shuffle", lambda values: values.reverse())
    assert sodaTest.shuffled(4) == [3, 2, 1, 0]

    monkeypatch.setattr(sodaTest.random, "randrange", lambda *args: 0)
    monkeypatch.setattr(sodaTest.random, "random", lambda: 0.5)
    random_election = sodaTest.randomElection(3)
    assert random_election.n == 3

    monkeypatch.setattr(sodaTest, "randomElection", lambda ncand: sodaTest.myEc)
    assert sodaTest.monteCarlo(2) == []
    assert "tick 0" in capsys.readouterr().out

    funky = SimpleNamespace(
        delg=[1],
        appr=[0],
        prefs=[[0]],
        matrix=np.matrix([[0]]),
        majSmith={"inside"},
        winner=lambda: "outside",
    )
    monkeypatch.setattr(sodaTest, "randomElection", lambda ncand: funky)
    assert sodaTest.monteCarlo(1) == [funky]
    assert "Unsmith!!! 0" in capsys.readouterr().out


def test_cached_property_can_be_recomputed_after_deletion():
    class Counter:
        def __init__(self):
            self.calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return SimpleNamespace(call=self.calls)

    counter = Counter()
    first = counter.value
    del counter.value
    second = counter.value

    assert first.call == 1
    assert second.call == 2
