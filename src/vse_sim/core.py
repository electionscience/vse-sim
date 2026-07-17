
import random
from collections import defaultdict
from dataclasses import dataclass, field

from numpy import isclose, mean

from .decorators import autoassign, decorator


def isnum(x):
    """Test whether an object is an instance of a built-in numeric type."""
    return next((1 for T in (int, float, complex) if isinstance(x, T)), 0)


class VseOneRun:
    @autoassign
    def __init__(self, result, tallyItems, strat):
        pass

class VseMethodRun:
    @autoassign
    def __init__(self, method, choosers, results):
        pass


@dataclass
class ElectionContext:
    """Mutable metadata scoped to one method instance and election."""

    extra_events: dict = field(default_factory=dict)


def normalized_vse(utility, best, random_baseline):
    """Normalize utility to VSE, including a tied-utility electorate.

    When every candidate has the same social utility there is no possible
    improvement over random selection, so every method receives neutral VSE.

    >>> normalized_vse(2, 2, 2)
    0.0
    >>> normalized_vse(3, 3, 1)
    1.0
    """
    denominator = best - random_baseline
    if isclose(denominator, 0):
        return 0.0
    return (utility - random_baseline) / denominator


class SideTally(defaultdict):
    """Used for keeping track of how many voters are being strategic, etc.

    DO NOT use plain +; for this class, it is equivalent to +=, but less readable.

    """
    def __init__(self):
        super().__init__(int)
        self._keys_initialized = False

    def initKeys(self, chooser):
        if self._keys_initialized:
            return
        try:
            self.keyList = chooser.allTallyKeys()
        except AttributeError:
            try:
                self.keyList = list(chooser)
            except TypeError:
                pass
        self._keys_initialized = True

    def serialize(self):
        try:
            return [self[key] for key in self.keyList]
        except AttributeError:
            return []

    def fullSerialize(self):
        try:
            return ([self[key] for key in self.keyList] +
                    [self[key] for key in self.keys() if key not in self.keyList])
        except AttributeError:
            return [self[key] for key in self.keys()]

    def itemList(self):
        try:
            kl = self.keyList
            return ([(k, self[k]) for k in kl] +
                    [(k, self[k]) for k in self.keys() if k not in kl])
        except AttributeError:
            return list(self.items())

class Tallies(list):
    """Used (ONCE) as an enumerator, gives an inexhaustible flow of SideTally objects.
    After that, use as list to see those objects.

    >>> ts = Tallies()
    >>> for i, j in zip(ts, [5,4,3]):
    ...     i[j] += j
    ...
    >>> [t.serialize() for t in ts]
    [[], [], [], []]
    >>> [t.fullSerialize() for t in ts]
    [[5], [4], [3], []]
    >>> [t.initKeys([k]) for (t,k) in zip(ts,[6,4,3])]
    [None, None, None]
    >>> [t.serialize() for t in ts]
    [[0], [4], [3], []]
    """
    def __iter__(self):
        if hasattr(self, "used"):
            return super().__iter__()
        self.used = True
        return self

    def __next__(self):
        tally = SideTally()
        self.append(tally)
        return tally

# Election methods
class Method:
    """Base class for election methods. Holds some of the duct tape."""

    def __init__(self):
        self.context = ElectionContext()

    @property
    def extraEvents(self):
        """Compatibility view of metadata for this method's current election."""
        return self.context.extra_events

    @extraEvents.setter
    def extraEvents(self, value):
        self.context.extra_events = value

    def __str__(self):
        return self.__class__.__name__

    def results(self, ballots, **kwargs):
        """Combines ballots into results. Override for comparative
        methods.

        Ballots is an iterable of list-or-tuple of numbers (utility) higher is better for the choice of that index.

        Returns a results-array which should be a list of the same length as a ballot with a number (higher is better) for the choice at that index.

        Test for subclasses, makes no sense to test this method in the abstract base class.
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        return list(map(self.candScore,zip(*ballots, strict=False)))

    @staticmethod #cls is provided explicitly, not through binding
    def honBallot(cls, utils):
        """Takes utilities and returns an honest ballot.
        """
        raise NotImplementedError(f"{cls} needs honBallot")

    @staticmethod
    def winner(results):
        """Simply find the winner once scores are already calculated. Override for
        ranked methods.


        >>> Method().winner([1,2,3,2,-100])
        2
        >>> 2 < Method().winner([1,2,1,3,3,3,2,1,2]) < 6
        True
        """
        winScore = max(result for result in results if isnum(result))
        winners = [cand for (cand, score) in enumerate(results) if score==winScore]
        return random.choice(winners)

    def honBallotFor(self, voters):
        """This is where you would do any setup necessary and create an honBallot
        function. But the base version just returns the honBallot function."""
        return self.honBallot

    def dummyBallotFor(self, polls):
        """Returns a (function which takes utilities and returns a dummy ballot)
        for the given "polling" info."""
        return lambda cls, utilities, stratTally: utilities

    def resultsFor(self, voters, chooser, tally=None, **kwargs):
        """Create ballots and get results.

        Again, test on subclasses.
        """
        if tally is None:
            tally = SideTally()
        tally.initKeys(chooser)
        return dict(results=self.results([chooser(self.__class__, voter, tally)
                                  for voter in voters],
                              **kwargs),
                chooser=chooser.__name__,
                tally=tally)

    def multiResults(self, voters, chooserFuns=(), media=(lambda x,t:x),
                checkStrat = True):
        """Runs two base elections: first with honest votes, then
        with strategic results based on the first results (filtered by
        the media). Then, runs a series of elections using each chooserFun
        in chooserFuns to select the votes for each voter.

        Returns a flat list of ``(result, chooser, tally_items)`` tuples for
        the honest, strategic, one-sided strategic, smart one-sided, and
        caller-provided chooser runs. Honest-run ``tally_items`` contain the
        election's extra event metadata. Strategic results use common polling
        information produced by ``media(honest_results)``.
        """
        from .strategies import OssChooser

        honTally = SideTally()
        self.context = ElectionContext()
        hon = self.resultsFor(voters, self.honBallotFor(voters), honTally, isHonest=True)

        stratTally = SideTally()

        polls = media(hon["results"], stratTally)
        winner, _w, target, _t = self.stratTargetFor(sorted(enumerate(polls),key=lambda x:-x[1]))

        strat = self.resultsFor(voters, self.stratBallotFor(polls), stratTally)

        ossTally = SideTally()
        oss = self.resultsFor(voters, self.ballotChooserFor(OssChooser()), ossTally)
        ossWinner = oss["results"].index(max(oss["results"]))
        ossTally["worked"] += (1 if ossWinner==target else
                                    (0 if ossWinner==winner else -1))

        smart = dict(results=(hon["results"]
                                    if ossTally["worked"] == 1
                                else oss["results"]),
                chooser="smartOss",
                tally=SideTally())

        extraTallies = Tallies()
        results = ([strat, oss, smart] +
                [self.resultsFor(voters, self.ballotChooserFor(chooserFun), aTally)
                    for (chooserFun, aTally) in zip(chooserFuns, extraTallies, strict=False)]
                  )
        return ([(hon["results"], hon["chooser"],
                        list(self.extraEvents.items()))]  +
                [(r["results"], r["chooser"], r["tally"].itemList()) for r in results])

    def vseOn(self, voters, chooserFuns=(), **args):
        """Finds honest and strategic voter satisfaction efficiency (VSE)
        for this method on the given electorate.
        """
        multiResults = self.multiResults(voters, chooserFuns, **args)
        utils = voters.socUtils
        best = max(utils)
        rand = mean(utils)

        vses = VseMethodRun(
            self.__class__,
            chooserFuns,
            [
                VseOneRun(
                    [normalized_vse(utils[self.winner(result)], best, rand)],
                    tally,
                    chooser,
                )
                for result, chooser, tally in multiResults
            ],
        )
        vses.extraEvents = dict(self.extraEvents)
        return vses

    def resultsTable(self, eid, emodel, cands, voters, chooserFuns=(), **args):
        multiResults = self.multiResults(voters, chooserFuns, **args)
        utils = voters.socUtils
        best = max(utils)
        rand = mean(utils)
        rows = []
        nvot=len(voters)
        for (result, chooser, tallyItems) in multiResults:
            winner = self.winner(result)
            utility = utils[winner]
            row = {
                "eid":eid,
                "emodel":emodel,
                "ncand":cands,
                "nvot":nvot,
                "best":best,
                "rand":rand,
                "method":str(self),
                "chooser":chooser,#.getName(),
                "util":utility,
                "vse":normalized_vse(utility, best, rand)
            }
            for (i, (k, v)) in enumerate(tallyItems):
                row[f"tallyName{str(i)}"] = str(k)
                row[f"tallyVal{str(i)}"] = str(v)
            rows.append(row)
        return(rows)


    @staticmethod
    def ballotChooserFor(chooserFun):
        """Takes a chooserFun; returns a ballot chooser using that chooserFun.
        """
        def ballotChooser(cls, voter, tally):
            return getattr(voter, f"{cls.__name__}_{chooserFun(cls, voter, tally)}")

        ballotChooser.__name__ = chooserFun.getName()
        return ballotChooser

    def stratTarget2(self,places):
        ((frontId,frontResult), (targId, targResult)) = places[:2]
        return (frontId, frontResult, targId, targResult)

    def stratTarget3(self,places):
        ((frontId,frontResult), (targId, targResult)) = places[:3:2]
        return (frontId, frontResult, targId, targResult)

    stratTargetFor = stratTarget2

    def stratBallotFor(self,polls):
        """Returns a (function which takes utilities and returns a strategic ballot)
        for the given "polling" info."""

        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        (frontId, frontResult, targId, targResult) = self.stratTargetFor(places)
        n = len(polls)
        @rememberBallots
        def stratBallot(cls, voter):
            stratGap = voter[targId] - voter[frontId]
            ballot = [0] * len(voter)
            isStrat = stratGap > 0
            extras = cls.fillStratBallot(voter, polls, places, n, stratGap, ballot,
                                frontId, frontResult, targId, targResult)
            result =  dict(strat=ballot, isStrat=isStrat, stratGap=stratGap)
            if extras:
                result.update(extras)
            return result
        return stratBallot

@decorator
def rememberBallot(fun):
    """A decorator for a function of the form xxxBallot(cls, voter)
    which memoizes the vote onto the voter in an attribute named <methName>_xxx.
    """
    def getAndRemember(cls, voter, tally=None):
        ballot = fun(cls, voter)
        setattr(voter, f"{cls.__name__}_{fun.__name__[:-6]}", ballot)
        return ballot

    getAndRemember.__name__ = fun.__name__
    getAndRemember.allTallyKeys = lambda:[]
    return getAndRemember

@decorator
def rememberBallots(fun):
    """A decorator for a function of the form xxxBallot(cls, voter)
    which memoizes the vote onto the voter in an attribute named <methName>_xxx.
    """
    def getAndRemember(cls, voter, tally=None):
        ballots = fun(cls, voter)
        for bType, ballot in ballots.items():

            setattr(voter, f"{cls.__name__}_{bType}", ballot)

        return ballots[fun.__name__[:-6]] #leave off the "...Ballot"

    getAndRemember.__name__ = fun.__name__
    getAndRemember.allTallyKeys = lambda:[]
    return getAndRemember

class CandidateWithCount:
    def __init__(self, c = [], v = 0):
        self.candidate = c
        self.votes = v