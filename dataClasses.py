
from mydecorators import autoassign, cached_property, setdefaultattr
import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor
from test.test_binop import isnum
from debugDump import *
from uuid import uuid4

from stratFunctions import *

class VseOneRun:
    @autoassign
    def __init__(self, result, tally, strat):
        pass

class VseMethodRun:
    @autoassign
    def __init__(self, method, choosers, results):
        pass


####data holders for output
from collections import defaultdict
class SideTally(defaultdict):
    """Used for keeping track of how many voters are being strategic, etc.

    DO NOT use plain +; for this class, it is equivalent to +=, but less readable.

    """
    def __init__(self):
        super().__init__(int)
    #>>> tally = SideTally()
    #>>> tally += {1:2,3:4}
    #>>> tally
    #{1: 2, 3: 4}
    #>>> tally += {1:2,3:4,5:6}
    #>>> tally
    #{1: 4, 3: 8, 5: 6}
    #"""
    #def __add__(self, other):
    #    for (key, val) in other.items():
    #        try:
    #            self[key] += val
    #        except KeyError:
    #            self[key] = val
    #    return self

    def initKeys(self, chooser):
        try:
            self.keyList = chooser.allTallyKeys()
        except AttributeError:
            try:
                self.keyList = list(chooser)
            except TypeError:
                pass
                #TODO: Why does this happen?
                #debug("Chooser has no tally keys:", str(chooser))
        self.initKeys = staticmethod(lambda x:x) #don't do it again

    def serialize(self):
        try:
            return [self[key] for key in self.keyList]
        except AttributeError:
            return []

    def fullSerialize(self):
        try:
            return [self[key] for key in self.keys()]
        except AttributeError:
            return []

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
        try:
            self.used
            return super().__iter__()
        except:
            self.used = True
            return self

    def __next__(self):
        tally = SideTally()
        self.append(tally)
        return tally

##Election Methods
class Method:
    """Base class for election methods. Holds some of the duct tape."""

    def __str__(self):
        return self.__class__.__name__

    def results(self, ballots, isHonest):
        """Combines ballots into results. Override for comparative
        methods.

        Test for subclasses, makes no sense to test this method in the abstract base class.
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        return list(map(self.candScore,zip(*ballots)))

    @staticmethod
    def winner(results):
        """Simply find the winner once scores are already calculated. Override for
        ranked methods.


        >>> Method().winner([1,2,3,2,-100])
        2
        >>> 2 < Method().winner([1,2,1,3,3,3,2,1,2]) < 6
        True
        """
        winScore = max([result for result in results if isnum(result)])
        winners = [cand for (cand, score) in enumerate(results) if score==winScore]
        return random.choice(winners)

    def honBallotFor(self, voters):
        """This is where you would do any setup necessary and create an honBallot
        function. But the base version just returns the honBallot function."""
        return self.honBallot

    def resultsFor(self, voters, chooser, tally=None, isHonest=False):
        """create ballots and get results.

        Again, test on subclasses.
        """
        if tally is None:
            tally = SideTally()
        tally.initKeys(chooser)
        return (self.results([chooser(self.__class__, voter, tally)
                                  for voter in voters],
                              isHonest), chooser.__name__)

    def multiResults(self, voters, chooserFuns=(), media=lambda x,t:x):
        """Runs two base elections: first with honest votes, then
        with strategic results based on the first results (filtered by
        the media). Then, runs a series of elections using each chooserFun
        in chooserFuns to select the votes for each voter.

        Returns a tuple of (honResults, stratResults, ...). The stratresults
        are based on common information, which is given by media(honresults).
        """
        honTally = SideTally()
        self.__class__.extraEvents = dict()
        hon = self.resultsFor(voters, self.honBallotFor(voters), honTally, True)

        stratTally = SideTally()
        info = media(hon[0], stratTally)
        strat = self.resultsFor(voters, self.stratBallotFor(info), stratTally)
        extraTallies = Tallies()
        extras = [self.resultsFor(voters, self.ballotChooserFor(chooserFun), aTally)
                  for (chooserFun, aTally) in zip(chooserFuns, extraTallies)]
        return [([(hon, honTally), (strat, stratTally)] +
                list(zip(extras, list(extraTallies)))),
                list(self.__class__.extraEvents.items())]

    def vseOn(self, voters, chooserFuns=(), **args):
        """Finds honest and strategic voter satisfaction efficiency (VSE)
        for this method on the given electorate.
        """
        multiResults = self.multiResults(voters, chooserFuns, **args)
        utils = voters.socUtils
        best = max(utils)
        rand = mean(utils)

        #import pprint
        #pprint.pprint(multiResults)
        vses = VseMethodRun(self.__class__, chooserFuns,
                    [VseOneRun([(utils[self.winner(result)] - rand) / (best - rand)],tally,chooser)
                        for ((result, chooser), tally) in multiResults[0]])
        vses.extraEvents=multiResults[1]
        return vses

    def resultsTable(self, eid, emodel, cands, voters, chooserFuns=(), **args):
        multiResults = self.multiResults(voters, chooserFuns, **args)
        utils = voters.socUtils
        best = max(utils)
        rand = mean(utils)
        rows = list()
        nvot=len(voters)
        for ((result, chooser), tally) in multiResults[0]:
            row = {
                "eid":eid,
                "emodel":emodel,
                "ncand":cands,
                "nvot":nvot,
                "best":best,
                "rand":rand,
                "method":str(self),
                "chooser":chooser,#.getName(),
                "util":utils[self.winner(result)],
                "vse":(utils[self.winner(result)] - rand) / (best - rand)
            }
            #print(tally)
            for (i, (k, v)) in enumerate(tally.items()):
                #print("Result: tally ",i,k,v)
                row["tallyName"+str(i)] = str(k)
                row["tallyVal"+str(i)] = str(v)
            rows.append(row)
        if len(multiResults[1]):
            row = {
                "eid":eid,
                "emodel":emodel,
                "method":self.__class__.__name__,
                "chooser":"extraEvents",
                "util":None
            }
            for (i, (k, v)) in enumerate(multiResults[1]):
                row["tallyName"+str(i)] = str(k)
                row["tallyVal"+str(i)] = str(v)
            rows.append(row)
        return(rows)


    @staticmethod
    def ballotChooserFor(chooserFun):
        """Takes a chooserFun; returns a ballot chooser using that chooserFun
        """
        def ballotChooser(cls, voter, tally):
            return getattr(voter, cls.__name__ + "_" + chooserFun(cls, voter, tally))
        ballotChooser.__name__ = chooserFun.getName()
        return ballotChooser

def rememberBallot(fun):
    """A decorator for a function of the form xxxBallot(cls, voter)
    which memoizes the vote onto the voter in an attribute named <methName>_xxx
    """
    def getAndRemember(cls, voter, tally=None):
        ballot = fun(cls, voter)
        setattr(voter, cls.__name__ + "_" + fun.__name__[:-6], ballot) #leave off the "...Ballot"
        return ballot
    getAndRemember.__name__ = fun.__name__
    getAndRemember.allTallyKeys = lambda:[]
    return getAndRemember

def rememberBallots(fun):
    """A decorator for a function of the form xxxBallot(cls, voter)
    which memoizes the vote onto the voter in an attribute named <methName>_xxx
    """
    def getAndRemember(cls, voter, tally=None):
        ballots = fun(cls, voter)
        for bType, ballot in ballots.items():

            setattr(voter, cls.__name__ + "_" + bType, ballot)

        return ballots[fun.__name__[:-6]] #leave off the "...Ballot"
    getAndRemember.__name__ = fun.__name__
    getAndRemember.allTallyKeys = lambda:[]
    return getAndRemember
