
from mydecorators import autoassign, cached_property, setdefaultattr, decorator
import random
import functools
import numpy as np
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor
from scipy.stats import beta
from test.test_binop import isnum
from scipy.optimize import fmin
from debugDump import *
from uuid import uuid4


from stratFunctions import *

class VseOneRun:
    @autoassign
    def __init__(self, result, tallyItems, strat):
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
            kl = self.keyList
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
class BaseMethod:
    """Base class for election methods. Holds some of the duct tape."""

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def results(self, ballots, **kwargs):
        """Combines ballots into results. Override for comparative
        methods.

        Ballots is an iterable of list-or-tuple of numbers (utility) higher is better for the choice of that index.

        Returns a results-array which should be a list of the same length as a ballot with a number (higher is better) for the choice at that index.

        Test for subclasses, makes no sense to test this method in the abstract base class.
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        return list(map(self.candScore,zip(*ballots)))

    @staticmethod #cls is provided explicitly, not through binding
    #@rememberBallot
    def honBallot(cls, utils):
        """Takes utilities and returns an honest ballot
        """
        raise NotImplementedError("{} needs honBallot".format(cls))

    @classmethod
    def lowInfoBallot(cls, utils, winProbs):
        """Takes utilities and information on each candidate's electability
        and returns a strategically optimal ballot based on that information
        """
        raise NotImplementedError("{} lacks lowInfoBallot".format(cls))

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

    def dummyBallotFor(self, polls):
        """Returns a (function which takes utilities and returns a dummy ballot)
        for the given "polling" info."""
        return lambda cls, utilities, stratTally: utilities

    def resultsFor(self, voters, chooser, tally=None, **kwargs):
        """create ballots and get results.

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

        Returns a tuple of (honResults, stratResults, ...). The stratresults
        are based on common polling information, which is given by media(honresults).
        """
        from stratFunctions import OssChooser

        honTally = SideTally()
        self.__class__.extraEvents = dict()
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
                    for (chooserFun, aTally) in zip(chooserFuns, extraTallies)]
                  )
        return ([(hon["results"], hon["chooser"],
                        list(self.__class__.extraEvents.items()))]  +
                [(r["results"], r["chooser"], r["tally"].itemList()) for r in results])

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
                        for (result, chooser, tally) in multiResults[0]])
        vses.extraEvents=multiResults[1]
        return vses

    def resultsTable(self, eid, emodel, cands, voters, chooserFuns=(), **args):
        multiResults = self.multiResults(voters, chooserFuns, **args)
        utils = voters.socUtils
        best = max(utils)
        rand = mean(utils)
        rows = list()
        nvot=len(voters)
        for (result, chooser, tallyItems) in multiResults:
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
            #print(tallyItems)
            for (i, (k, v)) in enumerate(tallyItems):
                #print("Result: tally ",i,k,v)
                row["tallyName"+str(i)] = str(k)
                row["tallyVal"+str(i)] = str(v)
            rows.append(row)
        # if len(multiResults[1]):
        #     row = {
        #         "eid":eid,
        #         "emodel":emodel,
        #         "method":self.__class__.__name__,
        #         "chooser":"extraEvents",
        #         "util":None
        #     }
        #     for (i, (k, v)) in enumerate(multiResults[1]):
        #         row["tallyName"+str(i)] = str(k)
        #         row["tallyVal"+str(i)] = str(v)
        #     rows.append(row)
        return(rows)

    @staticmethod
    def ballotChooserFor(chooserFun):
        """Takes a chooserFun; returns a ballot chooser using that chooserFun
        """
        def ballotChooser(cls, voter, tally):
            return getattr(voter, cls.__name__ + "_" + chooserFun(cls, voter, tally))
        ballotChooser.__name__ = chooserFun.getName()
        return ballotChooser

    @staticmethod
    def stratTarget2(places):
        ((frontId,frontResult), (targId, targResult)) = places[0:2]
        return (frontId, frontResult, targId, targResult)

    @staticmethod
    def stratTarget3(places):
        ((frontId,frontResult), (targId, targResult)) = places[0:3:2]
        return (frontId, frontResult, targId, targResult)

    stratTargetFor = stratTarget2

    @classmethod
    def stratBallot(cls, voter, polls, electabilities=None):
        """Returns a strategic (high-info) ballot, using the strategies in version 1
        """
        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        frontId, frontResult, targId, targResult = cls.stratTargetFor(places)
        n = len(polls)
        stratGap = voter[targId] - voter[frontId]
        ballot = [0] * len(voter)
        cls.fillStratBallot(voter, polls, places, n, stratGap, ballot,
                            frontId, frontResult, targId, targResult)
        return ballot

    def stratBallotFor(self,polls):
        """Returns a (function which takes utilities and returns a strategic ballot)
        for the given "polling" info."""

        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        #print("places",places)
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
    which memoizes the vote onto the voter in an attribute named <methName>_xxx
    """
    def getAndRemember(cls, voter, tally=None):
        ballot = fun(cls, voter)
        setattr(voter, cls.__name__ + "_" + fun.__name__[:-6], ballot) #leave off the "...Ballot"
        return ballot
    getAndRemember.__name__ = fun.__name__
    getAndRemember.allTallyKeys = lambda:[]
    return getAndRemember

@decorator
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

@functools.lru_cache(maxsize=10000)
def useStrat(voter, strategy, **kw):
    """Returns the ballot cast by voter using strategy.
    This function exists purely for the sake of memoization.
    """
    return strategy(voter, **kw)

def paramStrat(strategy, **kw):
    """A wrapper for strategy that gives it arguments in addition to voter, polls, and electabilities
    """
    def strat(voter, polls=None, electabilities=None):
        return strategy(voter, polls=polls, electabilities=electabilities, **kw)
    strat.__name__ = strategy.__name__
    for key, value in kw.items():
        strat.__name__ += "_"+str(key)[:4]+str(value)[:4]
    return strat

def makeResults(method, backgroundStrat=None, fgSrat=None, numVoters=None,
magicBestUtil=None, magicWorstUtil=None, meanCandidateUtil=None, r0ExpectedUtil=None, r0WinnerUtil=None,
r1WinnerUtil=None, probOfWin=None, winnerPlaceInR0=None, winnerPlaceInR1=None,
results=None, totalUtil=None,
fgUtil=None, fgUtilDiff=None, fgSize=None,
fgNumHelped=None, fgHelpedUtil=None, fgHelpedUtilDiff=None,
fgNumHarmed=None, fgHarmedUtil=None, fgHarmedUtilDiff=None,
#minfg is the smallest foreground that can change the outcome to the that when the whole foreground is strategic
#such that every member of this foreground is more eager to strategize than any voter outside it
minfgUtil=None, minfgUtilDiff=None, minfgSize=None,
minfgNumHelped=None, minfgHelpedUtil=None, minfgHelpedUtilDiff=None,
minfgNumHarmed=None, minfgHarmedUtil=None, minfgHarmedUtilDiff=None,
#t1fg is the smallest foreground that yield a different result than in r1 (pursuant to the eagerness limitation)
t1fgUtil=None, t1fgUtilDiff=None, t1fgSize=None,
t1fgNumHelped=None, t1fgHelpedUtil=None, t1fgHelpedUtilDiff=None,
t1fgNumHarmed=None, t1fgHarmedUtil=None, t1fgHarmedUtilDiff=None,
numWinnersFound=None
):
    return dict(method=method, backgroundStrat=backgroundStrat, fgSrat=fgSrat, numVoters=numVoters,
    magicBestUtil=magicBestUtil, magicWorstUtil=magicWorstUtil, meanCandidateUtil=meanCandidateUtil,
    r0ExpectedUtil=r0ExpectedUtil, r0WinnerUtil=r0WinnerUtil,
    r1WinnerUtil=r1WinnerUtil, probOfWin=probOfWin,
    winnerPlaceInR0=winnerPlaceInR0, winnerPlaceInR1=winnerPlaceInR1,
    results=results, totalUtil=totalUtil,
    fgUtil=fgUtil, fgUtilDiff=fgUtilDiff, fgSize=fgSize,
    fgNumHelped=fgNumHelped, fgHelpedUtil=fgHelpedUtil, fgHelpedUtilDiff=fgHelpedUtilDiff,
    fgNumHarmed=fgNumHarmed, fgHarmedUtil=fgHarmedUtil, fgHarmedUtilDiff=fgHarmedUtilDiff,
    minfgUtil=minfgUtil, minfgUtilDiff=minfgUtilDiff, minfgSize=minfgSize,
    minfgNumHelped=minfgNumHelped, minfgHelpedUtil=minfgHelpedUtil, minfgHelpedUtilDiff=minfgHelpedUtilDiff,
    minfgNumHarmed=minfgNumHarmed, minfgHarmedUtil=minfgHarmedUtil, minfgHarmedUtilDiff=minfgHarmedUtilDiff,
    t1fgUtil=t1fgUtil, t1fgUtilDiff=t1fgUtilDiff, t1fgSize=t1fgSize,
    t1fgNumHelped=t1fgNumHelped, t1fgHelpedUtil=t1fgHelpedUtil, t1fgHelpedUtilDiff=t1fgHelpedUtilDiff,
    t1fgNumHarmed=t1fgNumHarmed, t1fgHarmedUtil=t1fgHarmedUtil, t1fgHarmedUtilDiff=t1fgHarmedUtilDiff,
    numWinnersFound=numWinnersFound)

def simplePollsToProbs(polls, uncertainty=.05):
    """Takes approval-style polling as input i.e. a list of floats in the interval [0,1],
    and returns a list of the estimated probabilities of each candidate winning based on
    uncertainty. Uncertainty is a float that corresponds to the difference in polling
    that's required for one candidate to be twice as viable as another.
    >>> pollsToProbs([.5,.4,.4],.1)
    [0.5, 0.25, 0.25]
    """
    unnormalizedProbs = [2**(pollResult/uncertainty) for pollResult in polls]
    normFactor = sum(unnormalizedProbs)
    return [p/normFactor for p in unnormalizedProbs]

def marginToBetaSize(twosig):
    """Takes x, and returns a sample size n such that 2*stdev(beta(n/2,n/2))=x"""
    return (1 / (twosig**2)) - 1

def multi_beta_probs_of_highest(parms):
    """Given a list of beta distribution params, returns a quick-and-dirty
    approximation of the chance that each respective beta is the max across
    all of them."""

    betas = [beta(*parm) for parm in parms]
    def multi_beta_cdf_loss(x):
        """Given x, return the loss which (when minimized) leads x to be
        the median of the max when drawing one sample from each of the betas."""

        p = 1.
        for beta in betas:
            p = p * (beta.cdf(x))
        return (0.5-p)**2

    res = fmin(multi_beta_cdf_loss, np.array([.5]), disp=False)[0]

    probs = np.array([1-beta.cdf(res) for beta in betas])
    probs = probs / np.sum(probs)
    return probs

def principledPollsToProbs(polls, uncertainty=.05):
    """Takes approval-style polling as input i.e. a list of floats in the interval [0,1],
    and returns a list of the estimated probabilities of each candidate winning based on
    uncertainty. Uncertainty is a float that corresponds to margin of error (2 standard deviations) for
    a candidate that polls exactly 0.5.

    >>> a = principledPollsToProbs([.5,.4,.4],.1); a
    array([0.92336235, 0.03831882, 0.03831882])
    >>> a[1] == a[2]
    True
    >>> b = principledPollsToProbs([.5,.4,.4],.2); b
    array([0.6622815 , 0.16885925, 0.16885925])
    >>> a[1] < b[1]
    True
    >>> b = principledPollsToProbs([.5,.45,.45],.1); b
    array([0.6624054, 0.1687973, 0.1687973])
    >>> a[1] < b[1]
    True
    """
    betaSize =  marginToBetaSize(uncertainty)
    parms = [(betaSize*poll, betaSize*(1-poll)) for poll in polls]
    return multi_beta_probs_of_highest(parms)

pollsToProbs = principledPollsToProbs
