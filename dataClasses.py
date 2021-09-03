
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
import scipy.stats as stats
import scipy.integrate as integrate
from uuid import uuid4


from mydecorators import autoassign, cached_property, setdefaultattr, decorator, uniquify
from debugDump import *
from version import version



from stratFunctions import *


##Election Methods
class Method:
    """Base class for election methods. Holds some of the duct tape."""

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def results(cls, ballots, **kwargs):
        """Combines ballots into results. Override for comparative
        methods.

        Ballots is an iterable of list-or-tuple of numbers (utility) higher is better for the choice of that index.

        Returns a results-array which should be a list of the same length as a ballot with a number (higher is better) for the choice at that index.

        Test for subclasses, makes no sense to test this method in the abstract base class.
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        return list(map(cls.candScore,zip(*ballots)))

    @classmethod
    #@rememberBallot
    def honBallot(cls, utils, **kw):
        """Takes utilities and returns an honest ballot
        """
        raise NotImplementedError("{} needs honBallot".format(cls))

    @classmethod
    def vaBallot(cls, utils, electabilities, **kw):
        """Viability-aware strategy.
        Takes utilities and information on each candidate's electability
        and returns a strategically optimal ballot based on that information
        """
        return cls.honBallot(utils)

    @classmethod
    def abstain(cls, utils, **kw):
        return [0]*len(utils)

    @classmethod
    def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, electabilities=None, polls=None):
        "Returns a ballot using a diehard strategy with the given intensity"
        return cls.honBallot(utils)

    @classmethod
    def compBallot(cls, utils, intensity, candToHelp, candToHurt, electabilities=None, polls=None):
        "Returns a ballot using a compromising strategy with the given intensity"
        return cls.honBallot(utils)

    #lists of which diehard and compromising strategies are available for a voting method
    diehardLevels = []
    compLevels = []

    @classmethod
    def defaultbgs(cls):
        """Returns a list of the default backgrounds (see vse.threeRoundResults) for the voting method.
        These can be individual functions (like cls.honBallot) or (strat, bgargs) tuples
        (like (cls.vaBallot, {'pollingUncertainty': 0.4}))
        """
        return [cls.honBallot, (cls.vaBallot, {'pollingUncertainty': 0.4})]

    @classmethod
    def defaultfgs(cls):
        """Returns a list of the default foregrounds (see vse.threeRoundResults) for the voting method.
        """
        return [(cls.diehardBallot, targs, {'intensity': lev})
                for lev in cls.diehardLevels for targs in [select21, select31]]\
                + [(cls.compBallot, targs, {'intensity': lev})
                for lev in cls.compLevels for targs in [select21, select31]]\
                + [(cls.stratBallot, targs) for targs in [select21, select31]]\
                + [(cls.vaBallot, selectRand, {'info': 'p'}),
                (cls.vaBallot, selectRand, {'info': 'e'}),
                (cls.stratBallot, selectRand, {'info': 'p'}),
                (cls.stratBallot, selectRand, {'info': 'e'})]

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
        #return random.choice(winners)
        return winners[0] #made it deterministic to prevent nondeterministic behaviors in useful functions

    def honBallotFor(self, voters):
        """This is where you would do any setup necessary and create an honBallot
        function. But the base version just returns the honBallot function."""
        return self.honBallot

    def dummyBallotFor(self, polls):
        """Returns a (function which takes utilities and returns a dummy ballot)
        for the given "polling" info."""
        return lambda cls, utilities, stratTally: utilities

    @classmethod
    def stratThresholdSearch(cls, targetWinner, foundAt, bgBallots, fgBallots, fgBaselineBallots, winnersFound):
        """Returns the minimum number of strategist needed to elect targetWinner
        and modifies winnersFound to include any additional winners found during the search"""
        maxThreshold, minThreshold = foundAt, 0
        while maxThreshold > minThreshold: #binary search for min foreground size
            midpoint = int(floor((maxThreshold + minThreshold)/2))
            midpointBallots = bgBallots + fgBallots[:midpoint] + fgBaselineBallots[midpoint:]
            midpointWinner = cls.winner(cls.results(midpointBallots))
            if not any(midpointWinner == w for w, _ in winnersFound):
                winnersFound.append((midpointWinner, midpoint))
            if midpointWinner == targetWinner:
                maxThreshold = midpoint
            else:
                minThreshold = midpoint + 1
        return maxThreshold

    @classmethod
    def resultsFor(cls, voters):
        """Create (honest/naive) ballots and get results.
        Again, test on subclasses.
        """
        return cls.results(list(cls.honBallot(v) for v in voters))
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
    def stratBallot(cls, voter, polls, electabilities=None, info='p', **kw):
        """Returns a strategic (high-info) ballot, using the strategies in version 1
        """
        if info == 'e':
            polls = electabilities
        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        frontId, frontResult, targId, targResult = cls.stratTargetFor(places)
        n = len(polls)
        stratGap = voter[targId] - voter[frontId]
        ballot = [0] * len(voter)
        cls.fillStratBallot(voter, polls, places, n, stratGap, ballot,
                            frontId, frontResult, targId, targResult)
        return ballot

    def stratBallotFor(self,polls):
        """DEPRECATED - use stratBallot instead.
        Returns a (function which takes utilities and returns a strategic ballot)
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
    """DEPRECATED - to be removed
    A decorator for a function of the form xxxBallot(cls, voter)
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
    """DEPRECATED - to be removed
    A decorator for a function of the form xxxBallot(cls, voter)
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

def hybridStrat(voter, stratTuples, **kwForAll):
    """Randomly chooses a strategy and uses it.
    stratTuples is a list of (strategy, probability, kwargs) tuples
    (where kwargs is optional).
    Example:
    b = hybridStrat(Voter([0,9,10]),
            [(Approval.honBallot, 0.4), (Approval.diehardBallot, 0.6, {'intensity': 3})],
            electabilities=[.5,.5,.3], candToHelp=2, candToHurt=1)
    """
    cumProb, i = 0, -1
    r = random.Random(voter.id).random()
    while cumProb < r:
        i += 1
        cumProb += stratTuples[i][1]
    kwForMethod = {} if len(stratTuples[i]) == 2 else stratTuples[i][2]
    return stratTuples[i][0](voter, **kwForAll, **kwForMethod)


@functools.lru_cache(maxsize=10000)
def useStrat(voter, strategy, **kw):
    """Returns the ballot cast by voter using strategy.
    This function exists purely for the sake of memoization.
    """
    return strategy(voter, **kw)

def paramStrat(strategy, **kw):
    """A wrapper for strategy that gives it arguments in addition to voter, polls, electabilities, and targets.
    Incompatible with multithreading; use bgArgs and fgArgs in threeRoundResults instead.
    """
    def strat(voter, polls=None, electabilities=None, candToHelp=None, candToHurt=None):
        return strategy(voter, polls=polls, electabilities=electabilities,
        candToHelp=candToHelp, candToHurt=candToHurt, **kw)
    strat.__name__ = strategy.__name__
    for key, value in kw.items():
        strat.__name__ += "_"+str(key)[:4]+str(value)[:4]
    return strat

def wantToHelp(voter, candToHelp, candToHurt, **kw):
    return max(voter[candToHelp] - voter[candToHurt], 0)

def selectAB(candA, candB): #candA and candB are candidate IDs
    def fgSelect(voter, **kw):
        return max(voter[candA] - voter[candB], 0)
    fgSelect.__name__ = "select"+str(candA)+str(candB)
    return fgSelect

def selectRand(polls, **kw):
    return tuple(random.sample(range(len(polls)), 2))

def select21(polls, **kw):
    pollOrder = [cand for cand, poll in sorted(enumerate(polls),key=lambda x: -x[1])]
    return pollOrder[1], pollOrder[0]

def select31(polls, **kw):
    pollOrder = [cand for cand, poll in sorted(enumerate(polls),key=lambda x: -x[1])]
    return pollOrder[2], pollOrder[0]

def select012(polls, r0polls, **kw):
    pollOrder = [cand for cand, poll in sorted(enumerate(r0polls),key=lambda x: -x[1])]
    return pollOrder[0], pollOrder[1]

def nullTarget(*args, **kw):
    return 0, 0

def selectAll(**kw): return 1

def selectVoter(voter):
    def selectV(v, **kw):
        return 1 if v is voter else 0
    return selectV

resultColumns = ["method", "backgroundStrat", "fgStrat", "numVoters", "numCandidates",
        "magicBestUtil", "magicWorstUtil", "meanCandidateUtil", "r0ExpectedUtil", "r0WinnerUtil",
        "r1WinnerUtil", "probOfWin", "r1WinProb", "winnerPlaceInR0", "winnerPlaceInR1",
        "results", "bgArgs", "fgArgs", "totalUtil", "deciderUtilDiffs", "fgTargets",
        "totalStratUtilDiff", "margStrategicRegret", "avgStrategicRegret",
        "firstDeciderUtilDiff", "deciderUtilDiffSum", "deciderMargUtilDiffs", "numWinnersFound"]
for prefix in ["", "min", "t1", "o"]:
    for columnName in ["fgUtil", "fgUtilDiff", "fgSize",
            "fgNumHelped", "fgHelpedUtil", "fgHelpedUtilDiff",
            "fgNumHarmed", "fgHarmedUtil", "fgHarmedUtilDiff"]:
        resultColumns.append(prefix + columnName)

def makeResults(**kw):
    results = {c: kw.get(c, None) for c in resultColumns}
    results.update(kw)
    return results

def makePartialResults(fgVoters, winner, r1Winner, prefix=""):
    fgHelped = []
    fgHarmed = []
    numfg = len(fgVoters)
    if winner != r1Winner:
        for voter in fgVoters:
            if voter[winner] > voter[r1Winner]:
                fgHelped.append(voter)
            elif voter[winner] < voter[r1Winner]:
                fgHarmed.append(voter)

    tempDict = dict(fgUtil=sum(v[winner] for v in fgVoters)/numfg if numfg else 0,
    fgUtilDiff=sum(v[winner] - v[r1Winner] for v in fgVoters)/numfg if numfg else 0, fgSize=numfg,
    fgNumHelped=len(fgHelped), fgHelpedUtil=sum(v[winner] for v in fgHelped),
    fgHelpedUtilDiff= sum(v[winner] - v[r1Winner] for v in fgHelped),
    fgNumHarmed=len(fgHarmed), fgHarmedUtil=sum(v[winner] for v in fgHarmed),
    fgHarmedUtilDiff= sum(v[winner] - v[r1Winner] for v in fgHarmed))
    return{prefix+key:value for key, value in tempDict.items()}

def simplePollsToProbs(polls, uncertainty=.05):
    """Takes approval-style polling as input i.e. a list of floats in the interval [0,1],
    and returns a list of the estimated probabilities of each candidate winning based on
    uncertainty. Uncertainty is a float that corresponds to the difference in polling
    that's required for one candidate to be twice as viable as another.
    >>> simplePollsToProbs([.5,.4,.4],.1)
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

@functools.lru_cache(maxsize=1000)
def principledPollsToProbs(polls, uncertainty=.15):
    """Takes approval-style polling as input i.e. a list of floats in the interval [0,1],
    and returns a list of the estimated probabilities of each candidate winning based on
    uncertainty. Uncertainty is a float that corresponds to margin of error (2 standard deviations) for
    a candidate that polls exactly 0.5.

    >>> a = principledPollsToProbs((.5,.4,.4),.1); a
    array([0.92336235, 0.03831882, 0.03831882])
    >>> a[1] == a[2]
    True
    >>> b = principledPollsToProbs((.5,.4,.4),.2); b
    array([0.6622815 , 0.16885925, 0.16885925])
    >>> a[1] < b[1]
    True
    >>> b = principledPollsToProbs((.5,.45,.45),.1); b
    array([0.6624054, 0.1687973, 0.1687973])
    >>> a[1] < b[1]
    True
    """
    nonzeroPolls = []
    for poll in polls:
        if poll <= 0:
            nonzeroPolls.append(0.01)
        elif poll >= 1:
            nonzeroPolls.append(0.90)
        else:
            nonzeroPolls.append(poll)
    betaSize =  marginToBetaSize(uncertainty)
    parms = [(betaSize*poll, betaSize*(1-poll)) for poll in nonzeroPolls]
    return multi_beta_probs_of_highest(parms)

def pollsToProbs(polls, uncertainty=.15):
    return principledPollsToProbs(tuple(polls), uncertainty)

def runnerUpProbs(winProbs):
    unnormalizedRunnerUpProbs = [p*(1-p) for p in winProbs]
    normFactor = sum(unnormalizedRunnerUpProbs)
    return [u/normFactor for u in unnormalizedRunnerUpProbs]

def product(l):
    result = 1
    for i in l: result *= i
    return result

@functools.lru_cache(maxsize=1000)
def tieFor2NumIntegration(polls, uncertainty):
    """Takes approval polling as input and returns a list of the estimated "probabilities" of each candidate
    being in a two-way tie for second places, normalized such that the sum is 1.
    """
    n = len(polls)
    tieProbs = [[0]*n for i in range(n)]
    for t1 in range(n):
        for t2 in range(t1+1, n):
            def integrand(x):
                indicesLeft=list(range(t1)) + list(range(t1+1, t2)) + list(range(t2+1, n))
                return (stats.norm.pdf(x, loc=polls[t1], scale=uncertainty)
                *stats.norm.pdf(x, loc=polls[t2], scale=uncertainty)
                *sum(
                product((1 - stats.norm.cdf(x, loc=polls[j], scale=uncertainty))
                if i == j else stats.norm.cdf(x, loc=polls[j], scale=uncertainty)
                for j in indicesLeft)
                for i in indicesLeft))
            tieProbs[t1][t2] = integrate.quad(integrand, 0, 1)[0]
            tieProbs[t2][t1] = tieProbs[t1][t2]
    unnormalizedProbs = [sum(l) for l in tieProbs]
    normFactor = sum(unnormalizedProbs)
    return [u/normFactor for u in unnormalizedProbs]

def tieFor2Probs(polls, uncertainty=.15):
    if len(polls) < 3: return [0]*len(polls)
    return tieFor2NumIntegration(tuple(polls), uncertainty/2)

@functools.lru_cache(maxsize=1000)
def tieFor2Estimate(probs):
    """Estimates the probability of each candidate being in a tie for second place,
    normalized such that they sum to 1"""
    EXP = 2

    unnormalized = [x*(1-x)*(
    sum(sum((y*z)**EXP for k, z in enumerate(probs) if i != k != j)
    for j, y in enumerate(probs) if i != j)
    /sum(y**EXP for j, y in enumerate(probs) if i != j)
    )**(1/EXP) for i, x in enumerate(probs)]

    normFactor = sum(unnormalized)
    return [u/normFactor for u in unnormalized]


def adaptiveTieFor2(polls, uncertainty=.15):
    if False and len(polls) < 6:
        return tieFor2Probs(polls, uncertainty)
    else:
        return tieFor2Estimate(tuple(pollsToProbs(polls, uncertainty)))

def appendResults(filename, resultsList, globalComment = dict()):
    """append list of results created by makeResults to a csv file.
    for instance:

    csvs.saveFile()
    """
    needsHeader = not os.path.isfile(baseName)
    keys = resultsList[0].keys()  # important stuff first
    #keys.extend(list(self.rows[0].keys()))  # any other stuff I missed; dedup later
    keys = uniquify(keys)


    globalComment(version = version)

    with open(baseName + str(i) + ".csv", "a") as myFile:
        if needsHeader:
            print("# " + str(globalComment),
            #dict(
                        #media=self.media.__name__,
            #              version=self.repo_version,
            #              seed=self.seed,
            ##              model=self.model,
            #              methods=self.methods,
            #              nvot=self.nvot,
            #              ncand=self.ncand,
            #              niter=self.niter)),
                file=myFile)
        dw = csv.DictWriter(myFile, keys, restval="NA")
        dw.writeheader()
        for r in resultsList:
            dw.writerow(r)
