
from mydecorators import autoassign, cached_property, setdefaultattr, decorator
import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor, ceil
from numpy import percentile, argsort, sign
from test.test_binop import isnum
from debugDump import *
from math import log, nan

from stratFunctions import *
from dataClasses import *


class Borda(Method):
    candScore = staticmethod(mean)

    nRanks = 999 # infinity

    @classmethod
    def results(cls, ballots, **kwargs):
        """
        >>> Borda.results([[3,2,1,0]]*5+[[0,3,2,1]]*2)
        [0.5357142857142857, 0.5714285714285714, 0.32142857142857145, 0.07142857142857142]
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        n = len(ballots[0])
        return list(map(lambda x: mean(x)/n,zip(*ballots)))

    @staticmethod
    def fillPrefOrder(voter, ballot,
            whichCands=None, #None means "all"; otherwise, an iterable of cand indexes
            lowSlot=0,
            nSlots=None, #again, None means "all"
            remainderScore=None #what to give candidates that don't fit in nSlots
            ):

        venum = list(enumerate(voter))
        if whichCands:
            venum = [venum[c] for c in whichCands]
        prefOrder = sorted(venum,key=lambda x:-x[1]) #high to low
        Borda.fillCands(ballot, prefOrder, lowSlot, nSlots, remainderScore)
        #modifies ballot argument, returns nothing.

    @staticmethod
    def fillCands(ballot,
            whichCands, #list of tuples starting with cand id, in descending order
            lowSlot=0,
            nSlots=None, #again, None means "all"
            remainderScore=None #what to give candidates that don't fit in nSlots
            ):
        if nSlots is None:
            nSlots = len(whichCands)
        cur = lowSlot + nSlots - 1
        for i in range(nSlots):
            ballot[whichCands[i][0]] = cur
            cur -= 1
        if remainderScore is not None:
            i += 1
            while i < len(whichCands):
                ballot[whichCands[i][0]] = remainderScore
                i += 1
        #modifies ballot argument, returns nothing.

    @classmethod
    def honBallot(cls, utils, **kw):
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot)
        return ballot

    @classmethod
    def vaBallot(cls, utils, electabilities, polls=None, winProbs=None,
    pollingUncertainty=.3, info='e', **kw):
        """Uses a mix of compromising and burial.

        >>> Borda.vaBallot((0,1,2,3), [.2, .4, .4, .2])
        [1, 0, 3, 2]
        """
        if info == 'p':
            electabilities = polls
        if not winProbs:
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
        expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
        scores = [(u - expectedUtility)*p for u, p in zip(utils, winProbs)]
        ballot = [0] * len(utils)
        cls.fillPrefOrder(scores, ballot)
        return ballot

    @classmethod
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):
        """Mutates the `ballot` argument to be a strategic ballot.

        >>> Borda.stratBallot(Voter([-4,-5,-2,-1]), [4,5,2,1])
        [3, 0, 1, 2]
        """
        nRanks = min(cls.nRanks,n)
        if stratGap <= 0:
            ballot[frontId], ballot[targId] = (nRanks - 1), 0
        else:
            ballot[frontId], ballot[targId] = 0, (nRanks - 1)
        nRanks -= 2
        if nRanks > 0:
            cls.fillCands(ballot, places[2:][::-1],
                lowSlot=1, nSlots=nRanks, remainderScore=0)
        # (don't) return dict(strat=ballot, isStrat=isStrat, stratGap=stratGap)

RankedMethod = Borda #alias
RatedMethod = RankedMethod #Should have same strategies available, plus more

class Plurality(RankedMethod):

    nRanks = 2
    compLevels = [3]

    @classmethod
    def results(cls, ballots, **kwargs):
        """
        >>> Plurality.results([[1,0]]*3+[[0,1]]*2)
        [0.6, 0.4]
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        return list(map(mean,zip(*ballots)))

    @staticmethod
    def oneVote(utils, forWhom):
        ballot = [0] * len(utils)
        ballot[forWhom] = 1
        return ballot

    @classmethod
    def honBallot(cls, utils, **kw):
        """Takes utilities and returns an honest ballot

        >>> Plurality.honBallot(Voter([-3,-2,-1]))
        [0, 0, 1]
        """
        #return cls.oneVote(utils, cls.winner(utils))
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot,
            nSlots = 1, lowSlot=1, remainderScore=0)
        return ballot

    @classmethod
    def vaBallot(cls, utils, electabilities=None, polls=None, pollingUncertainty=.15,
    winProbs=None, info='e', **kw):
        """Uses compromising without a specific target

        >>> Plurality.vaBallot((0,1,2), [.3, .3, .2])
        [0, 1, 0]
        >>> Plurality.vaBallot((0,1,10), [.3, .3, .2])
        [0, 0, 1]
        """
        if info == 'p':
            electabilities = polls
        if not winProbs:
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
        expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
        scores = [(u - expectedUtility)*p for u, p in zip(utils, winProbs)]
        return cls.oneVote(scores, scores.index(max(scores)))

    @classmethod
    def compBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
        """
        >>> Plurality.compBallot((0,1,10), 3, candToHelp=1, candToHurt=0)
        [0, 1, 0]
        >>> Plurality.compBallot((0,1,10), 3, candToHelp=2, candToHurt=0)
        [0, 0, 1]
        """
        if intensity < 3 or utils[candToHelp] <= utils[candToHurt]:
            return super().compBallot(utils, intensity, candToHelp, candToHurt, **kw)
        else:
            return cls.oneVote(utils, candToHelp)

    #
    # @classmethod
    # def xxstratBallot(cls, voter, polls, places, n,
    #                     frontId, frontResult, targId, targResult):
    #     """Takes utilities and returns a strategic ballot
    #     for the given "polling" info.
    #
    #     >>> Plurality().stratBallotFor([4,2,1])(Plurality, Voter([-4,-2,-1]))
    #     [0, 1, 0]
    #     """
    #     stratGap = voter[targId] - voter[frontId]
    #     if stratGap <= 0:
    #         #winner is preferred; be complacent.
    #         isStrat = False
    #         strat = cls.oneVote(voter, frontId)
    #     else:
    #         #runner-up is preferred; be strategic in iss run
    #         isStrat = True
    #         #sort cuts high to low
    #         #cuts = (cuts[1], cuts[0])
    #         strat = cls.oneVote(voter, targId)
    #     return dict(strat=strat, isStrat=isStrat, stratGap=stratGap)


def top2(noRunoffMethod):
    """Returns a top-2 variant of the given voting method
    """
    class Top2Version(noRunoffMethod):
        """Ballots are (r1Ballot, r2Preferences) tuples
        """

        @classmethod
        def results(cls, ballots):
            r1Ballots, r2Preferences = zip(*ballots)
            baseResults = super().results(r1Ballots)
            (runnerUp,top) = sorted(range(len(baseResults)), key=lambda i: baseResults[i])[-2:]
            upset = sum(sign(rank[runnerUp] - rank[top]) for rank in r2Preferences)
            if upset > 0:
                baseResults[runnerUp] = baseResults[top] + 0.01
            return baseResults

        @classmethod
        def prefOrder(cls, utils):
            order = [0]*len(utils)
            Borda.fillPrefOrder(utils, order)
            return order

        @classmethod
        def honBallot(cls, utils, **kw):
            return super().honBallot(utils, **kw), cls.prefOrder(utils)

        @classmethod
        def vaBallot(cls, utils, electabilities=None, polls=None, winProbs=None,
        pollingUncertainty=.15, info='e', **kw):
            if info == 'p':
                electabilities = polls
            if not winProbs:
                winProbs = adaptiveTieFor2(electabilities, pollingUncertainty)
            return (super().vaBallot(utils, winProbs=winProbs,
            pollingUncertainty=pollingUncertainty),
            cls.prefOrder(utils))

        @classmethod
        def compBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
            return (super().compBallot(utils, intensity, candToHelp, candToHurt),
            cls.prefOrder(utils))

        @classmethod
        def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
            return (super().diehardBallot(utils, intensity, candToHelp, candToHurt),
            cls.prefOrder(utils))

        @classmethod
        def stratBallot(cls, voter, *args, **kws):
            return (super().stratBallot(voter, *args, **kws),
            cls.prefOrder(voter))

        @classmethod
        def abstain(cls, utils, **kw):
            return [0]*len(utils), [0]*len(utils)
    Top2Version.__name__ = noRunoffMethod.__name__ + "Top2"
    return Top2Version

class PluralityTop2(top2(Plurality)):
    """top2(Plurality).vaBallot can yield ridiculous results when used by the entire electorate
    since it's based on causal decision theory. This class fixes that.

    >>> PluralityTop2.results([([0, 0, 1], [0, 1, 2])]*3+[([1, 0, 0], [2, 1, 0])]*2)
    [0.4, 0.0, 0.6]
    >>> PluralityTop2.results([([0, 0, 1], [0, 1, 2])]*5+[([1, 0, 0], [2, 1, 0])]*4+[([0, 1, 0], [1, 2, 0])]*2)
    [0.46454545454545454, 0.18181818181818182, 0.45454545454545453]
    >>> PluralityTop2.honBallot((0,1,5,2,3))
    ([0, 0, 1, 0, 0], [0, 1, 4, 2, 3])
    >>> PluralityTop2.compBallot((0,1,10), 3, candToHelp=1, candToHurt=0)
    ([0, 1, 0], [0, 1, 2])
    >>> PluralityTop2.stratBallot([0,1,2,3],[.51,.5,.49,.4])
    ([0, 1, 0, 0], [0, 1, 2, 3])
    """
    @classmethod
    def vaBallot(cls, utils, electabilities=None, **kw):
        """
        >>> PluralityTop2.vaBallot((0,1,2), [.3, .3, .2])
        ([0, 0, 1], [0, 1, 2])
        >>> PluralityTop2.vaBallot((0,1,2,3), [.3, .3, .2, .1])
        ([0, 0, 1, 0], [0, 1, 2, 3])
        """
        if electabilities and utils.index(max(utils)) == electabilities.index(max(electabilities)):
            return cls.honBallot(utils)
        else: return super().vaBallot(utils, electabilities=electabilities, **kw)

def makeScoreMethod(topRank=10, asClass=False):
    class Score0to(Method):
        """Score voting, 0-10.


        Strategy establishes pivots
            >>> Score().stratBallotFor([0,1,2])(Score, Voter([5,6,7]))
            [0, 0, 10]
            >>> Score().stratBallotFor([2,1,0])(Score, Voter([5,6,7]))
            [0, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([5,6,7]))
            [0, 5.0, 10]

        Strategy (kinda) works for ties
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([5,6,6]))
            [0, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,6,7]))
            [0, 0, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,7,6]))
            [10, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,5,6]))
            [10, 0, 10]

        """

        #>>> qs += [Score().resultsFor(PolyaModel()(101,2),Score.honBallot)[0] for i in range(800)]
        #>>> std(qs)
        #2.770135393419682
        #>>> mean(qs)
        #5.1467202970297032
        bias2 = 2.770135393419682
        #>>> qs5 = [Score().resultsFor(PolyaModel()(101,5),Score.honBallot)[0] for i in range(400)]
        #>>> mean(qs5)
        #4.920247524752476
        #>>> std(qs5)
        #2.3536762480634343
        bias5 = 2.3536762480634343
        compLevels = [1,2]
        diehardLevels = [1,2, 4]

        @classmethod
        def candScore(cls,scores):
            """Takes the list of votes for a candidate; returns the candidate's score.

            Don't just use mean because we want to normalize to [0,1]"""
            return mean(scores)/cls.topRank

        @classmethod
        def interpolatedBallot(cls, utils, lowThreshold, highThreshold):
            """
            >>> Score.interpolatedBallot([0,1,2,3,4,5], 1.5, 3.5)
            [0, 0, 1.0, 4.0, 5, 5]
            """
            ballot = []
            for util in utils:
                if util < lowThreshold:
                    ballot.append(0)
                elif util > highThreshold:
                    ballot.append(cls.topRank)
                else:
                    ballot.append(floor((cls.topRank + .99)*(util-lowThreshold)/(highThreshold-lowThreshold)))
            return ballot

        def __str__(self):
            if self.topRank == 1:
                return "IdealApproval"
            return self.__class__.__name__ + str(self.topRank)

        @classmethod
        def honBallot(cls, utils, **kw):
            """Takes utilities and returns an honest ballot (on 0..10)


            honest ballots work as expected
                >>> Score().honBallot(Score, Voter([5,6,7]))
                [0.0, 5.0, 10.0]
                >>> Score().resultsFor(DeterministicModel(3)(5,3),Score().honBallot)["results"]
                [4.0, 6.0, 5.0]
            """
            #raise Exception("NOT")
            bot = min(utils)
            scale = max(utils)-bot
            return [floor((cls.topRank + .99) * (util-bot) / scale) for util in utils]

        @classmethod
        def vaBallot(cls, utils, electabilities=None, polls=None, winProbs=None,
        pollingUncertainty=.15, info='e', **kw):
            if info == 'p':
                electabilities = polls
            if not winProbs:
                winProbs = pollsToProbs(electabilities, pollingUncertainty)
            expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
            return [cls.topRank if u > expectedUtility else 0 for u in utils]

        @classmethod
        def vaIntermediateBallot(cls, utils, electabilities=None, polls=None,
        winProbs=None, pollingUncertainty=.15, midScoreWillingness=0.7, info='e', **kw):
            """Uses significant, but not total, strategic exaggeration
            >>> Score.vaIntermediateBallot([0,1,2,3,4,5], [.6,.4,.4,.4,.4,.5])
            [0.0, 3.0, 5, 5, 5, 5]
            """
            if info == 'p':
                electabilities = polls
            if not winProbs:
                winProbs = pollsToProbs(electabilities, pollingUncertainty)
            expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
            if all(u == utils[0] for u in utils[1:]):
                return [0]*len(utils)
            lowThreshold = max(min(utils), expectedUtility - midScoreWillingness*std(utils))
            highThreshold = min(max(utils), expectedUtility + midScoreWillingness*std(utils))
            #this is wrong. We should use a standard deviation that's weighted
            #by each candidate's chance of winning
            return cls.interpolatedBallot(utils, lowThreshold, highThreshold)

        @classmethod
        def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
            if intensity < 1 or utils[candToHelp] <= utils[candToHurt]:
                return super().diehardBallot(utils, intensity, candToHelp, candToHurt, **kw)
            if intensity == 1:
                return cls.interpolatedBallot(utils, utils[candToHurt], utils[candToHelp])
            if intensity == 4:
                return cls.bulletBallot(utils)
            else:
                return [cls.topRank if u >= utils[candToHelp] else 0 for u in utils]

        @classmethod
        def compBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
            if intensity < 1 or utils[candToHelp] <= utils[candToHurt]:
                return super().diehardBallot(utils, intensity, candToHelp, candToHurt, **kw)
            if intensity == 1:
                return cls.interpolatedBallot(utils, utils[candToHurt], utils[candToHelp])
            else:
                return [cls.topRank if u > utils[candToHurt] else 0 for u in utils]

        @classmethod
        def bulletBallot(cls, utils, **kw):
            best = utils.index(max(utils))
            ballot = [0]*len(utils)
            ballot[best] = cls.topRank
            return ballot

        @classmethod
        def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                            frontId, frontResult, targId, targResult):
            """Returns a (function which takes utilities and returns a strategic ballot)
            for the given "polling" info."""

            cuts = [voter[frontId], voter[targId]]
            if stratGap > 0:
                #sort cuts high to low
                cuts = (cuts[1], cuts[0])
            if cuts[0] == cuts[1]:
                strat = [(cls.topRank if (util >= cuts[0]) else 0) for util in voter]
            else:
                strat = [max(0,min(cls.topRank,floor(
                                (cls.topRank + .99) * (util-cuts[1]) / (cuts[0]-cuts[1])
                            )))
                        for util in voter]
            for i in range(n):
                ballot[i] = strat[i]

    Score0to.topRank = topRank
    if asClass:
        return Score0to
    return Score0to()

class Score(makeScoreMethod(5, True)):
    """
    >>> Score.honBallot((0,1,2,3,4,5,6,7,8,9,10,11))
    [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    >>> Score.vaBallot((0,1,2,3),[.4,.4,.4,.4])
    [0, 0, 5, 5]
    >>> Score.vaBallot((0,1,2,3),[.4,.4,.4,.6])
    [0, 0, 0, 5]
    >>> Score.vaBallot((0,1,2,3),[.6,.4,.4,.4])
    [0, 5, 5, 5]
    """
    pass

class Approval(makeScoreMethod(1,True)):
    diehardLevels = [1, 4]
    compLevels = [1]
    @classmethod
    def zeroInfoBallot(cls, utils, electabilities=None, polls=None, pickiness=0, **kw):
        """Returns a ballot based on utils and pickiness
        pickiness=0 corresponds to vaBallot with equal polling for all candidates
        pickiness=1 corresponds to bullet voting

        >>> Approval.zeroInfoBallot([1,2,3,10], pickiness=0)
        [0, 0, 0, 1]
        >>> Approval.zeroInfoBallot([1,2,3,-10], pickiness=0)
        [1, 1, 1, 0]
        >>> Approval.zeroInfoBallot([1,2,3,-10], pickiness=0.6)
        [0, 1, 1, 0]
        """
        expectedUtility = sum(u for u in utils)/len(utils)
        best = max(utils)
        normalizedUtils = [(u - expectedUtility)/(best - expectedUtility)
                           for u in utils]
        return [1 if u >= pickiness else 0 for u in normalizedUtils]

    @classmethod
    def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
        """
        >>> Approval.diehardBallot([-10,1,2,3,4,10],0,4,2)
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> Approval.diehardBallot([-10,1,2,3,4,10],1,4,2)
        [0, 0, 0, 0, 1, 1]
        """
        if intensity == 1:
            return super().diehardBallot(utils, 2, candToHelp, candToHurt, **kw)
        else:
            return super().diehardBallot(utils, intensity, candToHelp, candToHurt, **kw)

    @classmethod
    def compBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
        """
        >>> Approval.compBallot([-10,1,2,3,4,20],1,4,2)
        [0, 0, 0, 1, 1, 1]
        >>> Approval.compBallot([-10,1,2,3,4,20],0,4,2)
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        """
        if intensity == 1:
            return super().compBallot(utils, 2, candToHelp, candToHurt, **kw)
        else:
            return super().compBallot(utils, intensity, candToHelp, candToHurt, **kw)

class ApprovalTop2(top2(Approval)):
    """
    >>> ApprovalTop2.vaBallot([0,1,10],[.5,.5,.2])
    ([0, 0, 1], [0, 1, 2])
    >>> ApprovalTop2.vaBallot([0,1,2,10],[.5,.5,.3,.2])
    ([0, 0, 1, 1], [0, 1, 2, 3])
    """
    @classmethod
    def bulletBallot(cls, utils, **kw):
        return super().bulletBallot(utils), cls.prefOrder(utils)

    @classmethod
    def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, **kw):
        """
        >>> ApprovalTop2.diehardBallot([0, 8, 9, 10], 4, 2, 1)
        ([0, 0, 0, 1], [0, 1, 2, 3])
        >>> ApprovalTop2.diehardBallot([0, 8, 9, 10], 1, 2, 1)
        ([0, 0, 1, 1], [0, 1, 2, 3])
        """
        if intensity == 4:
            return cls.bulletBallot(utils)
        else:
            return super().diehardBallot(utils, intensity, candToHelp, candToHurt, **kw)

def BulletyApprovalWith(bullets=0.5, asClass=False):
    class BulletyApproval(Score(1,True)):

        bulletiness = bullets

        def __str__(self):
            return "BulletyApproval" + str(round(self.bulletiness * 100))


        @classmethod
        def honBallot(cls, utils, **kw):
            """Takes utilities and returns an honest ballot (on 0..10)


            honest ballots work as expected
                >>> Score().honBallot(Score, Voter([5,6,7]))
                [0.0, 5.0, 10.0]
                >>> Score().resultsFor(DeterministicModel(3)(5,3),Score().honBallot)["results"]
                [4.0, 6.0, 5.0]
            """
            if random.random() > cls.bulletiness:
                return cls.__bases__[0].honBallot(cls, utils)
            best = max(utils)
            return [1 if util==best else 0 for util in utils]

    if asClass:
        return BulletyApproval
    return BulletyApproval()


def makeSTARMethod(topRank=5):
    "STAR Voting"

    score0to = makeScoreMethod(topRank,True)

    class STAR0to(score0to):

        stratTargetFor = Method.stratTarget3
        diehardLevels = [1,2,3,4]
        compLevels = [1,2,3]

        @classmethod
        def results(cls, ballots, **kwargs):
            """STAR results.

            >>> STAR().resultsFor(DeterministicModel(3)(5,3),Irv().honBallot)["results"]
            [0, 1, 2]
            >>> STAR().results([[0,1,2]])[2]
            2
            >>> STAR().results([[0,1,2],[2,1,0]])[1]
            0
            >>> STAR().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
            [2, 0, 1]
            """
            baseResults = super(STAR0to, cls).results(ballots, **kwargs)
            (runnerUp,top) = sorted(range(len(baseResults)), key=lambda i: baseResults[i])[-2:]
            upset = sum(sign(ballot[runnerUp] - ballot[top]) for ballot in ballots)
            if upset > 0:
                baseResults[runnerUp] = baseResults[top] + 0.01
            return baseResults

        @classmethod
        def vaBallot(cls, utils, electabilities=None, polls=None, winProbs=None,
        pollingUncertainty=.15, scoreImportance=0.17, info='e', **kw):
            """
            >>> STAR.vaBallot([0,1,2,3,4,5],[.5,.5,.5,.5,.5,.5],scoreImportance=0.1)
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            >>> STAR.vaBallot([0,1,2,3,4,5],[.5,.5,.5,.5,.5,.5],scoreImportance=0.2)
            [0.0, 0.0, 1.0, 4.0, 5.0, 5.0]
            >>> STAR.vaBallot([0,1,2,3,4,5],[.5,.5,.5,.5,.5,.5],scoreImportance=3)
            [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
            >>> STAR.vaBallot([0,1,2,3,4,5],[.6,.5,.5,.5,.5,.5],scoreImportance=0.2)
            [0.0, 1.0, 2.0, 5.0, 5.0, 5.0]
            """
            if info == 'p':
                electabilities = polls
            if not winProbs:
                winProbs = pollsToProbs(electabilities, pollingUncertainty)
            #runoffCoefficients[i][j] is how valuable it is to score i over j
            runoffCoefficients = [[(u1 - u2)*p1*p2
                                   for u2, p2 in zip(utils, winProbs)]
                                  for u1, p1 in zip(utils, winProbs)]
            eRunnerUpUtil = sum(u*p for u, p in zip(utils, adaptiveTieFor2(electabilities)))
            #scoreCoefficients[i] is how vauable it is for i to have a high score
            scoreCoefficients = [scoreImportance*(u-eRunnerUpUtil)*p
                                 for u, p in zip(utils, adaptiveTieFor2(electabilities))]

            #create a tentative ballot
            numCands = len(utils)
            bot = min(utils)
            scale = max(utils)-bot
            ballot = [floor((cls.topRank + .99) * (util-bot) / scale) for util in utils]

            #optimize the ballot
            improvementFound = True
            while improvementFound:
                improvementFound = False
                for cand in range(numCands):
                    #Should cand be scored higher?
                    if (ballot[cand] < cls.topRank and
                        sum(runoffCoefficients[cand][j]*sign(ballot[cand] + 1 - ballot[j])
                            for j in range(numCands))
                        + scoreCoefficients[cand]
                        > sum(runoffCoefficients[cand][j]*sign(ballot[cand] - ballot[j])
                              for j in range(numCands))):
                        ballot[cand] += 1
                        improvementFound = True
                    #Should cand be scored lower?
                    elif (ballot[cand] > 0 and
                        sum(runoffCoefficients[cand][j]*sign(ballot[cand] - 1 - ballot[j])
                            for j in range(numCands))
                        - scoreCoefficients[cand]
                        > sum(runoffCoefficients[cand][j]*sign(ballot[cand] - ballot[j])
                              for j in range(numCands))):
                        ballot[cand] -= 1
                        improvementFound = True
            return ballot

        @classmethod
        def compBallot(cls, utils, intensity, candToHelp, candToHurt, baseBallotFunc=None, **kw):
            """Intensity determines the strategy used. 0 naive, 1 is honest, 2 semi-honest, 3 is favorite betrayal
            >>> STAR.compBallot([0,1,2,3,4,5,6,7],0,5,3)
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
            >>> STAR.compBallot([0,1,2,3,4,5,6,7],1,5,3)
            [0, 0, 0, 0, 3.0, 4, 5.0, 5.0]
            >>> STAR.compBallot([0,1,2,3,4,5,6,7],2,5,3)
            [0, 0, 0, 0, 3.0, 5, 5, 5]
            >>> STAR.compBallot([0,1,2,3,4,5,6,7],3,5,3)
            [0, 0, 0, 0, 1, 5, 1, 1]
            """
            if baseBallotFunc is None: baseBallotFunc = cls.honBallot
            baseBallot = baseBallotFunc(utils, candToHelp=candToHelp, candToHurt=candToHurt, **kw)
            helpUtil, hurtUtil = utils[candToHelp], utils[candToHurt]
            if intensity < 1 or helpUtil <= hurtUtil:
                return baseBallot
            if intensity == 1:
                for i, u in enumerate(utils):
                    if u >= helpUtil:
                        baseBallot[i] = max(cls.topRank - 1, baseBallot[i])
                    elif u <= hurtUtil:
                        baseBallot[i] = 0
            if intensity == 2:
                for i, u in enumerate(utils):
                    if u >= helpUtil:
                        baseBallot[i] = cls.topRank
                    elif u <= hurtUtil:
                        baseBallot[i] = 0
            if intensity >= 3:
                baseBallot = [1 if u > hurtUtil else 0 for u in utils]
                baseBallot[candToHelp] = cls.topRank
            return baseBallot

        @classmethod
        def diehardBallot(cls, utils, intensity, candToHelp, candToHurt, electabilities=None, polls=None,
        baseBallotFunc=None, info='p', **kw):
            """Intensity determines the strategy used. 0 naive, 1 is honest, 2 semi-honest, 3 is pushover,
            and 4 is bullet voting
            >>> STAR.diehardBallot([0,1,2,3,4,5,6,7],1,5,3)
            [0.0, 0.0, 1, 1, 3.0, 5, 5, 5]
            >>> STAR.diehardBallot([0,1,2,3,4,5,6,7],2,5,3)
            [0, 0, 0, 0, 3.0, 5, 5, 5]
            >>> STAR.diehardBallot([0,1,2,3,4,5,6,7],3,5,3,polls=[.4,.6,.4,.6,.4,.5,.4,.6])
            [4, 0, 4, 0, 4, 5, 5, 5]
            >>> STAR.diehardBallot([0,1,2,3,4,5,6,7],4,5,3)
            [0, 0, 0, 0, 0, 0, 0, 5]
            """
            if info == 'e':
                polls = electabilities
            if baseBallotFunc is None: baseBallotFunc = cls.honBallot
            baseBallot = baseBallotFunc(utils, polls=polls, candToHelp=candToHelp, candToHurt=candToHurt, **kw)
            helpUtil, hurtUtil = utils[candToHelp], utils[candToHurt]
            if intensity < 1 or helpUtil <= hurtUtil:
                return baseBallot
            if intensity == 1:
                for i, u in enumerate(utils):
                    if u <= hurtUtil:
                        baseBallot[i] = min(1, baseBallot[i])
                    elif u >= helpUtil:
                        baseBallot[i] = cls.topRank
            if intensity == 2:
                for i, u in enumerate(utils):
                    if u <= hurtUtil:
                        baseBallot[i] = 0
                    elif u >= helpUtil:
                        baseBallot[i] = cls.topRank
            if intensity == 3:
                for i, u in enumerate(utils):
                    if u >= helpUtil:
                        baseBallot[i] = cls.topRank
                    elif polls[i] < polls[candToHelp]:
                        baseBallot[i] = cls.topRank - 1
                    else:
                        baseBallot[i] = 0
                    baseBallot[candToHurt] = 0
            if intensity == 4:
                return cls.bulletBallot(utils)
            return baseBallot

        @classmethod
        def always510Ballot(cls, utils, **kw):
            """Gives the first choice candidate a 5, second choice a 1, and last choice a 0.
            Only really intended for three-candidate elections.
            >>> STAR.always510Ballot([9,5,10])
            [1, 0, 5]
            """
            return [cls.topRank if u == max(utils) else 0 if u == min(utils) else 1 for u in utils]

        @classmethod
        def never23Ballot(cls, utils, **kw):
            """Gives the first choice a 5, last a 0, and everyone else 1 or 4
            >>> STAR.never23Ballot([0,2,3,5])
            [0, 1, 4, 5]
            """
            high, low = max(utils), min(utils)
            ballot = []
            for u in utils:
                if u == high:
                    ballot.append(cls.topRank)
                elif u == low:
                    ballot.append(0)
                elif high - u < u - low:
                    ballot.append(cls.topRank - 1)
                else:
                    ballot.append(1)
            return ballot

        @classmethod
        def utilGapBallot(cls, utils, **kw):
            """With six or fewer candidates, gives each candidate a different score.
            With five or fewer candidates, there is only one gap between scores
            and it occurs where there is the greatest gap in utilities.
            With seven or more candidates, the canddiates with the most similar
            utilities are given the same scores.
            Does not work well when some utilities are equal.
            >>> STAR.utilGapBallot([0,6,5,7])
            [0, 4, 3, 5]
            >>> STAR.utilGapBallot([0,6,5,2])
            [0, 5, 4, 1]
            >>> STAR.utilGapBallot([0,1,2])
            [0, 1, 5]
            >>> STAR.utilGapBallot([0,1,2,3,4,10,11,13])
            [0, 0, 1, 1, 2, 3, 4, 5]
            """
            sortedUtils = sorted(enumerate(utils), key=lambda x: x[1])
            high, low = max(utils), min(utils)
            n = len(utils)
            ballot = [0]*n
            if n <= cls.topRank + 1:
                maxGap = 0
                gapLoc = None
                for i in range(n-1):
                    if sortedUtils[i + 1][1] - sortedUtils[i][1] >= maxGap:
                        maxGap = sortedUtils[i + 1][1] - sortedUtils[i][1]
                        gapLoc = (sortedUtils[i + 1][1] + sortedUtils[i][1])/2
                for rank, (cand, util) in enumerate(sortedUtils):
                    if util < gapLoc:
                        ballot[cand] = rank
                    else:
                        ballot[cand] = rank + cls.topRank + 1 - n
            else:
                utilRanges = [[u,u] for u in sorted(utils)]
                #use a greedy algorithm to put most similar utilities together
                while len(utilRanges) > cls.topRank + 1:
                    minSpread = float('inf')
                    minLoc = None
                    for i in range(len(utilRanges) - 1):
                        spread = utilRanges[i+1][1] - utilRanges[i][0]
                        if spread < minSpread:
                            minSpread = spread
                            minLoc = i
                    utilRanges[minLoc][1] = utilRanges[minLoc+1][1]
                    del utilRanges[minLoc+1]
                score = 0
                for cand, util in sortedUtils:
                    if util > utilRanges[score][1]:
                        score += 1
                    ballot[cand] = score
            return ballot

        @classmethod
        def defaultbgs(cls):
            return super().defaultbgs() + [cls.utilGapBallot]

        @classmethod
        def defaultfgs(cls):
            return super().defaultfgs()\
            + [(cls.utilGapBallot, targs) for targs in [selectRand, select21]]


    if topRank==5:
        STAR0to.__name__ = "STAR"
    else:
        STAR0to.__name__ = "STAR" + str(topRank)
    return STAR0to

class STAR(makeSTARMethod(5)): pass

def toVote(cutoffs, util):
    """maps one util to a vote, using cutoffs.

    Used by Mav, but declared outside to avoid method binding overhead."""
    for vote in range(len(cutoffs)):
        if util <= cutoffs[vote]:
            return vote
    return vote + 1


class Mav(Method):
    """Majority Approval Voting
    """


    #>>> mqs = [Mav().resultsFor(PolyaModel()(101,5),Mav.honBallot)[0] for i in range(400)]
    #>>> mean(mqs)
    #1.5360519801980208
    #>>> mqs += [Mav().resultsFor(PolyaModel()(101,5),Mav.honBallot)[0] for i in range(1200)]
    #>>> mean(mqs)
    #1.5343069306930679
    #>>> std(mqs)
    #1.0970202515275356
    bias5 = 1.0970202515275356


    baseCuts = [-0.8, 0, 0.8, 1.6]
    specificCuts = None
    specificPercentiles = [25,50,75,90]

    @classmethod
    def candScore(self, scores):
        """For now, only works correctly for odd nvot

        Basic tests
            >>> Mav().candScore([1,2,3,4,5])
            3.0
            >>> Mav().candScore([1,2,3,3,3])
            2.5
            >>> Mav().candScore([1,2,3,4])
            2.5
            >>> Mav().candScore([1,2,3,3])
            2.5
            >>> Mav().candScore([1,2,2,2])
            1.5
            >>> Mav().candScore([1,2,3,3,5])
            2.7
            """
        scores = sorted(scores)
        nvot = len(scores)
        nGrades = (len(self.baseCuts) + 1)
        i = int((nvot - 1) / 2)
        base = scores[i]
        while (i < nvot and scores[i] == base):
            i += 1
        upper =  (base + 0.5) - (i - nvot/2) * nGrades / nvot
        lower = (base) - (i - nvot/2) / nvot
        return max(upper, lower)

    @classmethod
    def honBallotFor(cls, voters):
        cls.specificCuts = percentile(voters,cls.specificPercentiles)
        return cls.honBallot

    @classmethod
    def honBallot(cls, voter, **kw):
        """Takes utilities and returns an honest ballot (on 0..4)

        honest ballot works as intended, gives highest grade to highest utility:
            >>> Mav.honBallot(Voter([-1,-0.5,0.5,1,1.1]))
            [0, 1, 2, 3, 4]

        Even if they don't rate at least an honest "B":
            >>> Mav.honBallot(Voter([-1,-0.5,0.5]))
            [0, 1, 4]
        """
        cuts = cls.specificCuts if (cls.specificCuts is not None) else cls.baseCuts
        cuts = [min(cut, max(voter) - 0.001) for cut in cuts]
        return [toVote(cuts, util) for util in voter]

    @classmethod
    def vaBallot(cls, utils, electabilities=None, polls=None, winProbs=None,
            pollingUncertainty=.15, info='e', **kw):
        """Acts like the VA approval strat in deciding whether to give a candidate
        ANY support, but those that are supported are scored in accordance with their utilities

        >>> Mav.vaBallot([0,1,2,3],[.4,.4,.4,.4])
        [0, 0, 1.0, 4.0]
        >>> Mav.vaBallot([0,1,2,3],[.6,.4,.4,.4])
        [0, 1.0, 3.0, 4.0]
        >>> Mav.vaBallot([0,1,2,3],[.4,.4,.4,.5])
        [0, 0, 0, 4.0]
        """
        if info == 'p':
            electabilities = polls
        if not winProbs:
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
        expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
        scale = max(utils)-expectedUtility
        return [0 if expectedUtility > util else
                floor((4 + .99) * (util-expectedUtility) / scale) for util in utils]

    @classmethod
    def stratBallot(cls, voter, polls, electabilities=None, info='p', **kw):
        if info == 'e':
            polls = electabilities
        places = sorted(enumerate(polls),key=lambda x:-x[1])
        ((frontId,frontResult), (targId, targResult)) = places[0:2]
        frontUtils = [voter[frontId], voter[targId]] #utils of frontrunners
        stratGap = frontUtils[1] - frontUtils[0]
        if stratGap == 0:
            return [(4 if (util >= frontUtils[0]) else 0)
                                 for util in voter]

        if stratGap < 0:
            return cls.honBallot(voter)
        else:
            #runner-up is preferred; be strategic in iss run
            #sort cuts high to low
            frontUtils = (frontUtils[1], frontUtils[0])
            top = max(voter)
            #print("lll312")
            #print(self.baseCuts, front)
            cutoffs = [(  (min(frontUtils[0], cls.baseCuts[i]))
                             if (i < floor(targResult)) else
                        ( (frontUtils[1])
                             if (i < floor(frontResult) + 1) else
                          min(top, cls.baseCuts[i])
                          ))
                       for i in range(len(cls.baseCuts))]
            return [toVote(cutoffs, util) for util in voter]

    def stratBallotFor(self, polls):
        """Returns a function which takes utilities and returns a dict(
            strat=<ballot in which all grades are exaggerated
                             to outside the range of the two honest frontrunners>,
            extraStrat=<ballot in which all grades are exaggerated to extremes>,
            isStrat=<whether the runner-up is preferred to the frontrunner (for reluctantStrat)>,
            stratGap=<utility of runner-up minus that of frontrunner>
            )
        for the given "polling" info.



        Strategic tests:
            >>> Mav().stratBallotFor([0,1.1,1.9,0,0])(Mav, Voter([-1,-0.5,0.5,1,2]))
            [0, 1, 2, 3, 4]
            >>> Mav().stratBallotFor([0,2.1,2.9,0,0])(Mav, Voter([-1,-0.5,0.5,1,2]))
            [0, 1, 3, 3, 4]
            >>> Mav().stratBallotFor([0,2.1,1.9,0,0])(Mav, Voter([-1,0.4,0.5,1,2]))
            [0, 1, 3, 3, 4]
            >>> Mav().stratBallotFor([1,0,2])(Mav, Voter([6,7,6]))
            [4, 4, 4]
            >>> Mav().stratBallotFor([1,0,2])(Mav, Voter([6,5,6]))
            [4, 0, 4]
            >>> Mav().stratBallotFor([2.1,0,3])(Mav, Voter([6,5,6]))
            [4, 0, 4]
            >>> Mav().stratBallotFor([2.1,0,3])(Mav, Voter([6,5,6.1]))
            [2, 2, 4]
        """
        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        #print("places",places)
        ((frontId,frontResult), (targId, targResult)) = places[0:2]

        @rememberBallots
        def stratBallot(cls, voter):
            frontUtils = [voter[frontId], voter[targId]] #utils of frontrunners
            stratGap = frontUtils[1] - frontUtils[0]
            if stratGap == 0:
                strat = extraStrat = [(4 if (util >= frontUtils[0]) else 0)
                                     for util in voter]
                isStrat = True

            else:
                if stratGap < 0:
                    #winner is preferred; be complacent.
                    isStrat = False
                else:
                    #runner-up is preferred; be strategic in iss run
                    isStrat = True
                    #sort cuts high to low
                    frontUtils = (frontUtils[1], frontUtils[0])
                top = max(voter)
                #print("lll312")
                #print(self.baseCuts, front)
                cutoffs = [(  (min(frontUtils[0], self.baseCuts[i]))
                                 if (i < floor(targResult)) else
                            ( (frontUtils[1])
                                 if (i < floor(frontResult) + 1) else
                              min(top, self.baseCuts[i])
                              ))
                           for i in range(len(self.baseCuts))]
                strat = [toVote(cutoffs, util) for util in voter]
                extraStrat = [max(0,min(10,floor(
                                4.99 * (util-frontUtils[1]) / (frontUtils[0]-frontUtils[1])
                            )))
                        for util in voter]
            return dict(strat=strat, extraStrat=extraStrat, isStrat=isStrat,
                        stratGap = stratGap)
        return stratBallot


class Mj(Mav):
    def candScore(self, scores):
        """This formula will always give numbers within 0.5 of the raw median.
        Unfortunately, with 5 grade levels, these will tend to be within 0.1 of
        the raw median, leaving scores further from the integers mostly unused.
        This is only a problem aesthetically.

        For now, only works correctly for odd nvot

        tests:
            >>> Mj().candScore([1,2,3,4,5])
            3
            >>> Mj().candScore([1,2,3,3,5])
            2.7
            >>> Mj().candScore([1,3,3,3,5])
            3
            >>> Mj().candScore([1,3,3,4,5])
            3.3
            >>> Mj().candScore([1,3,3,3,3])
            2.9
            >>> Mj().candScore([3] * 24 + [1])
            2.98
            >>> Mj().candScore([3] * 24 + [4])
            3.02
            >>> Mj().candScore([3] * 13 + [4] * 12)
            3.46
            """
        scores = sorted(scores)
        nvot = len(scores)
        lo = hi = mid = nvot // 2
        base = scores[mid]
        while (hi < nvot and scores[hi] == base):
            hi += 1
        while (lo >= 0 and scores[lo] == base):
            lo -= 1

        if (hi-mid) == (mid-lo):
            return base
        elif (hi-mid) < (mid-lo):
            return base + 0.5 - (hi-mid) / nvot
        else:
            return base - 0.5 + (mid-lo) / nvot

class Irv(Method):
    """
    IRV.

    High numbers are good for both results and votes (pretty sure).
    """

    stratTargetFor = Method.stratTarget3
    compLevels = [3]

    @classmethod
    def oldResort(self, ballots, loser, ncand, piles):
        """No error checking; only works for exhaustive ratings."""
        #print("resort",ballots, loser, ncand)
        #print(piles)
        for ballot in ballots:
            if loser < 0:
                nextrank = ncand - 1
            else:
                nextrank = ballot[loser] - 1
            while 1:
                try:
                    piles[ballot.index(nextrank)].append(ballot)
                    break
                except AttributeError:
                    nextrank -= 1
                    if nextrank < 0:
                        raise

    @classmethod
    def resort(cls, ballotsToSort, candsLeft, piles):
        for b in ballotsToSort:
            vote, bestRank = None, 0
            for c in candsLeft:
                if b[c] > bestRank:
                    vote, bestRank = c, b[c]
            if vote is not None:
                piles[vote].append(b)

    @classmethod
    def results(cls, ballots):
        """
        >>> Irv.resultsFor(DeterministicModel(3)(5,3))
        [0.2, 0.4, 0.6]
        >>> Irv.results([[0,1,2]])
        [0.0, 0.0, 1.0]
        >>> Irv.results([[0,1,2]]*4+[[0,2,1]]*4+[[0,0,0]]*2)
        [0.0, 0.4, 0.4]
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        ncand = len(ballots[0])
        nbal = len(ballots)
        piles = [[] for i in range(ncand)]
        candsLeft = set(range(ncand))
        ballotsToSort = ballots
        eliminations = [] #(candidateIndex, defeatMargin) tuples
        while len(candsLeft) > 1:
            cls.resort(ballotsToSort, candsLeft, piles)
            loser, loserVotes, defeatMargin = 0, float('inf'), float('inf')
            for cand in candsLeft: #determine who gets eliminated
                if len(piles[cand]) < loserVotes:
                    loser = cand
                    defeatMargin = loserVotes - len(piles[cand])
                    loserVotes = len(piles[cand])
                elif len(piles[cand]) - loserVotes < defeatMargin:
                    defeatMargin = len(piles[cand]) - loserVotes
            candsLeft.remove(loser)
            ballotsToSort = piles[loser]
            eliminations.append((loser, defeatMargin))

        winner = candsLeft.pop()
        voteCount = len(piles[winner])
        results = [0]*ncand
        results[winner] = voteCount/nbal
        for loser, margin in reversed(eliminations):
            voteCount = max(0, voteCount - margin)
            results[loser] = voteCount/nbal
        return results


    @classmethod
    def oldResults(self, ballots, **kwargs):
        """IRV results.

        >>> #Irv.resultsFor(DeterministicModel(3)(5,3))
        [0, 1, 2]
        >>> #Irv.results([[0,1,2]])[2]
        2
        >>> #Irv.results([[0,1,2],[2,1,0]])[1]
        0
        >>> #Irv.results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [2, 0, 1]
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        ncand = len(ballots[0])
        results = [-1] * ncand
        piles = [[] for i in range(ncand)]
        loserpile = ballots
        loser = -1
        for i in range(ncand):
            self.resort(loserpile, loser, ncand, piles)
            negscores = ["x" if isnum(pile) else -len(pile)
                         for pile in piles]
            loser = self.winner(negscores)
            results[loser] = i
            loserpile, piles[loser] = piles[loser], -1
        return results


    @classmethod
    def honBallot(cls, voter, **kw):
        """Takes utilities and returns an honest ballot

        >>> Irv.honBallot(Voter([4,1,6,3]))
        [2, 0, 3, 1]
        """
        ballot = [-1] * len(voter)
        order = sorted(enumerate(voter), key=lambda x:x[1])
        for i, cand in enumerate(order):
            ballot[cand[0]] = i
        #print("hballot",ballot)
        return ballot

    @classmethod
    def vaBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.15,
    winProbs=None, info='e', **kw):
        """Ranks good electable candidates over great unelectable candidates
        Electabilities are interpreted as a metric for the ability to win in the final round.

        >>> Irv.vaBallot([0,1,2,3],[.4,.5,.5,.4])
        [0, 1, 3, 2]
        >>> Irv.vaBallot([0,1,2,10],[.4,.5,.5,.4])
        [0, 1, 2, 3]
        >>> Irv.vaBallot([0,1,2,10],[.6,.5,.4,.38])
        [0, 3, 1, 2]
        """
        #if info == 'p': commented out because this can't handle IRV polls
            #electabilities = polls
        if not winProbs:
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
        expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
        scores = [(u - expectedUtility)*p for u, p in zip(utils, winProbs)]
        goodCandidates = sorted(filter(lambda x: x[1] > 0, enumerate(scores)), key=lambda x:x[1]) #from worst to best score
        badCandidates = sorted([(i,utils[i]) for i in range(len(utils)) if scores[i] <= 0], key=lambda x:x[1]) #from worst to best utility
        order = badCandidates + goodCandidates
        ballot = [-1] * len(utils)
        for i in range(len(utils)):
            ballot[order[i][0]] = i
        return ballot

    @classmethod
    def compBallot(cls, utils, intensity, candToHelp, candToHurt=None, **kw):
        """Rank candToHelp first, then vote honestly
        >>> Irv.compBallot([0,1,2,10],3,1,0)
        [0, 3, 1, 2]
        """
        ballot = cls.honBallot(utils)
        if intensity < 3: return ballot
        helpRank = ballot[candToHelp]
        for cand, rank in enumerate(ballot):
            if rank > helpRank:
                ballot[cand] -= 1
        ballot[candToHelp] = len(utils) - 1
        return ballot

    @classmethod
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):
        """
        >>> Irv.stratBallot(Voter([3,6,5,2]),[3,2,1,0])
        [1, 2, 3, 0]
        """
        i = n - 1
        winnerQ = voter[frontId]
        targQ = voter[targId]
        placesToFill = list(range(n-1,0,-1))
        if targQ > winnerQ:
            ballot[targId] = i
            i -= 1
            del placesToFill[-2]
        for j in placesToFill:
            nextLoser, loserScore = places[j] #all but winner, low to high
            if voter[nextLoser] > winnerQ:
                ballot[nextLoser] = i
                i -= 1
        ballot[frontId] = i
        i -= 1
        for j in placesToFill:
            nextLoser, loserScore = places[j]
            if voter[nextLoser] <= winnerQ:
                ballot[nextLoser] = i
                i -= 1
        #assert list(range(n)) == sorted(ballot)
        assert i == -1

class V321(Mav):
    baseCuts = [-.1,.8]
    specificPercentiles = [45, 75]

    stratTargetFor = Method.stratTarget3

    @classmethod
    def results(self, ballots, isHonest=False, **kwargs):
        """3-2-1 Voting results.

        >>> V321.resultsFor(DeterministicModel(3)(5,3))
        [-0.75, 2, 1]
        >>> V321.results([[0,1,2]])[2]
        2
        >>> V321.results([[0,1,2],[2,1,0]])[1]
        2.5
        >>> V321.results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [1, 1.5, -0.25]
        >>> V321.results([[0,1,2,1]]*29 + [[1,2,0,1]]*30 + [[2,0,1,1]]*31 + [[1,1,1,2]]*10)
        [3, 0.5, 1, 0]
        >>> V321.results([[1,0,2,1]]*29 + [[0,2,1,1]]*30 + [[2,1,0,1]]*31 + [[1,1,1,2]]*10)
        [3.375, 2.875, 0.25, 0]
        """
        candScores = list(zip(*ballots))
        n2s = [sum(1 if s>1 else 0 for s in c) for c in candScores]
        o2s = argsort(n2s) #order
        r2s = [-1] * len(n2s) #ranks
        for r,i in enumerate(o2s):
            r2s[i] = r
        semifinalists = o2s[-3:] #[third, second, first] by top ranks
        #print(semifinalists)
        n1s = [sum(1 if s>0 else 0 for s in candScores[sf]) for sf in semifinalists]
        o1s = argsort(n1s)
        #print("n1s",n1s)
        #print("o1s",o1s)
        #print([semifinalists[o] for o in o1s]) #[third, second, first] by above-bottom
        #print("r2s",r2s)
        r2s[semifinalists[o1s[0]]] -= (o1s[0] +1) * .75 #non-finalist below finalists
        semiupset = o1s[1] < o1s[2] #semifinalist and finalist order are different



        (runnerUp,top) = semifinalists[o1s[1]], semifinalists[o1s[2]]
        upset = sum(sign(ballot[runnerUp] - ballot[top]) for ballot in ballots)
        if upset > 0:
            runnerUp, top = top, runnerUp
            r2s[runnerUp], r2s[top] = r2s[top] - .125, r2s[runnerUp] + .125
        r2s[top] = max(r2s[top], r2s[runnerUp] + 0.5)
        if isHonest:
            upset2 =  sum(sign(ballot[semifinalists[o1s[0]]] - ballot[semifinalists[o1s[2]]]) for ballot in ballots)
            self.__class__.extraEvents["3beats1"] = upset2 > 0
            upset3 =  sum(sign(ballot[semifinalists[o1s[0]]] - ballot[semifinalists[o1s[1]]]) for ballot in ballots)
            self.__class__.extraEvents["3beats2"] = upset3 > 0
            if len(o2s) > 3:
                fourth = o2s[-4]
                fourthNotLasts = sum(1 if s>1 else 0 for s in candScores[fourth])
                fourthWin = (fourthNotLasts > n1s[o1s[1]] and
                             sum(sign(ballot[fourth] - ballot[semifinalists[o1s[2]]])
                                    for ballot in ballots)
                                > 0)
                self.__class__.extraEvents["4beats1"] = fourthWin

        return r2s

    def stratBallotFor(self, polls):
        """Returns a function which takes utilities and returns a dict(
            isStrat=
        for the given "polling" info.
        """
        ncand = len(polls)

        places = sorted(enumerate(polls),key=lambda x:-x[1]) #high to low
        top3 = [c for c,r in places[:3]]

        #@rememberBallots ... do it later
        def stratBallot(cls, voter):
            stratGap = voter[top3[1]] - voter[top3[0]]
            myPrefs = [c for c,v in sorted(enumerate(voter),key=lambda x:-x[1])] #high to low
            my3order = [myPrefs.index(c) for c in top3]
            rating = 2
            ballot = [0] * len(voter)
            if my3order[0] == min(my3order): #agree on winner
                for i in range(my3order[0]+1):
                    ballot[myPrefs[i]] = 2
                if my3order[1] <= my3order[2]:
                    for i in range(my3order[0]+1,my3order[1]+1):
                        ballot[myPrefs[i]] = 1
                #print("agree",top3, my3order,ballot,[float('%.1g' % c) for c in voter])
                return dict(strat=ballot, isStrat=False, stratGap=stratGap)
            for c in myPrefs:
                ballot[c] = rating
                if rating and (c in top3):
                    if c == top3[0]:
                        rating = 0
                    else:
                        rating -= 1

            #print("disagree",top3,my3order,ballot,[float('%.1g' % c) for c in voter])
            return dict(strat=ballot, isStrat=True, stratGap=stratGap)
        if self.extraEvents["3beats1"]:
            @rememberBallots
            def stratBallo2(cls, voter):
                stratGap = voter[top3[1]] - voter[top3[0]]
                myprefs = sorted(enumerate(voter),key=lambda x:-x[1]) #high to low
                rating = 2
                ballot = [None] * len(voter)
                isStrat=False
                stratGap = 0
                for c, util in myprefs:
                    ballot[c] = rating
                    if rating and (c in top3):
                        if (c == top3[2]):
                            isStrat= (rating == 2)
                            rating = 0
                        else:
                            rating -= 1
                isStrat = (voter[top3[0]] == max(voter[c] for c in top3))
                return dict(strat=ballot, isStrat=isStrat, stratGap=stratGap)
            stratBallo2.__name__ = "stratBallot" #God, that's ugly.
            return stratBallo2

        if self.extraEvents["4beats1"]:
            fourth = places[3][1]
            first = top3[1]
            @rememberBallots
            def stratBallo3(cls, voter):
                stratGap = voter[top3[1]] - voter[top3[0]]
                myprefs = sorted(enumerate(voter),key=lambda x:-x[1]) #high to low

                rating = 2
                ballot = [None] * len(voter)
                if voter[fourth] > voter[first]:

                    for c, util in myprefs:
                        ballot[c] = rating
                        if rating and (c == fourth):
                            rating -= 2
                        return dict(strat=ballot, isStrat=True, stratGap=stratGap)

                return stratBallot(cls,voter)
            stratBallo3.__name__ = "stratBallot" #God, that's ugly.
            return stratBallo3


        return rememberBallots(stratBallot)

class Condorcet(RankedMethod):

    diehardLevels = [3]
    compLevels = [3]

    @classmethod
    def resolveCycle(cls, cmat, n, ballots):
        raise NotImplementedError

    @classmethod
    def results(cls, ballots, **kwargs):
        """
        >>> Condorcet.results([[0,1,2]]*3+[[2,1,0]]*2)
        [0.4, 0.4, 0.6]
        >>> Condorcet.results([[0,1,2]]*3+[[2,1,0]]*4+[[1,2,0]]*2)
        [0.4444444444444444, 0.5555555555555556, 0.33333333333333337]
        """
        cmat = cls.compMatrix(ballots)
        smith = cls.smithSet(cmat)
        numCands = len(ballots[0])
        results = [0]*numCands
        if len(smith) == 1:
            winner = smith.pop()
            results[winner] = .5 + min(marg for marg in cmat[winner] if marg != 0)/(2*len(ballots))
            for i in range(numCands):
                if i == winner: continue
                results[i] = .5 + cmat[i][winner]/(2*len(ballots))
            return results
        else:
            ordinalResults = cls.resolveCycle([row.copy() for row in cmat], numCands, ballots)
            return [.4 + r/(10*numCands) for r in ordinalResults] #hideous hack


    @staticmethod
    def compMatrix(ballots):
        """
        >>> Condorcet.compMatrix([[0,2,1]]*5)
        [[0, -5, -5], [5, 0, 5], [5, -5, 0]]
        >>> Condorcet.compMatrix([[0,2,1]]*5+[[2,1,0]]*4+[[1,0,2]]*3)
        [[0, 2, -4], [-2, 0, 6], [4, -6, 0]]
        """
        n = len(ballots[0])
        cmat = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    cmat[i][j] = sum(sign(ballot[i] - ballot[j]) for ballot in ballots)
        return cmat

    @staticmethod
    def smithSet(cmat):
        """Returns the Smith set for a given comparison matrix.

        >>> Condorcet.smithSet([[0,-4,-3],[4,0,1],[3,-1,0]])
        {1}
        >>> Condorcet.smithSet([[0,-4,-3,-5],[4,0,1,-1],[3,-1,0,2],[5,1,-2,0]])
        {1, 2, 3}
        >>> Condorcet.smithSet([[0,0,-3,-5],[0,0,1,-1],[3,-1,0,2],[5,1,-2,0]])
        {0, 1, 2, 3}
        """
        winCount = [sum(1 if matchup > 0 else 0 for matchup in row) for row in cmat]
        s = set(candID for candID, wins in enumerate(winCount) if wins == max(winCount))
        extensionFound = True
        while extensionFound:
            extensionFound = False
            for cand, matchups in enumerate(cmat):
                if cand not in s and any(matchups[i] >= 0 for i in s):
                    s.add(cand)
                    extensionFound = True
        return s

    @classmethod
    def scenarioType(cls, electorate):
        """Returns the type of scenario presented by the given electorate.

        >>> Condorcet.scenarioType([[2,1,0]]*9 + [[1,0,2]]*8 + [[0,2,1]]*7)
        'cycle'
        >>> Condorcet.scenarioType([[0,1,2]]*2 + [[2,1,0]])
        'easy'
        >>> Condorcet.scenarioType([[0,1,2],[2,1,0]])
        'easy'
        >>> Condorcet.scenarioType([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        'chicken'
        >>> Condorcet.scenarioType([[0,1,2]] * 4 + [[2,1,0]] * 2 + [[1,2,0]] * 3)
        'squeeze'
        >>> Condorcet.scenarioType([[3,2,1,0]] * 5 + [[2,3,1,0]] * 2 + [[0,1,0,3]] * 6 + [[0,0,3,0]] * 3)
        'other'
        >>> Condorcet.scenarioType([[3,0,0,0]] * 5 + [[2,3,0,0]] * 2 + [[0,0,0,3]] * 6 + [[0,0,3,0]] * 3)
        'spoiler'
        """
        ballots = [cls.honBallot(voter) for voter in electorate]
        n = len(electorate[0])
        cmat = cls.compMatrix(ballots)
        numWins = [sum(1 for j, matchup in enumerate(row) if matchup > 0 or (matchup == 0 and i < j))
                for i, row in enumerate(cmat)]
        condOrder = sorted(enumerate(numWins),key=lambda x:-x[1])
        if condOrder[0][1] < n-1:
            return "cycle"
        plurTally = [0] * n
        plur3Tally = [0] * 3
        cond3 = [c for c,v in condOrder[:3]]
        for b in ballots:
            b3 = [b[c] for c in cond3]
            plurTally[b.index(max(b))] += 1
            plur3Tally[b3.index(max(b3))] += 1
        plurOrder = sorted(enumerate(plurTally),key=lambda x:-x[1])
        plur3Order = sorted(enumerate(plur3Tally),key=lambda x:-x[1])
        if plurOrder[0][0] == condOrder[0][0]:
            return "easy"
        elif plur3Order[0][0] == condOrder[0][0]:
            return "spoiler"
        elif plur3Order[2][0] == condOrder[0][0]:
            return "squeeze"
        elif plur3Order[0][0] == condOrder[2][0]:
            return "chicken"
        else:
            return "other"

    @classmethod
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):

        if stratGap > 0:
            others = [c for (c, r) in places[2:]]
            notTooBad = min(voter[frontId], voter[targId])
            decentOnes = [c for c in others if voter[c] >= notTooBad]
            cls.fillPrefOrder(voter, ballot,
                whichCands=decentOnes,
                lowSlot=n-len(decentOnes))
                #ballot[frontId], ballot[targId] = n-len(decentOnes)-1, n-len(decentOnes)-2
            ballot[frontId], ballot[targId] = 0, n-len(decentOnes)-1
            cls.fillPrefOrder(voter, ballot,
                whichCands=[c for c in others if voter[c] < notTooBad],
                lowSlot=1)
        else:
            ballot[frontId] = n - 1
            cls.fillPrefOrder(voter, ballot,
                whichCands=[c for (c, r) in places[1:]],
                lowSlot=0)

    @classmethod #copy-pasted from Irv. The alternatives seemed uglier.
    def compBallot(cls, utils, intensity, candToHelp, candToHurt=None, **kw):
        """Useless unless there's a cycle
        >>> Schulze.compBallot([0,1,2,10],3,1,0)
        [0, 3, 1, 2]
        """
        ballot = cls.honBallot(utils)
        if intensity < 3: return ballot
        helpRank = ballot[candToHelp]
        for cand, rank in enumerate(ballot):
            if rank > helpRank:
                ballot[cand] -= 1
        ballot[candToHelp] = len(utils) - 1
        return ballot

    @classmethod
    def diehardBallot(cls, utils, intensity, candToHurt, candToHelp=None, **kw):
        """Buries candToHurt
        >>> Schulze.diehardBallot([0,1,2,3,4],3,candToHelp=3,candToHurt=2)
        [1, 2, 0, 3, 4]
        """
        ballot = cls.honBallot(utils)
        if intensity < 3: return ballot
        hurtRank = ballot[candToHurt]
        for cand, rank in enumerate(ballot):
            if rank < hurtRank:
                ballot[cand] += 1
        ballot[candToHurt] = 0
        return ballot

    @classmethod
    def defaultbgs(cls):
        return [cls.honBallot]

    @classmethod
    def defaultfgs(cls):
        """
        >>> len(Minimax.defaultfgs())
        12
        """
        return super().defaultfgs()\
        + [(Borda.vaBallot, targs) for targs in [select21, select31]]

class Schulze(Condorcet):

    @classmethod
    def resolveCycle(cls, cmat, n, ballots):

        beatStrength = [[0] * n] * n
        numWins = [0] * n
        for i in range(n):
            for j in range(n):
                if (i != j):
                    if cmat[i][j] > cmat[j][i]:
                        beatStrength[i][j] = cmat[i][j]
                    else:
                        beatStrength[i][j] = 0

                for i in range(n):
                    for j in range(n):
                        if (i != j):
                            for k in range(n):
                                if (i != k and j != k):
                                    beatStrength[j][k] = max ( beatStrength[j][k],
                                        min ( beatStrength[j][i], beatStrength[i][k] ) )

        for i in range(n):
            for j in range(n):
                if i != j:
                    if beatStrength[i][j]>beatStrength[j][i]:
                        numWins[i] += 1
                    if beatStrength[i][j]==beatStrength[j][i] and i<j: #break ties deterministically
                        numWins[i] += 1

        return numWins

class Rp(Condorcet):
    @classmethod
    def resolveCycle(cls, cmat, n, ballots):
        """Note: mutates cmat destructively.

        >>> Rp.resultsFor(DeterministicModel(3)(5,3))
        [0.43333333333333335, 0.4666666666666667, 0.4]
        """
        matches = [(i, j, cmat[i][j]) for i in range(n) for j in range(i,n) if i != j]
        rps = sorted(matches,key=lambda x:-abs(x[2]))
        for (i, j, margin) in rps:
            if margin < 0:
                i, j = j, i
            if cmat[j][i] is True:
                #print("rejecting",cmat)
                pass #reject this victory
            else: #lock-in
                #print(i,j,cmat)
                cmat[i][j] = True
                #print("....",i,j,cmat)
                for k in range(n):
                    if k not in (i, j):
                        if cmat[j][k] is True:
                            cmat[i][k] = True
                        if cmat[k][i] is True:
                            cmat[k][j] = True

                            #print(".......",i,j,k,cmat)

        #print(cmat)
        numWins = [sum(1 for j in range(n) if cmat[i][j] is True)
                    for i in range(n)]
        return numWins

class Minimax(Condorcet):
    """Smith Minimax margins"""
    @classmethod
    def resolveCycle(cls, cmat, n, ballots):
        """ Unnecessary for Minimax; this method is never called.
        """
        smith = cls.smithSet(cmat)
        places = sorted([cand for cand in range(n) if cand not in smith],
                        key=lambda c: min(cmat[c]))\
                + sorted([cand for cand in smith], key=lambda c: min(cmat[c]))
        return [places.index(cand) for cand in range(n)]

    @classmethod
    def results(cls, ballots, **kw):
        """
        >>> Minimax.results([[2,1,0]]*5+[[1,2,0]]*4+[[0,1,2]]*2)
        [0.45454545454545453, 0.5454545454545454, 0.18181818181818182]
        >>> Minimax.results([[2,1,0]]*6 + [[1,0,2]]*5 + [[0,2,1]]*4)
        [0.4, 0.26666666666666666, 0.33333333333333337]
        >>> Minimax.results([[2,1,0]]+[[1,2,0]])
        [0.5, 0.5, 0.0]
        """
        numCands = len(ballots[0])
        cmat = cls.compMatrix(ballots)
        smith = cls.smithSet(cmat)
        if len(smith) == 1:
            results = [0]*numCands
            winner = smith.pop()
            results[winner] = .5 + min(marg for marg in cmat[winner] if marg != 0)/(2*len(ballots))
            for i in range(numCands):
                if i == winner: continue
                results[i] = .5 + cmat[i][winner]/(2*len(ballots))
            return results
        else:
            return [0.5 + min(cmat[cand][i] for i in smith if cand != i)/(2*len(ballots))
                    for cand in range(numCands)]

class Raynaud(Condorcet):
    """Raynaud margins
    """
    @classmethod
    def resolveCycle(cls, cmat, n, *args):
        """
        >>> Raynaud.resolveCycle([[0, 2, -4], [-2, 0, 6], [4, -6, 0]],3)
        [2, 1, 0]
        >>> Raynaud.results([[2,1,0]]*6 + [[1,0,2]]*5 + [[0,2,1]]*4)
        [0.43333333333333335, 0.4, 0.4666666666666667]
        """
        candsLeft = set(range(n))
        results = [0]*n
        numEliminated = 0
        while len(candsLeft) > 1:
            worstMargin = 1
            loser = None
            for cand in candsLeft:
                candWorstMargin = min(cmat[cand][c] for c in candsLeft if c != cand)
                if candWorstMargin < worstMargin:
                    loser = cand
                    worstMargin = candWorstMargin
            results[loser] = numEliminated
            candsLeft.remove(loser)
            numEliminated += 1
        results[candsLeft.pop()] = n - 1
        return results

class SmithIRV(Condorcet):
    """Determines the Smith set, then elect the IRV winner from among it.
    """
    @classmethod
    def resolveCycle(cls, cmat, n, ballots):
        """
        >>> SmithIRV.results([[2,1,0]]*6 + [[1,0,2]]*5 + [[0,2,1]]*4)
        [0.43333333333333335, 0.4, 0.4666666666666667]
        >>> SmithIRV.results([[3,2,0,1]]*6 + [[2,1,0,3]]*5 + [[0,3,1,2]]*4)
        [0.45, 0.42500000000000004, 0.4, 0.47500000000000003]
        """
        smithList = sorted(list(cls.smithSet(cmat)))
        shortenedBallots = [[b[i] for i in smithList] for b in ballots]
        irvResults = Irv.results(shortenedBallots)
        irvTuples = sorted(zip(smithList, irvResults), key=lambda x: x[1])
        otherResults = sorted([(c, min(cmat[c])) for c in range(n) if c not in smithList], key=lambda x: x[1])
        resultTuples = otherResults + irvTuples #worst to best
        ordinalResults = [0]*n
        for i in range(n):
            ordinalResults[resultTuples[i][0]] = i
        return ordinalResults


class IRNR(RankedMethod):
    stratMax = 10

    stratTargetFor = Method.stratTarget3 # strategize in favor of third place, because second place is pointless (can't change pairwise)
    def results(self, ballots, **kwargs):
        enabled = [True] * len(ballots[0])
        numEnabled = sum(enabled)
        results = [None] * len(enabled)
        while numEnabled > 1:
            tsum = [0.0] * len(enabled)
            for bal in ballots:
                vsum = 0.0
                for i, v in enumerate(bal):
                    if enabled[i]:
                        vsum += abs(v)
                if vsum == 0.0:
                    # TODO: count spoiled ballot
                    continue
                for i, v in enumerate(bal):
                    if enabled[i]:
                        tsum[i] += v / vsum
            mini = None
            minv = None
            for i, v in enumerate(tsum):
                if enabled[i]:
                    if (minv is None) or (tsum[i] < minv):
                        minv = tsum[i]
                        mini = i
            enabled[mini] = False
            results[mini] = minv
            numEnabled -= 1
        for i, v in enumerate(tsum):
            if enabled[i]:
                results[i] = tsum[i]
        return results

    @classmethod
    def honBallot(cls, utils, **kw):
        """Takes utilities and returns an honest ballot
        """
        return utils



    @classmethod
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):
        if stratGap <= 0:
            ballot[frontId], ballot[targId] = cls.stratMax, 0
        else:
            ballot[frontId], ballot[targId] = 0, cls.stratMax
        cls.fillPrefOrder(voter, ballot,
            whichCands=[c for (c, r) in places[2:]],
            nSlots = 1, lowSlot=1, remainderScore=0)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
