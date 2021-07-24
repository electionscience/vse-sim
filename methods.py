
from mydecorators import autoassign, cached_property, setdefaultattr, decorator
import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor, ceil
from numpy import percentile, argsort, sign
from test.test_binop import isnum
from debugDump import *
from math import log

from stratFunctions import *
from dataClasses import *


class Method(BaseMethod):
    """This is separate from BaseMethod to avoid circular dependencies
    """
    @classmethod
    def threeRoundResults(cls, voters, backgroundStrat, foregrounds=[],
                          r1Media=(lambda x:x), pickiness=0.4):
        """
        Performs three elections: a single approval voting contest in which everyone
        votes honestly to give an intentionally crude estimate of electability
        (which is filtered by r1Media),
        then an election using no information beyond the first round of "polling" in which all voters
        use backgroundStrat, and a third round which may use the results of both the prior rounds.
        The third round is repeated for each choice of foreground.
        A foreground is a (foregroundSelectionFunction, foregroundStrat, media) tuple
        where foregroundSelectionFunction receives the input of
        (voter, electabilities, media(round1Results)) and returns a positive float representing
        the voter's eagerness to be strategic if the voter will be part of the strategic foregrounds
        and 0 if the voter will just use backgroundStrat
        """
        if isinstance(backgroundStrat, str):
            backgroundStrat = getattr(cls, backgroundStrat)
        if isinstance(foregrounds, tuple):
            foregrounds = [foregrounds]
        for i, f in enumerate(foregrounds):
            if len(f) == 2: #if media isn't provided, default to truth
                foregrounds[i] = (f[0], f[1], lambda x:x)

        r0Results = Approval.results([useStrat(voter, Approval.zeroInfoBallot, pickiness=pickiness)
        for voter in voters])
        r0Winner = cls.winner(r0Results)
        electabilities = tuple(r1Media(r0Results))
        backgroundBallots = [useStrat(voter, backgroundStrat, electabilities=electabilities) for voter in voters]
        r1Results = cls.results(backgroundBallots)
        r1Winner = cls.winner(r1Results)

        allResults = [makeResults(results=r0Results, totalUtil=voters.socUtils[r0Winner]),
        makeResults(results=r1Results, totalUtil=voters.socUtils[r1Winner])]
        for foregroundSelect, foregroundStrat, r2Media in foregrounds:
            polls = tuple(r2Media(r1Results))
            foreground = {(ID, voter) for ID, voter in enumerate(voters)
                          if foregroundSelect(voter, electabilities, polls)}
            ballots = [useStrat(voter, foregroundStrat, polls=polls, electabilities=electabilities)
                       if (i, voter) in foreground
                       else backgroundBallots[i] for i, voter in enumerate(voters)]
            results = cls.results(ballots)
            winner = cls.winner(results)
            foregroundBaseUtil = sum(voter[1][r1Winner] for voter in foreground)/len(foreground)
            foregroundStratUtil = sum(voter[1][winner] for voter in foreground)/len(foreground)
            totalUtil = voters.socUtils[winner]
            allResults.append(makeResults(results=results, foregroundUtil=foregroundStratUtil,
            foregroundUtilDiff=foregroundStratUtil-foregroundBaseUtil, totalUtil=totalUtil))
        return allResults


####EMs themselves
class Borda(Method):
    candScore = staticmethod(mean)

    nRanks = 999 # infinity

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
    def lowInfoBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.1):
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

        >>> Borda().stratBallotFor([4,5,2,1])(Borda, Voter([-4,-5,-2,-1]))
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

    @staticmethod
    def oneVote(utils, forWhom):
        ballot = [0] * len(utils)
        ballot[forWhom] = 1
        return ballot

    @classmethod
    def honBallot(cls, utils, **kw):
        """Takes utilities and returns an honest ballot

        >>> Plurality.honBallot(Plurality, Voter([-3,-2,-1]))
        [0, 0, 1]
        >>> Plurality().stratBallotFor([3,2,1])(Plurality, Voter([-3,-2,-1]))
        [0, 1, 0]
        """
        #return cls.oneVote(utils, cls.winner(utils))
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot,
            nSlots = 1, lowSlot=1, remainderScore=0)
        return ballot

    @classmethod
    def lowInfoBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.07):
        winProbs = pollsToProbs(electabilities, pollingUncertainty)
        expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
        scores = [(u - expectedUtility)*p for u, p in zip(utils, winProbs)]
        return cls.oneVote(scores, scores.index(max(scores)))
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



def Score(topRank=10, asClass=False):
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
        candScore = staticmethod(mean)
            #"""Takes the list of votes for a candidate; returns the candidate's score."""


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
            raise Exception("NOT")
            bot = min(utils)
            scale = max(utils)-bot
            return [floor((cls.topRank + .99) * (util-bot) / scale) for util in utils]

        @classmethod
        def lowInfoBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.07):
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
            expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
            return [cls.topRank if u > expectedUtility else 0 for u in utils]

        @classmethod
        def lowInfoIntermediateBallot(cls, utils, electabilities, polls=None,
        pollingUncertainty=.07, midScoreWillingness=0.7):
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
            expectedUtility = sum(u*p for u, p in zip(utils, winProbs))
            if all(u == utils[0] for u in utils[1:]):
                return [0]*len(utils)
            lowThreshold = max(min(utils), expectedUtility - midScoreWillingness*std(utils))
            highThreshold = min(max(utils), expectedUtility + midScoreWillingness*std(utils))
            #this is wrong. We should use a standard deviation that's weighted
            #by each candidate's chance of winning
            ballot = []
            for util in utils:
                if util < lowThreshold:
                    ballot.append(0)
                elif util > highThreshold:
                    ballot.append(cls.topRank)
                else:
                    ballot.append(floor((cls.topRank + .99)*(util-lowThreshold)/(highThreshold-lowThreshold)))
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

class Approval(Score(1,True)):
    @classmethod
    def zeroInfoBallot(cls, utils, electabilities=None, polls=None, pickiness=0):
        """Returns a ballot based on utils and pickiness
        pickiness=0 corresponds to lowInfoBallot with equal polling for all candidates
        pickiness=1 corresponds to bullet voting
        """
        expectedUtility = sum(u for u in utils)/len(utils)
        best = max(utils)
        normalizedUtils = [(u - expectedUtility)/(best - expectedUtility)
                           for u in utils]
        return [1 if u >= pickiness else 0 for u in normalizedUtils]

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


def STAR(topRank=5):
    "STAR Voting"

    score0to = Score(topRank,True)

    class STAR0to(score0to):

        stratTargetFor = Method.stratTarget3

        @classmethod
        def results(self, ballots, **kwargs):
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
            baseResults = super(STAR0to, self).results(ballots, **kwargs)
            (runnerUp,top) = sorted(range(len(baseResults)), key=lambda i: baseResults[i])[-2:]
            upset = sum(sign(ballot[runnerUp] - ballot[top]) for ballot in ballots)
            if upset > 0:
                baseResults[runnerUp] = baseResults[top] + 0.01
            return baseResults

        @classmethod
        def lowInfoBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.07, scoreImportance=0.17):
            winProbs = pollsToProbs(electabilities, pollingUncertainty)
            #runoffCoefficients[i][j] is how valuable it is to score i over j
            runoffCoefficients = [[(u1 - u2)*p1*p2
                                   for u2, p2 in zip(utils, winProbs)]
                                  for u1, p1 in zip(utils, winProbs)]
            unnormalizedRunnerUpProbs = [p*(1-p) for p in winProbs]
            normFactor = sum(unnormalizedRunnerUpProbs)
            runnerUpProbs = [u/normFactor for u in unnormalizedRunnerUpProbs]
            eRunnerUpUtil = sum(u*p for u, p in zip(utils, runnerUpProbs))
            #scoreCoefficients[i] is how vauable it is for i to have a high score
            scoreCoefficients = [scoreImportance*(u-eRunnerUpUtil)*p
                                 for u, p in zip(utils, runnerUpProbs)]

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

    return STAR0to()


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
            >>> Mav().honBallot(Mav, Voter([-1,-0.5,0.5,1,1.1]))
            [0, 1, 2, 3, 4]

        Even if they don't rate at least an honest "B":
            >>> Mav().honBallot(Mav, Voter([-1,-0.5,0.5]))
            [0, 1, 4]
        """
        cuts = cls.specificCuts if (cls.specificCuts is not None) else cls.baseCuts
        cuts = [min(cut, max(voter) - 0.001) for cut in cuts]
        return [toVote(cuts, util) for util in voter]


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

    def resort(self, ballots, loser, ncand, piles):
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
    def results(self, ballots, **kwargs):
        """IRV results.

        >>> Irv().resultsFor(DeterministicModel(3)(5,3),Irv().honBallot)["results"]
        [0, 1, 2]
        >>> Irv().results([[0,1,2]])[2]
        2
        >>> Irv().results([[0,1,2],[2,1,0]])[1]
        0
        >>> Irv().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
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

        >>> Irv.honBallot(Irv,Voter([4,1,6,3]))
        [2, 0, 3, 1]
        """
        ballot = [-1] * len(voter)
        order = sorted(enumerate(voter), key=lambda x:x[1])
        for i, cand in enumerate(order):
            ballot[cand[0]] = i
        #print("hballot",ballot)
        return ballot

    @classmethod
    def lowInfoBallot(cls, utils, electabilities, polls=None, pollingUncertainty=.1):
        """Electabilities should be interpreted as a metric for the ability to win in the final round.
        """
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
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):
        """
        >>> Irv().stratBallotFor([3,2,1,0])(Irv,Voter([3,6,5,2]))
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

    def results(self, ballots, isHonest=False, **kwargs):
        """3-2-1 Voting results.

        >>> V321().resultsFor(DeterministicModel(3)(5,3),V321().honBallot)["results"]
        [-0.75, 2, 1]
        >>> V321().results([[0,1,2]])[2]
        2
        >>> V321().results([[0,1,2],[2,1,0]])[1]
        2.5
        >>> V321().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [1, 1.5, -0.25]
        >>> V321().results([[0,1,2,1]]*29 + [[1,2,0,1]]*30 + [[2,0,1,1]]*31 + [[1,1,1,2]]*10)
        [3, 0.5, 1, 0]
        >>> V321().results([[1,0,2,1]]*29 + [[0,2,1,1]]*30 + [[2,1,0,1]]*31 + [[1,1,1,2]]*10)
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


        >>> Irv().stratBallotFor([3,2,1,0])(Irv,Voter([3,6,5,2]))
        [1, 2, 3, 0]
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

class Schulze(RankedMethod):
    def resolveCycle(self, cmat, n):

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

    @classmethod
    def results(self, ballots, isHonest=False, **kwargs):
        """Schulze results.

        >>> Schulze().resultsFor(DeterministicModel(3)(5,3),Schulze().honBallot,isHonest=True)["results"]
        [2, 0, 1]
        >>> Schulze.extraEvents
        {'scenario': 'cycle'}
        >>> Schulze().results([[0,1,2]],isHonest=True)[2]
        2
        >>> Schulze.extraEvents
        {'scenario': 'easy'}
        >>> Schulze().results([[0,1,2],[2,1,0]],isHonest=True)[1]
        1
        >>> Schulze.extraEvents
        {'scenario': 'easy'}
        >>> Schulze().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2,isHonest=True)
        [1, 2, 0]
        >>> Schulze.extraEvents
        {'scenario': 'chicken'}
        >>> Schulze().results([[0,1,2]] * 4 + [[2,1,0]] * 2 + [[1,2,0]] * 3,isHonest=True)
        [1, 2, 0]
        >>> Schulze.extraEvents
        {'scenario': 'squeeze'}
        >>> Schulze().results([[3,2,1,0]] * 5 + [[2,3,1,0]] * 2 + [[0,1,0,3]] * 6 + [[0,0,3,0]] * 3,isHonest=True)
        [2, 3, 1, 0]
        >>> Schulze.extraEvents
        {'scenario': 'other'}
        >>> Schulze().results([[3,0,0,0]] * 5 + [[2,3,0,0]] * 2 + [[0,0,0,3]] * 6 + [[0,0,3,0]] * 3,isHonest=True)
        [3, 0, 1, 2]
        >>> Schulze.extraEvents
        {'scenario': 'spoiler'}
        """
        n = len(ballots[0])
        cmat = [[0 for i in range(n)] for j in range(n)]
        numWins = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    cmat[i][j] = sum(sign(ballot[i] - ballot[j]) for ballot in ballots)
                    if cmat[i][j]>0:
                        numWins[i] += 1
                    elif cmat[i][j]==0 and i<j:
                        numWins[i] += 1
        condOrder = sorted(enumerate(numWins),key=lambda x:-x[1])
        if condOrder[0][1] == n-1:
            cycle = 0
            result = numWins
        else: #cycle
            cycle = 1
            result = self.resolveCycle(cmat, n)
            order = None

        if isHonest:
            self.__class__.extraEvents = dict()
            #check scenarios
            plurTally = [0] * n
            plur3Tally = [0] * 3
            cond3 = [c for c,v in condOrder[:3]]
            if condOrder==None:
                condOrder = sorted(enumerate(result),key=lambda x:-x[1])
            for b in ballots:
                b3 = [b[c] for c in cond3]
                plurTally[b.index(max(b))] += 1
                plur3Tally[b3.index(max(b3))] += 1
            plurOrder = sorted(enumerate(plurTally),key=lambda x:-x[1])
            plur3Order = sorted(enumerate(plur3Tally),key=lambda x:-x[1])
            if cycle:
                self.__class__.extraEvents["scenario"] = "cycle"
            elif plurOrder[0][0] == condOrder[0][0]:
                self.__class__.extraEvents["scenario"] = "easy"
            elif plur3Order[0][0] == condOrder[0][0]:
                self.__class__.extraEvents["scenario"] = "spoiler"
            elif plur3Order[2][0] == condOrder[0][0]:
                self.__class__.extraEvents["scenario"] = "squeeze"
            elif plur3Order[0][0] == condOrder[2][0]:
                self.__class__.extraEvents["scenario"] = "chicken"
            else:
                self.__class__.extraEvents["scenario"] = "other"

        return result


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

class Rp(Schulze):
    def resolveCycle(self, cmat, n):
        """Note: mutates cmat destructively.

        >>> Rp().resultsFor(DeterministicModel(3)(5,3),Rp().honBallot,isHonest=True)["results"]
        [1, 2, 0]
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

class Minimax(Schulze):
    """Smith Minimax margins"""
    def resolveCycle(self, cmat, n):
        winCount = [sum(1 if matchup > 0 else 0 for matchup in row) for row in cmat]
        smithSet = set(candID for candID, wins in enumerate(winCount) if wins == max(winCount))
        extensionFound = True
        while extensionFound:
            extensionFound = False
            for cand, matchups in enumerate(cmat):
                if cand not in smithSet and any(matchups[i] > 0 for i in smithSet):
                    smithSet.add(cand)
                    extensionFound = True
        places = sorted([cand for cand in range(n) if cand not in smithSet],
                        key=lambda c: min(cmat[c]))\
    + sorted([cand for cand in smithSet], key=lambda c: min(cmat[c]))
        return [places.index(cand) for cand in range(n)]


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
