from dataClasses import *
from methods import *

def makeBlock(method):
    class BlockVoting(method):
        @classmethod
        def winnerSet(cls, ballots, numWinners):
            winners = []
            nCands = len(ballots[0])
            unelectedCands = list(range(nCands))
            while len(winners) < numWinners:
                shortenedBallots = [[b[i] for i in unelectedCands] for b in ballots]
                shortWinnerID = super().winner(super().results(shortenedBallots))
                winner = unelectedCands[shortWinnerID]
                winners.append(winner)
                unelectedCands.remove(winner)
            return winners
    BlockVoting.__name__ = "Block" + method.__name__
    return BlockVoting

class BlockApproval(makeBlock(Approval)): pass #What's used in Fargo
class PBV(makeBlock(Irv)): pass #Preferential BLock Voting, used in Utah
class BlockSTAR(makeBlock(STAR)): pass
class BlockMinimax(makeBlock(Minimax)): pass
class SNTV(makeBlock(Plurality)): pass #not typically viewed as block voting, but this implements it

class weightedBallot(list):
    def __init__(self, *args, **kw):
        self.weight = kw.get('weight', 1)
        super().__init__(*args)

def exactHare(numVoters, numWinners):
    return numVoters/numWinners

def droop(numVoters, numWinners):
    return int(numVoters/(numWinners + 1)) + 1

class RRV(Score):
    @classmethod
    def divisor(cls, winners):
        """D'Hondt"""
        return winners + 1

    @classmethod
    def winnerSet(cls, ballots, numWinners):
        winners = []
        nCands = len(ballots[0])
        unelectedCands = list(range(nCands))
        scoreToWinners = [0]*len(ballots)
        while len(winners) < numWinners:
            candTotals = [-1 if c in winners else 0 for c in range(nCands)]
            for ballot, s in zip(ballots, scoreToWinners):
                for c in unelectedCands:
                    candTotals[c] += ballot[c]/cls.divisor(s/cls.topRank)
            winner = candTotals.index(max(candTotals))
            for i, ballot in enumerate(ballots):
                scoreToWinners[i] += ballot[winner]
            winners.append(winner)
            unelectedCands.remove(winner)
        return winners

class SPAV(RRV):
    """Sequential Proportional Approval Voting
    >>> SPAV.winnerSet([[1,1,1,0]]*10+[[1,1,0,0]]*10+[[0,0,0,1]]*9+[[1,0,0,0]],3)
    [0, 1, 3]
    """
    topRank = 1

class AllocatedScore(STAR):
    """
    >>> AllocatedScore.winnerSet([[5,4,2,0]]*10+[[4,5,0,0]]*10+[[0,1,2,5]]*9+[[5,0,0,0]],3)
    [1, 0, 3]
    >>> AllocatedScore.winnerSet([[5,4,2,0]]*10+[[4,5,0,0]]*10+[[0,0,2,5]]*9+[[5,0,0,0]],3)
    [0, 1, 3]
    >>> AllocatedScore.winnerSet([[5,4,2,0]]*10+[[4,5,0,0]]*10+[[0,0,2,5]]*9+[[5,0,0,0]],2)
    [0, 3]
    >>> AllocatedScore.winnerSet([[5,4,0]]*5+[[4,5,0]]*4+[[0,1,5]]*5,2)
    [1, 2]
    """
    methodQuota = staticmethod(exactHare)
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, *args):
        """
        Picks a winner from unelectedCands using the inputted weighted ballots
        """
        candTotals = [0 if c in unelectedCands else -1 for c in range(len(ballots[0]))]
        for ballot in ballots:
            for c in unelectedCands:
                candTotals[c] += ballot[c]*ballot.weight
        return candTotals.index(max(candTotals))

    @classmethod
    def reweight(cls, ballots, winner, quota):
        """Reweights the inputted (weighted) ballots based on how they voted for winner
        """
        totalSupport = 0
        score = cls.topRank + 1
        while totalSupport < quota:
            score -= 1
            scoreSupport = 0
            for ballot in ballots:
                if ballot[winner] == score:
                    scoreSupport += ballot.weight
            totalSupport += scoreSupport
        surplusFraction = (totalSupport-quota)/scoreSupport
        for ballot in ballots:
            if ballot[winner] > score:
                ballot.weight = 0
            elif ballot[winner] == score:
                ballot.weight *= surplusFraction

    @classmethod
    def winnerSet(cls, ballots, numWinners):
        winners = []
        nCands = len(ballots[0])
        unelectedCands = list(range(nCands))
        wBallots = [weightedBallot(b) for b in ballots]
        #weights = [1]*len(ballots)
        quota = cls.methodQuota(len(ballots), numWinners)
        while True:
            winner = cls.pickWinner(wBallots, unelectedCands, quota)
            unelectedCands.remove(winner)
            winners.append(winner)
            if(len(winners) == numWinners):
                return winners
            cls.reweight(wBallots, winner, quota)
        return winners

    @classmethod
    def twoSlopeBallot(cls, utils, threshold=0.7, thresholdScore=2, **kw):
        """
        >>> AllocatedScore.twoSlopeBallot([0,1,2,3,4,5,6,7,8,9,10,11], threshold=0, thresholdScore=0)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        """
        bot = min(utils)
        top = max(utils)
        meanUtil = sum(utils)/len(utils)
        utilThreshold = (top - meanUtil)*threshold + meanUtil
        ballot = []
        for util in utils:
            if util > utilThreshold:
                ballot.append(floor((cls.topRank - thresholdScore + .99)*(util-utilThreshold)/(top-utilThreshold) + thresholdScore))
            else:
                ballot.append(floor((thresholdScore + .99)*(util-bot)/(utilThreshold-bot)))
        return ballot

    @classmethod
    def vaRangeBallot(cls, utils, electabilities, numWinners, exponent=2, boundConst=0.5, **kw):
        """
        >>> AllocatedScore.vaRangeBallot([0,1,2,3,4,5], [0.5,0,0,0,0,.5], 4, exponent=1)
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> AllocatedScore.vaRangeBallot([0,1,2,3,4,5], [0,0,.5,0,.5,0], 4, exponent=1)
        [0.0, 0.0, 0.0, 2.0, 5.0, 5]
        >>> AllocatedScore.vaRangeBallot([0,1,2,3,4,5], [0.5,0,0,0,0.5,0], 4, exponent=2)
        [0.0, 0.0, 1.0, 3.0, 5.0, 5]
        """
        highBound, lowBound = boundConst/numWinners, boundConst/numWinners
        sortedTuples = sorted(zip(utils, electabilities), key=lambda x: x[0])
        totalProb = 0
        for u, e in sortedTuples:
            totalProb += e
            if totalProb > lowBound:
                lowUtil = u
                break
        totalProb = 0
        for u, e in reversed(sortedTuples):
            totalProb += e
            if totalProb > highBound:
                highUtil = u
                break
        if highUtil == lowUtil:
            return Score.zeroInfoBallot(utils, exponent)
        adjustedUtils = [max(u - lowUtil, 0)**exponent for u in utils]
        return cls.interpolatedBallot(adjustedUtils, 0, (highUtil-lowUtil)**exponent)


class ASFinalRunoff(AllocatedScore):
    """
    Allocated score with a runoff in the final round
    >>> ASFinalRunoff.winnerSet([[5,0,3]]*5+[[0,1,3]]*6, 2)
    [2, 1]
    """
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota):
        if len(ballots[0]) - len(unelectedCands) < len(ballots)/quota - 1.01:
            return super().pickWinner(ballots, unelectedCands)
        candTotals = [0 if c in unelectedCands else -1 for c in range(len(ballots[0]))]
        for ballot in ballots:
            for c in unelectedCands:
                candTotals[c] += ballot[c]*ballot.weight
        (runnerUp,top) = sorted(range(len(ballots[0])), key=lambda i: candTotals[i])[-2:]
        upset = sum(sign(ballot[runnerUp] - ballot[top])*ballot.weight for ballot in ballots)
        if upset > 0:
            return runnerUp
        else: return top

class ASRunoffs(AllocatedScore):
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota):
        candTotals = [sum(b[c]*b.weight for b in ballots) if c in unelectedCands else -1 for c in range(len(ballots[0]))]
        (runnerUp,top) = sorted(range(len(ballots[0])), key=lambda i: candTotals[i])[-2:]
        upset = sum(sign(ballot[runnerUp] - ballot[top])*ballot.weight for ballot in ballots)
        if upset > 0:
            return runnerUp
        else: return top

class ASRDroop(ASRunoffs):
    methodQuota = staticmethod(droop)

class ASR2(ASRunoffs):
    topRank = 25
    @classmethod
    def winnerSet(cls, ballots, numWinners):
        newBallots = [[s**2 for s in ballot] for ballot in ballots]
        return super().winnerSet(newBallots, numWinners)

class SequentialMonroe(AllocatedScore):
    """
    >>> SequentialMonroe.winnerSet([[5,4,0]]*5+[[4,5,0]]*4+[[0,1,5]]*5,2)
    [0, 2]
    """
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota):
        ncand = len(ballots[0])
        quotaStrengths = [0]*ncand
        supportersCounted = [0]*ncand
        quotasLeft = unelectedCands.copy()
        score = cls.topRank + 1
        while quotasLeft and score > 0:
            score -= 1
            for ballot in ballots:
                for c in quotasLeft:
                    if ballot[c] == score:
                        quotaStrengths[c] += score*ballot.weight
                        supportersCounted[c] += ballot.weight
            quotasFilled = set()
            for c in quotasLeft:
                if supportersCounted[c] >= quota:
                    quotaStrengths[c] -= score*(supportersCounted[c] - quota)
                    quotasFilled.add(c)
            for c in quotasFilled:
                quotasLeft.remove(c)
        bestQuota = max(quotaStrengths)
        bestCands = [c for c in unelectedCands if quotaStrengths[c]==bestQuota]
        if len(bestCands) == 1:
            return bestCands[0]
        else:
            scores = [sum(ballot[c]*ballot.weight for ballot in ballots) if c in bestCands else -1 for c in range(ncand)]
            return scores.index(max(scores))

class SSS(AllocatedScore):
    """
    Sequentially Spent Score
    >>> SSS.winnerSet([[5,4,0]]*5+[[4,5,0]]*4+[[0,1,5]]*5,2)
    [1, 2]
    >>> SSS.winnerSet([[5,4,0]]*5+[[4,5,0]]*4+[[0,3,5]]*5,2)
    [1, 0]
    """
    @classmethod
    def reweight(cls, ballots, winner, quota):
        winnerTotal = sum(b[winner]*b.weight for b in ballots)
        winnerSurplus = max(winnerTotal - quota*cls.topRank, 0)
        for ballot in ballots:
            ballot.weight *= (1 - (1 - winnerSurplus/winnerTotal)*ballot[winner]/cls.topRank)

class S5H(SSS):
    """
    Sequentially Spent Score with sorted surplus handling
    >>> S5H.winnerSet([[5,4,0]]*5+[[4,5,0]]*4+[[0,3,5]]*5,2)
    [1, 2]
    """
    @classmethod
    def reweight(cls, ballots, winner, quota):
        """Reweights the inputted (weighted) ballots based on how they voted for winner
        """
        totalSupport = 0
        score = cls.topRank + 1
        while totalSupport < quota*cls.topRank and score > 1:
            score -= 1
            scoreSupport = 0
            for ballot in ballots:
                if ballot[winner] == score:
                    scoreSupport += score*ballot.weight
            totalSupport += scoreSupport
        surplusFraction = max((totalSupport-quota*cls.topRank)/scoreSupport, 0) if scoreSupport else 0
        for ballot in ballots:
            if ballot[winner] > score:
                ballot.weight *= 1 - ballot[winner]/cls.topRank
            elif ballot[winner] == score:
                ballot.weight *= (1 - (1 - surplusFraction)*ballot[winner]/cls.topRank)

class S5HRunoffs(S5H):
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota):
        return ASRunoffs.pickWinner(ballots, unelectedCands, quota)

class SSSRunoffs(SSS):
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota):
        return ASRunoffs.pickWinner(ballots, unelectedCands, quota)

class S5HRDroop(S5HRunoffs):
    methodQuota = staticmethod(droop)

class STV(Irv):
    """Weighted inclusive Gregory method (I think) with the Droop quota
    >>> STV.winnerSet([[0,1,2,3,4,5,6]]*40+[[6,5,4,3,2,1,0]]*10, 4)
    [6, 5, 4, 0]
    >>> STV.winnerSet([[3,2,1,0]]*10+[[0,2,1,3]]*10+[[0,2,3,1]]*12, 2)
    [2, 3]
    """
    @classmethod
    def winnerSet(cls, ballots, numWinners):
        quota = int(len(ballots)/(numWinners+1) + 1)
        winners = []
        nCands = len(ballots[0])
        candsLeft = set(range(nCands))
        wBallots = [weightedBallot(b) for b in ballots]
        ballotsToSort = wBallots
        piles = [[] for i in range(nCands)]
        while len(winners) + len(candsLeft) > numWinners:
            cls.resort(ballotsToSort, candsLeft, piles)
            newWinners = [c for c in candsLeft if sum(b.weight for b in piles[c]) >= quota]
            if newWinners:
                ballotsToSort = []
                winners.extend(newWinners)
                for w in newWinners:
                    totalVotes = sum(b.weight for b in piles[w])
                    reweight = (totalVotes-quota)/totalVotes
                    for b in piles[w]:
                        b.weight *= reweight
                    candsLeft.remove(w)
                    ballotsToSort.extend(piles[w])
            else:
                loser, loserVotes = None, float('inf')
                for cand in candsLeft: #determine who gets eliminated
                    if sum(b.weight for b in piles[cand]) < loserVotes:
                        loser = cand
                        loserVotes = sum(b.weight for b in piles[cand])
                candsLeft.remove(loser)
                ballotsToSort = piles[loser]
        for c in candsLeft:
            winners.append(c)
        return winners
