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
    NOT the same Allocated Score as is on Electowiki
    This version doesn't care about ballot weights for who gets put in a quota
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
        surplusFraction = max(0, (totalSupport-quota)/scoreSupport)
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
        quota = cls.methodQuota(len(ballots), numWinners)
        while True:
            winner = cls.pickWinner(wBallots, unelectedCands, quota, numWinners)
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

    @classmethod
    def truncatedBallot(cls, utils, threshold=0, exp=1, **kw):
        """
        >>> AllocatedScore.truncatedBallot([0,1,2,3,4,5,6], threshold=0, exp=3)
        [0, 0, 0, 0, 1.0, 3.0, 5.0]
        """
        baseBallot = Score.zeroInfoBallot(utils, exp)
        meanUtil = sum(utils)/len(utils)
        bestUtil = max(utils)
        return [baseBallot[i] if (utils[i]-meanUtil)/(bestUtil-meanUtil)>threshold else 0
                for i in range(len(utils))]

    @classmethod
    def truncatedBallot2(cls, utils, threshold=0, exp=1, **kw):
        """
        >>> AllocatedScore.truncatedBallot2([0,1,2,3,4,5,6], threshold=3, exp=2)
        [0, 0, 0, 0, 0, 4.0, 5.0]
        """
        baseBallot = Score.zeroInfoBallot(utils, exp)
        return [s if s >= threshold else 0 for s in baseBallot]

class ASC(AllocatedScore):
    """Allocated Score Classic, the version on Electowiki
    >>> ASC.winnerSet([[5,4,0,5]]*10+[[4,3,5,0]]*5,3)
    [0, 1, 3]
    >>> AllocatedScore.winnerSet([[5,4,0,5]]*10+[[4,3,5,0]]*5,3)
    [0, 1, 2]
    """
    @classmethod
    def reweight(cls, ballots, winner, quota):
        sortedBallots = sorted(ballots, key=lambda b: -b[winner]*b.weight)
        totalSupport = 0
        scoreSupport = 0
        lastScore = -1
        for b in sortedBallots:
            bScore = b[winner]*b.weight
            if bScore == 0 or (totalSupport > quota and bScore != lastScore):
                break
            totalSupport += b.weight
            if bScore == lastScore:
                scoreSupport += b.weight
            else:
                lastScore = bScore
                scoreSupport = b.weight
        surplusFraction = max(0,(totalSupport-quota)/scoreSupport) if scoreSupport > 0 else 0
        for ballot in ballots:
            if ballot[winner]*ballot.weight > lastScore:
                ballot.weight = 0
            elif ballot[winner]*ballot.weight == lastScore:
                ballot.weight *= surplusFraction

class ASCD(ASC):
    methodQuota = staticmethod(droop)

class ASCDFinalRunoff(ASCD):
    """
    >>> ASCDFinalRunoff.winnerSet([[5,0]]+[[0,1]]*3, 1)
    [1]
    >>> ASCDFinalRunoff.winnerSet([[5,2,0]]*5+[[4,5,0]]*5+[[0,1,5]]*6, 2)
    [0, 2]
    """
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
        return ASFinalRunoff.pickWinner(ballots, unelectedCands, quota, numWinners)

class ASCDR(ASCD):
    """
    Classic Allocated Score with the Droop quota and a runoff in every round
    >>> ASCDR.winnerSet([[5,2,0]]*5+[[4,5,0]]*5+[[0,1,5]]*6, 2)
    [1, 2]
    """
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
        return ASRunoffs.pickWinner(ballots, unelectedCands, quota, numWinners)

class ASFinalRunoff(AllocatedScore):
    """
    Allocated score with a runoff in the final round
    >>> ASFinalRunoff.winnerSet([[5,0,3]]*5+[[0,1,3]]*6, 2)
    [2, 1]
    """
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
        if len(ballots[0]) - len(unelectedCands) < numWinners - 1:
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
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
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
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
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
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
        return ASRunoffs.pickWinner(ballots, unelectedCands, quota, numWinners)

class SSSRunoffs(SSS):
    @classmethod
    def pickWinner(cls, ballots, unelectedCands, quota, numWinners):
        return ASRunoffs.pickWinner(ballots, unelectedCands, quota, numWinners)

class S5HRDroop(S5HRunoffs):
    methodQuota = staticmethod(droop)

class TEA(AllocatedScore):
    """Threshold Equal Approvals"""
    @classmethod
    def winnerSet(cls, ballots, numWinners):
        threshold = cls.topRank
        numCands = len(ballots[0])
        quota = cls.methodQuota(len(ballots), numWinners)
        wBallots = [weightedBallot(b) for b in ballots]
        unelectedCands = list(range(numCands))
        winners = []
        while threshold > 0 and len(winners) < numWinners:
            approvalCounts = [sum(ballot.weight if ballot[i] >= threshold else 0 for ballot in wBallots)
                                if i in unelectedCands else -1 for i in range(numCands)]
            if any(c >= quota for c in approvalCounts):
                costs = [2]*numCands
                for cand in unelectedCands:
                    if approvalCounts[cand] >= quota:
                        sortedWeights = sorted(b.weight for b in wBallots if b[cand] >= threshold and b.weight > 0)
                        weightSum = 0
                        for i, w in enumerate(sortedWeights):
                            if weightSum + w*(len(sortedWeights)-i) >= quota or w==sortedWeights[-1]:
                                costs[cand] = (quota - weightSum)/(len(sortedWeights)-i)
                                break
                            else:
                                weightSum += w
                winnerCost = min(costs)
                winner = costs.index(winnerCost)
                if winner not in unelectedCands: print(costs, approvalCounts)
                for b in wBallots:
                    if b[winner] >= threshold:
                        b.weight = max(b.weight - winnerCost, 0)
                unelectedCands.remove(winner)
                winners.append(winner)
            else: threshold -= 1
            #print(threshold, winners, [b.weight for b in wBallots])
        while len(winners) < numWinners:
            approvalCounts = [sum(ballot.weight if ballot[i] > 0 else 0 for ballot in wBallots)
                                if i in unelectedCands else -1 for i in range(numCands)]
            winner = approvalCounts.index(max(approvalCounts))
            unelectedCands.remove(winner)
            for b in wBallots:
                if b[winner] > 0:
                    b.weight = 0
            winners.append(winner)
            #print(threshold, winners, [b.weight for b in wBallots])
        return winners


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

class MinimaxSTV(STV):
    """
    Uses STV to elect all but the last winner. Uses minimax (with eliminated candidates readded) in the final round.
    Uses plain minimax(margins) instead of Smith//minimax due to laziness
    >>> MinimaxSTV.winnerSet([[0,1,2,3]]*5+[[2,1,0,3]]*6+[[0,2,1,3]]*2,2)
    [3, 1]
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
        while len(winners) < numWinners - 1 and len(winners) + len(candsLeft) > numWinners - 1:
            cls.resort(ballotsToSort, candsLeft, piles)
            newWinners = [c for c in candsLeft if sum(b.weight for b in piles[c]) >= quota][:max(0, numWinners-len(winners)-1)]
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
        if len(winners) < numWinners - 1:
            for c in candsLeft:
                winners.append(c)

        #Minimax step
        compMatrix = [[float('inf') if i in winners or j in winners
                        else sum(b.weight*sign(b[i] - b[j]) for b in wBallots)
                        for j in range(nCands)] for i in range(nCands)]
        best = float('-inf')
        for i, row in enumerate(compMatrix):
            if i not in winners and min(row) > best:
                best = min(row)
                finalWinner = i
        winners.append(finalWinner)
        return winners

def assignBallot(ballot, candsLeft):
    activeScore = max(ballot[i] for i in candsLeft)
    if activeScore == 0:
        return set()
    else:
        return set(i for i in candsLeft if ballot[i] == activeScore)

class S5HtoSTV(S5H):
    """Uses S5H for as long as a Droop quota can be filled, then switches to STV
    """
    methodQuota = staticmethod(droop)
    @classmethod
    def winnerSet(cls, ballots, numWinners):
        """
        >>> S5HtoSTV.winnerSet([[0,1,2]]*5+[[2,1,0]]*4+[[0,2,1]]*2,2)
        [2, 0]
        >>> S5HtoSTV.winnerSet([[0,1,5]]*5+[[5,1,0]]*4+[[0,5,1]]*2,2)
        [2, 0]
        >>> S5HtoSTV.winnerSet([[0,4,5]]*5+[[5,4,0]]*4+[[0,5,4]]*2,2)
        [1, 2]
        """
        quota = cls.methodQuota(len(ballots), numWinners)
        wBallots = [weightedBallot(b) for b in ballots]
        winners = []
        unelectedCands = list(range(len(ballots[0])))
        while any(sum(b[i]*b.weight for b in wBallots) >= quota*cls.topRank for i in unelectedCands):
            winner = cls.oneS5HRound(wBallots, unelectedCands, quota, numWinners)
            winners.append(winner)
            unelectedCands.remove(winner)
        if len(winners) < numWinners:
            cls.useSTV(wBallots, numWinners, winners, quota, True)
        return winners

    @classmethod
    def oneS5HRound(cls, ballots, unelectedCands, quota, numWinners):
        winner = S5H.pickWinner(ballots, unelectedCands, quota, numWinners)
        S5H.reweight(ballots, winner, quota)
        return winner

    @classmethod
    def useSTV(cls, wBallots, numWinners, winners, quota, useForLastRound):
        numToElect = numWinners if useForLastRound else numWinners - 1
        nCands = len(wBallots[0])
        candsLeft = set(i for i in range(nCands) if i not in winners)
        assignedBallots = [[b, assignBallot(b, candsLeft)] for b in wBallots]
        while len(winners) < numToElect and len(winners) + len(candsLeft) > numToElect:
            candTotals = [sum(b.weight/len(topCands) for b, topCands in assignedBallots if cand in topCands)
                            if cand in candsLeft else -1 for cand in range(nCands)]
            if max(candTotals) >= quota:
                winner = candTotals.index(max(candTotals))
                weightFactor = (candTotals[winner] - quota)/candTotals[winner]
                winners.append(winner)
                candsLeft.remove(winner)
                for b, topCands in assignedBallots:
                    if winner in topCands:
                        b.weight *= weightFactor
                        topCands.remove(winner)
                        if len(topCands) == 0:
                            topCands.update(assignBallot(b, candsLeft))
            else:
                loser = candTotals.index(min(t for t in candTotals if t != -1))
                candsLeft.remove(loser)
                for b, topCands in assignedBallots:
                    if loser in topCands:
                        topCands.remove(loser)
                        if len(topCands) == 0:
                            topCands.update(assignBallot(b, candsLeft))
        if len(winners) < numToElect:
            for c in candsLeft:
                winners.append(c)
