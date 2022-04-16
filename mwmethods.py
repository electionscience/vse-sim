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
            candTotals = [0]*nCands
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
    """
    @classmethod
    def pickWinner(cls, ballots, weights, unelectedCands, *args):
        """
        Picks a winner from unelectedCands using ballots and weights
        """
        candTotals = [0]*len(ballots[0])
        for ballot, w in zip(ballots, weights):
            for c in unelectedCands:
                candTotals[c] += ballot[c]*w
        return candTotals.index(max(candTotals))

    @classmethod
    def winnerSet(cls, ballots, numWinners):
        winners = []
        nCands = len(ballots[0])
        unelectedCands = list(range(nCands))
        weights = [1]*len(ballots)
        quota = len(ballots)/numWinners #exact Hare
        while True:
            winner = cls.pickWinner(ballots, weights, unelectedCands, quota)
            unelectedCands.remove(winner)
            winners.append(winner)
            if(len(winners) == numWinners):
                return winners

            #allocate voters to the winner's quota
            totalSupport = 0
            score = cls.topRank + 1
            while totalSupport < quota:
                score -= 1
                scoreSupport = 0
                for ballot, weight in zip(ballots, weights):
                    if ballot[winner] == score:
                        scoreSupport += weight
                totalSupport += scoreSupport
            surplusFraction = (totalSupport-quota)/scoreSupport
            for i in range(len(ballots)):
                if ballots[i][winner] > score:
                    weights[i] = 0
                elif ballots[i][winner] == score:
                    weights[i] *= surplusFraction
        return winners

class SequentialMonroe(AllocatedScore):
    @classmethod
    def pickWinner(cls, ballots, weights, unelectedCands, quota):
        ncand = len(ballots[0])
        quotaStrengths = [0]*ncand
        supportersCounted = [0]*ncand
        quotasLeft = unelectedCands.copy()
        score = cls.topRank + 1
        while quotasLeft and score > 0:
            score -= 1
            for ballot, weight in zip(ballots, weights):
                for c in quotasLeft:
                    if ballot[c] == score:
                        quotaStrengths[c] += score*weight
                        supportersCounted[c] += weight
            for c in quotasLeft:
                if supportersCounted[c] >= quota:
                    quotaStrengths[c] -= score*(supportersCounted[c] - quota)
                    quotasLeft.remove(c)
        bestQuota = max(quotaStrengths)
        bestCands = [c for c in unelectedCands if quotaStrengths[c]==bestQuota]
        if len(bestCands) == 1:
            return bestCands[0]
        else:
            scores = [sum(ballot[c]*weight for ballot, weight in zip(ballots, weights)) if c in bestCands else -1 for c in range(ncand)]
            return scores.index(max(scores))

class weightedBallot(list):
    def __init__(self, *args, **kw):
        self.weight = kw.get('weight', 1)
        super().__init__(*args)

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
