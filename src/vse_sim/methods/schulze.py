from numpy import sign

from ..voter_models import DeterministicModel  # noqa: F401
from .borda import RankedMethod


class Schulze(RankedMethod):
    def resolveCycle(self, cmat, n):

        beatStrength = [[0] * n for _ in range(n)]
        numWins = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    beatStrength[i][j] = cmat[i][j] if cmat[i][j] > cmat[j][i] else 0

        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in range(n):
                        if i != k and j != k:
                            beatStrength[j][k] = max(
                                beatStrength[j][k],
                                min(beatStrength[j][i], beatStrength[i][k]),
                            )

        for i in range(n):
            for j in range(n):
                if i != j:
                    if beatStrength[i][j]>beatStrength[j][i]:
                        numWins[i] += 1
                    if beatStrength[i][j]==beatStrength[j][i] and i<j: #break ties deterministically
                        numWins[i] += 1

        return numWins

    def results(self, ballots, isHonest=False, **kwargs):
        """Schulze results.

        >>> schulze = Schulze()
        >>> schulze.resultsFor(DeterministicModel(3)(5,3),schulze.honBallot,isHonest=True)["results"]
        [1, 2, 0]
        >>> schulze.extraEvents
        {'scenario': 'cycle'}
        >>> schulze.results([[0,1,2]],isHonest=True)[2]
        2
        >>> schulze.extraEvents
        {'scenario': 'easy'}
        >>> schulze.results([[0,1,2],[2,1,0]],isHonest=True)[1]
        1
        >>> schulze.extraEvents
        {'scenario': 'easy'}
        >>> schulze.results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2,isHonest=True)
        [1, 2, 0]
        >>> schulze.extraEvents
        {'scenario': 'chicken'}
        >>> schulze.results([[0,1,2]] * 4 + [[2,1,0]] * 2 + [[1,2,0]] * 3,isHonest=True)
        [1, 2, 0]
        >>> schulze.extraEvents
        {'scenario': 'squeeze'}
        >>> schulze.results([[3,2,1,0]] * 5 + [[2,3,1,0]] * 2 + [[0,1,0,3]] * 6 + [[0,0,3,0]] * 3,isHonest=True)
        [2, 3, 1, 0]
        >>> schulze.extraEvents
        {'scenario': 'other'}
        >>> schulze.results([[3,0,0,0]] * 5 + [[2,3,0,0]] * 2 + [[0,0,0,3]] * 6 + [[0,0,3,0]] * 3,isHonest=True)
        [3, 0, 1, 2]
        >>> schulze.extraEvents
        {'scenario': 'spoiler'}
        """
        n = len(ballots[0])
        cmat = [[0 for _ in range(n)] for _ in range(n)]
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

        if isHonest:
            self.extraEvents = {}
            plurTally = [0] * n
            plur3Tally = [0] * 3
            cond3 = [c for c,v in condOrder[:3]]
            if condOrder is None:
                condOrder = sorted(enumerate(result),key=lambda x:-x[1])
            for b in ballots:
                b3 = [b[c] for c in cond3]
                plurTally[b.index(max(b))] += 1
                plur3Tally[b3.index(max(b3))] += 1
            plurOrder = sorted(enumerate(plurTally),key=lambda x:-x[1])
            plur3Order = sorted(enumerate(plur3Tally),key=lambda x:-x[1])
            if cycle:
                self.extraEvents["scenario"] = "cycle"
            elif plurOrder[0][0] == condOrder[0][0]:
                self.extraEvents["scenario"] = "easy"
            elif plur3Order[0][0] == condOrder[0][0]:
                self.extraEvents["scenario"] = "spoiler"
            elif plur3Order[2][0] == condOrder[0][0]:
                self.extraEvents["scenario"] = "squeeze"
            elif plur3Order[0][0] == condOrder[2][0]:
                self.extraEvents["scenario"] = "chicken"
            else:
                self.extraEvents["scenario"] = "other"

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
            ballot[frontId], ballot[targId] = 0, n-len(decentOnes)-1
            cls.fillPrefOrder(voter, ballot,
                whichCands=[c for c in others if voter[c] < notTooBad],
                lowSlot=1)
        else:
            ballot[frontId] = n - 1
            cls.fillPrefOrder(voter, ballot,
                whichCands=[c for (c, r) in places[1:]],
                lowSlot=0)
