from ..core import Method, rememberBallot
from .ranked import RankedMethod


class IRNR(RankedMethod):
    """Implement Instant Runoff Normalized Ratings.

    In each round, every ballot is normalized by the absolute ratings of its
    remaining candidates. The candidate with the lowest normalized total is
    eliminated until one remains.
    """

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
            for i, _v in enumerate(tsum):
                if enabled[i] and ((minv is None) or (tsum[i] < minv)):
                    minv = tsum[i]
                    mini = i
            enabled[mini] = False
            results[mini] = minv
            numEnabled -= 1
        for i, _v in enumerate(tsum):
            if enabled[i]:
                results[i] = tsum[i]
        return results

    @staticmethod #cls is provided explicitly, not through binding
    @rememberBallot
    def honBallot(cls, utils):
        """Takes utilities and returns an honest ballot.
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
