from numpy import mean

from ..core import Method, rememberBallot
from ..voter_models import Voter  # noqa: F401


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

    @staticmethod #cls is provided explicitly, not through binding
    @rememberBallot
    def honBallot(cls, utils):
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot)
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

RankedMethod = Borda #alias
RatedMethod = RankedMethod #Should have same strategies available, plus more
