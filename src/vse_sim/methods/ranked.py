from ..core import Method, rememberBallot


class RankedMethod(Method):
    """Base class for methods that use candidate-aligned rank vectors.

    Larger ballot values represent stronger preferences. The class provides
    shared helpers for constructing complete or truncated ranked ballots and
    the default ranked-method strategy used by Borda and plurality.
    """

    @staticmethod
    def fillPrefOrder(
        voter,
        ballot,
        whichCands=None,
        lowSlot=0,
        nSlots=None,
        remainderScore=None,
    ):
        """Fill ``ballot`` with candidates ordered by decreasing utility."""
        venum = list(enumerate(voter))
        if whichCands:
            venum = [venum[c] for c in whichCands]
        prefOrder = sorted(venum, key=lambda x: -x[1])
        RankedMethod.fillCands(
            ballot, prefOrder, lowSlot, nSlots, remainderScore
        )

    @staticmethod
    def fillCands(
        ballot,
        whichCands,
        lowSlot=0,
        nSlots=None,
        remainderScore=None,
    ):
        """Assign descending ranks to candidate tuples in ``whichCands``."""
        if nSlots is None:
            nSlots = len(whichCands)
        cur = lowSlot + nSlots - 1
        for i in range(nSlots):
            ballot[whichCands[i][0]] = cur
            cur -= 1
        if remainderScore is not None:
            for candidate, *_ in whichCands[nSlots:]:
                ballot[candidate] = remainderScore

    @staticmethod
    @rememberBallot
    def honBallot(cls, utils):
        """Return a complete rank vector ordered by utility."""
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot)
        return ballot

    @classmethod
    def fillStratBallot(
        cls,
        voter,
        polls,
        places,
        n,
        stratGap,
        ballot,
        frontId,
        frontResult,
        targId,
        targResult,
    ):
        """Mutate ``ballot`` with the default strategy for ranked methods."""
        nRanks = min(cls.nRanks, n)
        if stratGap <= 0:
            ballot[frontId], ballot[targId] = (nRanks - 1), 0
        else:
            ballot[frontId], ballot[targId] = 0, (nRanks - 1)
        nRanks -= 2
        if nRanks > 0:
            cls.fillCands(
                ballot,
                places[2:][::-1],
                lowSlot=1,
                nSlots=nRanks,
                remainderScore=0,
            )


RatedMethod = RankedMethod
