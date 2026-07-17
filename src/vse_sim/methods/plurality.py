from ..core import rememberBallot
from ..voter_models import Voter  # noqa: F401
from .borda import RankedMethod


class Plurality(RankedMethod):

    nRanks = 2

    @staticmethod
    def oneVote(utils, forWhom):
        ballot = [0] * len(utils)
        ballot[forWhom] = 1
        return ballot

    @staticmethod #cls is provided explicitly, not through binding
    @rememberBallot
    def honBallot(cls, utils):
        """Takes utilities and returns an honest ballot.

        >>> Plurality.honBallot(Plurality, Voter([-3,-2,-1]))
        [0, 0, 1]
        >>> Plurality().stratBallotFor([3,2,1])(Plurality, Voter([-3,-2,-1]))
        [0, 1, 0]
        """
        ballot = [0] * len(utils)
        cls.fillPrefOrder(utils, ballot,
            nSlots = 1, lowSlot=1, remainderScore=0)
        return ballot
