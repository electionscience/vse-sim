from numpy import mean

from .ranked import RankedMethod


class Borda(RankedMethod):
    """Implement Borda count with larger rank values representing preference.

    Honest ballots assign consecutive scores from least to most preferred.
    """

    candScore = staticmethod(mean)

    nRanks = 999 # infinity
