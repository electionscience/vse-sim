from ..core import Method
from .irv import Irv


class IrvPrime(Irv):
    """Implement IRV Prime, preserving pairwise challengers during elimination.

    The classic IRV winner and candidates that defeat it pairwise are protected
    until all other candidates have been eliminated. See
    https://electowiki.org/wiki/IRV_Prime.
    """

    stratTargetFor = Method.stratTarget3

    def results(self, ballots, **kwargs):
        """IRV Prime results.

        >>> IrvPrime().results([[0,1,2]])[2]
        2
        >>> IrvPrime().results([[0,1,2],[2,1,0]])[1]
        0
        >>> IrvPrime().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [1, 2, 0]
        >>> IrvPrime().results([[2,1,0]] * 100 + [[1,0,2]] + [[0,2,1]] * 100)
        [1, 0, 2]
        >>> # Favorite betrayal example from http://rangevoting.org/IncentToExagg.html
        >>> IrvPrime().results([[1,2,0]] * 8 + [[2,0,1]] * 6 + [[0,1,2]] * 5)
        [0, 1, 2]
        >>> IrvPrime().results([[0,4,3,1,2]] * 5 + [[1,4,3,2,1]] * 4 + [[2,3,4,0,1]] * 6)
        [4, 2, 3, 0, 1]
        >>> # Elections 3-5 from http://votingmatters.org.uk/ISSUE6/P4.HTM
        >>> IrvPrime().results([[0,1,2,3,4,5]] * 12 + [[2,0,1,3,4,5]] * 11 + [[1,2,0,3,4,5]] * 10 +
        ...     [[3,4,5]] * 27)
        [1, 2, 3, 0, 4, 5]
        >>> IrvPrime().results([[0,1]] * 11 + [[1]] * 7 + [[2]] * 12)
        [1, 2, 0]
        >>> IrvPrime().results([[0,3,2,1]] * 5 + [[1,2,0,3]] * 5 + [[2,0,1,3]] * 8 +
        ...    [[3,0,1,2]] * 4 + [[3,1,2,0]] * 8)
        [0, 3, 2, 1]
        >>> IrvPrime().results([[0,2,1,3]] * 6 + [[0,3,1,2]] * 3 + [[0,3,2,1]] * 3 +
        ...     [[1,2,0,3]] * 4 + [[2,0,1,3]] * 4 + [[3,1,2,0]] * 5)
        [2, 0, 3, 1]
        >>> # Failure of later-no-harm
        >>> IrvPrime().results([[0, 1, 2]] * 32 + [[0, 2, 1]] * 20 + [[1,2,0]] * 30 +
        ...     [[1,0,2]] * 21 + [[2,0,1]] * 30 + [[2,1,0]] * 20)
        [2, 0, 1]
        >>> IrvPrime().results([[0, 1, 2]] * 32 + [[0, 2, 1]] * 20 + [[1,2,0]] * 30 +
        ...     [[1,0,2]] * 21 + [[2,1,0]] * 30 + [[2,1,0]] * 20)
        [1, 0, 2]
        """

        if type(ballots) is not list:
            ballots = list(ballots)

        remaining = self.buildPreferenceSchedule(ballots)
        ncand = len(self.candidateVotes(remaining))
        classic = self.runIrv(remaining, ncand)

        # Keep the winner from the classic IRV
        winners = {classic[0]}

        # Find all candidates that can beat classic IRV winner; this may be a superset
        # of schwartz/smith, but it's all that matters
        winnersPrime = set()
        for possibleWinner in range(ncand):
            if possibleWinner in winners:
                continue

            numWins = 0
            numLosses = 0
            for ranking, votes in remaining.items():
                possibleWinnerRanking = winnerRanking = len(ranking) + 1
                for pos in range(len(ranking)):
                    if ranking[pos] == possibleWinner:
                        possibleWinnerRanking = pos
                    # We can change this to a loop if there's > 1 winner
                    elif ranking[pos] == next(iter(winners)):
                        winnerRanking = pos
                if possibleWinnerRanking < winnerRanking:
                    numWins += votes
                elif winnerRanking < possibleWinnerRanking:
                    numLosses += votes
            if numWins > numLosses:
                winnersPrime.add(possibleWinner)

        # Now re-run IRV preserving all winners + winners prime
        keepers = winners.union(winnersPrime)
        results = [-1] * ncand
        for i in range(ncand):
            votes = self.candidateVotes(remaining)
            toEliminate = self.getLeast(votes, keepers)
            if toEliminate is None:
                # Begin "step 4", i.e. continue elimination without preserving anyone
                keepers = set()
                toEliminate = self.getLeast(votes)
            results[ncand - i - 1] = toEliminate.candidate
            remaining = self.eliminateCandidate(remaining, toEliminate.candidate)

        return results
