from ..core import CandidateWithCount, Method, rememberBallot
from ..voter_models import DeterministicModel, Voter  # noqa: F401


class Irv(Method):
    """Implement Instant-Runoff Voting over complete ranked ballots.

    Ballots are candidate-aligned rank vectors where larger values indicate
    stronger preference. Tabulation repeatedly eliminates the candidate with
    the fewest active first preferences and returns candidate-aligned finish
    scores, again with larger values preferred.
    """

    stratTargetFor = Method.stratTarget3

    def buildPreferenceSchedule(self, ballots):
        """Gets a dictionary of the form {ranking as tuple, vote count}."""

        prefs = {}
        for b in ballots:
            key = tuple(b)
            if key in prefs:
                prefs[key] += 1
            else:
                prefs[key] = 1
        return prefs

    def eliminateCandidate(self, inputPrefs, toEliminate):
        """Gets a dictionary of the form {ranking as tuple, vote count} with toEliminate removed."""

        if not isinstance(toEliminate, CandidateWithCount):
            return inputPrefs

        prefs = {}
        for ranking, votes in inputPrefs.items():
            newranking = [
                candidate
                for candidate in ranking
                if candidate != toEliminate.candidate
            ]

            if not newranking:
                continue
            newkey = tuple(newranking)
            if newkey in prefs:
                prefs[newkey] += votes
            else:
                prefs[newkey] = votes
        return prefs

    def candidateVotes(self, prefSchedule):
        """Gets a list of CandidateWithCount, from highest to lowest."""
        candidates = {}
        for ranking, votes in prefSchedule.items():
            candidate = ranking[0]
            if candidate in candidates:
                candidates[candidate].votes += votes
            else:
                candidates[candidate] = CandidateWithCount(candidate, votes)

        # Simply for VSE which requires ranking of non-winners; in real election we don't really
        # care
        alternates = []
        trackedalt = set()
        for ranking, _votes in prefSchedule.items():
            for alternate in ranking[1:]:
                if (alternate not in candidates) and alternate not in trackedalt:
                    alternates.append(CandidateWithCount(alternate, 0))
                    trackedalt.add(alternate)

        return sorted(candidates.values(), key=lambda c: (c.votes, c.candidate), reverse = True) + alternates

    def getLeast(self, voteRanking, keep = {}):
        for candidate in reversed(voteRanking):
            if candidate.candidate not in keep:
                return candidate

    def runIrv(self, remaining, ncand):
        """IRV results."""
        results = [-1] * ncand
        for i in range(ncand):
            votes = self.candidateVotes(remaining)
            toEliminate = self.getLeast(votes)
            results[ncand - i - 1] = toEliminate.candidate
            remaining = self.eliminateCandidate(remaining, toEliminate)
        return results

    @staticmethod
    def rankVectorToPreference(ballot):
        """Return candidate IDs in descending preference order from a rank vector."""
        return sorted(range(len(ballot)), key=lambda candidate: ballot[candidate],
                      reverse=True)

    @staticmethod
    def finishOrderToResults(finishOrder):
        """Convert winner-first finish order to high-is-better candidate scores."""
        ncand = len(finishOrder)
        results = [-1] * ncand
        for score, candidate in enumerate(reversed(finishOrder)):
            results[candidate] = score
        return results

    def results(self, ballots, **kwargs):
        """IRV results.

        >>> Irv().resultsFor(DeterministicModel(3)(5,3),Irv().honBallot)["results"]
        [0, 1, 2]
        >>> Irv().results([[0,1,2]])[2]
        2
        >>> Irv().results([[0,1,2],[2,1,0]])[1]
        0
        >>> Irv().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [2, 0, 1]
        """
        if type(ballots) is not list:
            ballots = list(ballots)
        rankings = [self.rankVectorToPreference(ballot) for ballot in ballots]
        finishOrder = self.runIrv(self.buildPreferenceSchedule(rankings), len(ballots[0]))
        return self.finishOrderToResults(finishOrder)

    @staticmethod #cls is provided explicitly, not through binding
    @rememberBallot
    def honBallot(cls, voter):
        """Takes utilities and returns an honest ballot.

        >>> Irv.honBallot(Irv,Voter([4,1,6,3]))
        [2, 0, 3, 1]
        """
        ballot = [-1] * len(voter)
        order = sorted(enumerate(voter), key=lambda x:x[1])
        for i, cand in enumerate(order):
            ballot[cand[0]] = i
        return ballot


    @classmethod
    def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                        frontId, frontResult, targId, targResult):
        """
        >>> Irv().stratBallotFor([3,2,1,0])(Irv,Voter([3,6,5,2]))
        [1, 2, 3, 0]
        """
        i = n - 1
        winnerQ = voter[frontId]
        targQ = voter[targId]
        placesToFill = list(range(n-1,0,-1))
        if targQ > winnerQ:
            ballot[targId] = i
            i -= 1
            del placesToFill[-2]
        for j in placesToFill:
            nextLoser, loserScore = places[j] #all but winner, low to high
            if voter[nextLoser] > winnerQ:
                ballot[nextLoser] = i
                i -= 1
        ballot[frontId] = i
        i -= 1
        for j in placesToFill:
            nextLoser, loserScore = places[j]
            if voter[nextLoser] <= winnerQ:
                ballot[nextLoser] = i
                i -= 1
        assert i == -1
