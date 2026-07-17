from ..core import CandidateWithCount, Method, rememberBallot
from ..voter_models import DeterministicModel, Voter  # noqa: F401


def build_preference_schedule(ballots):
    """Count identical candidate rankings."""
    preferences = {}
    for ballot in ballots:
        ranking = tuple(ballot)
        preferences[ranking] = preferences.get(ranking, 0) + 1
    return preferences


def eliminate_candidate(preferences, candidate_to_eliminate):
    """Return a schedule with one candidate removed from every ranking."""
    if not isinstance(candidate_to_eliminate, CandidateWithCount):
        return preferences

    updated_preferences = {}
    for ranking, votes in preferences.items():
        updated_ranking = tuple(
            candidate
            for candidate in ranking
            if candidate != candidate_to_eliminate.candidate
        )
        if updated_ranking:
            updated_preferences[updated_ranking] = (
                updated_preferences.get(updated_ranking, 0) + votes
            )
    return updated_preferences


def candidate_votes(preference_schedule):
    """Return active candidates ordered from most to fewest first choices."""
    candidates = {}
    for ranking, votes in preference_schedule.items():
        candidate = ranking[0]
        if candidate in candidates:
            candidates[candidate].votes += votes
        else:
            candidates[candidate] = CandidateWithCount(candidate, votes)

    # VSE needs a complete ranking even for candidates with no active first
    # choices.
    alternates = []
    tracked_alternates = set()
    for ranking in preference_schedule:
        for alternate in ranking[1:]:
            if alternate not in candidates and alternate not in tracked_alternates:
                alternates.append(CandidateWithCount(alternate, 0))
                tracked_alternates.add(alternate)

    active = sorted(
        candidates.values(),
        key=lambda candidate: (candidate.votes, candidate.candidate),
        reverse=True,
    )
    return active + alternates


def least_candidate(vote_ranking, keep=None):
    """Return the lowest-ranked candidate not present in ``keep``."""
    keep = () if keep is None else keep
    for candidate in reversed(vote_ranking):
        if candidate.candidate not in keep:
            return candidate
    return None


def rank_vector_to_preference(ballot):
    """Return candidate IDs in descending preference order from a rank vector."""
    return sorted(
        range(len(ballot)),
        key=lambda candidate: ballot[candidate],
        reverse=True,
    )


def finish_order_to_results(finish_order):
    """Convert winner-first finish order to high-is-better candidate scores."""
    results = [-1] * len(finish_order)
    for score, candidate in enumerate(reversed(finish_order)):
        results[candidate] = score
    return results


class Irv(Method):
    """Implement Instant-Runoff Voting over complete ranked ballots.

    Ballots are candidate-aligned rank vectors where larger values indicate
    stronger preference. Tabulation repeatedly eliminates the candidate with
    the fewest active first preferences and returns candidate-aligned finish
    scores, again with larger values preferred.
    """

    stratTargetFor = Method.stratTarget3

    buildPreferenceSchedule = staticmethod(build_preference_schedule)
    eliminateCandidate = staticmethod(eliminate_candidate)
    candidateVotes = staticmethod(candidate_votes)
    getLeast = staticmethod(least_candidate)
    rankVectorToPreference = staticmethod(rank_vector_to_preference)
    finishOrderToResults = staticmethod(finish_order_to_results)

    def runIrv(self, remaining, ncand):
        """IRV results."""
        results = [-1] * ncand
        for i in range(ncand):
            votes = self.candidateVotes(remaining)
            toEliminate = self.getLeast(votes)
            results[ncand - i - 1] = toEliminate.candidate
            remaining = self.eliminateCandidate(remaining, toEliminate)
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
