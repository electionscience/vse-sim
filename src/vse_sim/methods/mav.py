from numpy import floor, percentile

from ..core import Method, rememberBallot, rememberBallots
from ..voter_models import Voter  # noqa: F401


def toVote(cutoffs, util):
    """Maps one util to a vote, using cutoffs.

    Used by Mav, but declared outside to avoid method binding overhead."""
    for vote in range(len(cutoffs)):
        if util <= cutoffs[vote]:
            return vote
    return vote + 1


class Mav(Method):
    """Majority Approval Voting.
    """

    bias5 = 1.0970202515275356


    baseCuts = [-0.8, 0, 0.8, 1.6]
    specificPercentiles = [25,50,75,90]

    def candScore(self, scores):
        """For now, only works correctly for odd nvot.

        Basic tests
            >>> Mav().candScore([1,2,3,4,5])
            3.0
            >>> Mav().candScore([1,2,3,3,3])
            2.5
            >>> Mav().candScore([1,2,3,4])
            2.5
            >>> Mav().candScore([1,2,3,3])
            2.5
            >>> Mav().candScore([1,2,2,2])
            1.5
            >>> Mav().candScore([1,2,3,3,5])
            2.7
            """
        scores = sorted(scores)
        nvot = len(scores)
        nGrades = (len(self.baseCuts) + 1)
        i = int((nvot - 1) / 2)
        base = scores[i]
        while i < nvot and scores[i] == base:
            i += 1
        upper =  (base + 0.5) - (i - nvot/2) * nGrades / nvot
        lower = (base) - (i - nvot/2) / nvot
        return max(upper, lower)

    def honBallotFor(self, voters):
        """Return an honest ballot function with election-scoped cutoffs."""
        cuts = percentile(voters, self.specificPercentiles)

        def honBallot(cls, voter, tally=None):
            ballot = cls._honBallotWithCuts(voter, cuts)
            setattr(voter, f"{cls.__name__}_hon", ballot)
            return ballot

        honBallot.__name__ = "honBallot"
        honBallot.allTallyKeys = lambda: []
        return honBallot

    @staticmethod
    def _honBallotWithCuts(voter, cuts):
        cuts = [min(cut, max(voter) - 0.001) for cut in cuts]
        return [toVote(cuts, util) for util in voter]

    @staticmethod #cls is provided explicitly, not through binding
    @rememberBallot
    def honBallot(cls, voter):
        """Takes utilities and returns an honest ballot (on 0..4).

        honest ballot works as intended, gives highest grade to highest utility:
            >>> Mav().honBallot(Mav, Voter([-1,-0.5,0.5,1,1.1]))
            [0, 1, 2, 3, 4]

        Even if they don't rate at least an honest "B":
            >>> Mav().honBallot(Mav, Voter([-1,-0.5,0.5]))
            [0, 1, 4]
        """
        return cls._honBallotWithCuts(voter, cls.baseCuts)


    def stratBallotFor(self, polls):
        """Returns a function which takes utilities and returns a dict(
            strat=<ballot in which all grades are exaggerated
                             to outside the range of the two honest frontrunners>,
            extraStrat=<ballot in which all grades are exaggerated to extremes>,
            isStrat=<whether the runner-up is preferred to the frontrunner (for reluctantStrat)>,
            stratGap=<utility of runner-up minus that of frontrunner>
            )
        for the given "polling" info.



        Strategic tests:
            >>> Mav().stratBallotFor([0,1.1,1.9,0,0])(Mav, Voter([-1,-0.5,0.5,1,2]))
            [0, 1, 2, 3, 4]
            >>> Mav().stratBallotFor([0,2.1,2.9,0,0])(Mav, Voter([-1,-0.5,0.5,1,2]))
            [0, 1, 3, 3, 4]
            >>> Mav().stratBallotFor([0,2.1,1.9,0,0])(Mav, Voter([-1,0.4,0.5,1,2]))
            [0, 1, 3, 3, 4]
            >>> Mav().stratBallotFor([1,0,2])(Mav, Voter([6,7,6]))
            [4, 4, 4]
            >>> Mav().stratBallotFor([1,0,2])(Mav, Voter([6,5,6]))
            [4, 0, 4]
            >>> Mav().stratBallotFor([2.1,0,3])(Mav, Voter([6,5,6]))
            [4, 0, 4]
            >>> Mav().stratBallotFor([2.1,0,3])(Mav, Voter([6,5,6.1]))
            [2, 2, 4]
        """
        places = sorted(enumerate(polls),key=lambda x:-x[1]) #from high to low
        ((frontId,frontResult), (targId, targResult)) = places[:2]

        @rememberBallots
        def stratBallot(cls, voter):
            frontUtils = [voter[frontId], voter[targId]] #utils of frontrunners
            stratGap = frontUtils[1] - frontUtils[0]
            if stratGap == 0:
                strat = extraStrat = [(4 if (util >= frontUtils[0]) else 0)
                                     for util in voter]
                isStrat = True

            else:
                if stratGap < 0:
                    #winner is preferred; be complacent.
                    isStrat = False
                else:
                    #runner-up is preferred; be strategic in iss run
                    isStrat = True
                    #sort cuts high to low
                    frontUtils = (frontUtils[1], frontUtils[0])
                top = max(voter)
                cutoffs = [(  (min(frontUtils[0], self.baseCuts[i]))
                                 if (i < floor(targResult)) else
                            ( (frontUtils[1])
                                 if (i < floor(frontResult) + 1) else
                              min(top, self.baseCuts[i])
                              ))
                           for i in range(len(self.baseCuts))]
                strat = [toVote(cutoffs, util) for util in voter]
                extraStrat = [max(0,min(10,floor(
                                4.99 * (util-frontUtils[1]) / (frontUtils[0]-frontUtils[1])
                            )))
                        for util in voter]
            return dict(strat=strat, extraStrat=extraStrat, isStrat=isStrat,
                        stratGap = stratGap)

        return stratBallot
