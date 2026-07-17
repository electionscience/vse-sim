from numpy import floor, mean

from ..core import Method, rememberBallot
from ..voter_models import DeterministicModel, Voter  # noqa: F401


def Score(topRank=10, asClass=False):

    class Score0to(Method):
        """Score voting, 0-10.


        Strategy establishes pivots
            >>> Score().stratBallotFor([0,1,2])(Score, Voter([5,6,7]))
            [0, 0, 10]
            >>> Score().stratBallotFor([2,1,0])(Score, Voter([5,6,7]))
            [0, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([5,6,7]))
            [0, 5.0, 10]

        Strategy (kinda) works for ties
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([5,6,6]))
            [0, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,6,7]))
            [0, 0, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,7,6]))
            [10, 10, 10]
            >>> Score().stratBallotFor([1,0,2])(Score, Voter([6,5,6]))
            [10, 0, 10]

        """

        bias2 = 2.770135393419682
        bias5 = 2.3536762480634343
        candScore = staticmethod(mean)


        def __str__(self):
            if self.topRank == 1:
                return "IdealApproval"
            return self.__class__.__name__ + str(self.topRank)

        @staticmethod #cls is provided explicitly, not through binding
        @rememberBallot
        def honBallot(cls, utils):
            """Takes utilities and returns an honest ballot (on 0..10).


            honest ballots work as expected
                >>> Score().honBallot(Score, Voter([5,6,7]))
                [0.0, 5.0, 10.0]
                >>> Score().resultsFor(DeterministicModel(3)(5,3),Score().honBallot)["results"]
                [4.0, 6.0, 5.0]
            """
            bot = min(utils)
            scale = max(utils)-bot
            if scale == 0:
                return [cls.topRank] * len(utils)
            return [floor((cls.topRank + .99) * (util-bot) / scale) for util in utils]


        @classmethod
        def fillStratBallot(cls, voter, polls, places, n, stratGap, ballot,
                            frontId, frontResult, targId, targResult):
            """Returns a (function which takes utilities and returns a strategic ballot)
            for the given "polling" info."""

            cuts = [voter[frontId], voter[targId]]
            if stratGap > 0:
                #sort cuts high to low
                cuts = (cuts[1], cuts[0])
            if cuts[0] == cuts[1]:
                strat = [(cls.topRank if (util >= cuts[0]) else 0) for util in voter]
            else:
                strat = [max(0,min(cls.topRank,floor(
                                (cls.topRank + .99) * (util-cuts[1]) / (cuts[0]-cuts[1])
                            )))
                        for util in voter]
            for i in range(n):
                ballot[i] = strat[i]

    Score0to.topRank = topRank
    return Score0to if asClass else Score0to()
