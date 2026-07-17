import random

from ..core import rememberBallot
from ..voter_models import DeterministicModel, Voter  # noqa: F401
from .score import Score


def BulletyApprovalWith(bullets=0.5, asClass=False):



    class BulletyApproval((Score(1,True))):

        bulletiness = bullets

        def __str__(self):
            return f"BulletyApproval{str(round(self.bulletiness * 100))}"



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
            if random.random() > cls.bulletiness:
                return cls.__bases__[0].honBallot(cls, utils)
            best = max(utils)
            return [1 if util==best else 0 for util in utils]


    return BulletyApproval if asClass else BulletyApproval()
