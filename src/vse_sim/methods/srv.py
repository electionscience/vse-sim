from numpy import sign

from ..core import Method
from ..voter_models import DeterministicModel  # noqa: F401
from .irv import Irv  # noqa: F401
from .score import Score


def Srv(topRank=10):
    """Score Runoff Voting
        >>> Srv().resultsFor(DeterministicModel(3)(5,3),Irv().honBallot)["results"]
        [0.8, 1.2, 1.21]
        >>> Srv().results([[0,1,2]])[2]
        2.0
        >>> Srv().results([[0,1,2],[2,1,0]])[1]
        1.0
        >>> Srv().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [0.8888888888888888, 1.2222222222222223, 0.8888888888888888]
        >>> Srv().results([[2,1,0]] * 100 + [[1,0,2]] + [[0,2,1]] * 100)
        [1.502537313432836, 1.492537313432836, 0.5074626865671642]
        >>> Srv().results([[1,2,0]] * 8 + [[2,0,1]] * 6 + [[0,1,2]] * 5)
        [1.0526315789473684, 1.105263157894737, 0.8421052631578947]
        >>> Srv().results([[0,4,3,1,2]] * 5 + [[1,4,3,2,1]] * 4 + [[2,3,4,0,1]] * 6)
        [1.0666666666666667, 3.6, 3.4, 0.8666666666666667, 1.3333333333333333]
    """

    score0to = Score(topRank,True)

    class Srv0to(score0to):

        stratTargetFor = Method.stratTarget3

        def results(self, ballots, **kwargs):
            """Srv results."""
            baseResults = super(Srv0to, self).results(ballots, **kwargs)
            (runnerUp,top) = sorted(range(len(baseResults)), key=lambda i: baseResults[i])[-2:]
            upset = sum(sign(ballot[runnerUp] - ballot[top]) for ballot in ballots)
            if upset > 0:
                baseResults[runnerUp] = baseResults[top] + 0.01
            return [result.item() if hasattr(result, "item") else result for result in baseResults]
    return Srv0to()
