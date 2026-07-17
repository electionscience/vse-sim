from numpy import argsort, sign

from ..core import Method, rememberBallots
from ..voter_models import DeterministicModel, Voter  # noqa: F401
from .irv import Irv  # noqa: F401
from .mav import Mav


class V321(Mav):
    baseCuts = [-.1,.8]
    specificPercentiles = [45, 75]

    stratTargetFor = Method.stratTarget3

    def results(self, ballots, isHonest=False, **kwargs):
        """3-2-1 Voting results.

        >>> V321().resultsFor(DeterministicModel(3)(5,3),V321().honBallot)["results"]
        [-0.75, 2, 1]
        >>> V321().results([[0,1,2]])[2]
        2
        >>> V321().results([[0,1,2],[2,1,0]])[1]
        2.5
        >>> V321().results([[0,1,2]] * 4 + [[2,1,0]] * 3 + [[1,2,0]] * 2)
        [1, 1.5, -0.25]
        >>> V321().results([[0,1,2,1]]*29 + [[1,2,0,1]]*30 + [[2,0,1,1]]*31 + [[1,1,1,2]]*10)
        [3, 0.5, 1, 0]
        >>> V321().results([[1,0,2,1]]*29 + [[0,2,1,1]]*30 + [[2,1,0,1]]*31 + [[1,1,1,2]]*10)
        [3.375, 2.875, 0.25, 0]
        """
        candScores = list(zip(*ballots, strict=False))
        n2s = [sum(1 if s>1 else 0 for s in c) for c in candScores]
        o2s = argsort(n2s) #order
        r2s = [-1] * len(n2s) #ranks
        for r,i in enumerate(o2s):
            r2s[i] = r
        semifinalists = o2s[-3:] #[third, second, first] by top ranks
        n1s = [sum(1 if s>0 else 0 for s in candScores[sf]) for sf in semifinalists]
        o1s = argsort(n1s)
        r2s[semifinalists[o1s[0]]] -= (o1s[0] +1) * .75 #non-finalist below finalists
        (runnerUp,top) = semifinalists[o1s[1]], semifinalists[o1s[2]]
        upset = sum(sign(ballot[runnerUp] - ballot[top]) for ballot in ballots)
        if upset > 0:
            runnerUp, top = top, runnerUp
            r2s[runnerUp], r2s[top] = r2s[top] - .125, r2s[runnerUp] + .125
        r2s[top] = max(r2s[top], r2s[runnerUp] + 0.5)
        if isHonest:
            self.extraEvents.update({"3beats1": False, "3beats2": False, "4beats1": False})
            upset2 =  sum(sign(ballot[semifinalists[o1s[0]]] - ballot[semifinalists[o1s[2]]]) for ballot in ballots)
            self.extraEvents["3beats1"] = upset2 > 0
            upset3 =  sum(sign(ballot[semifinalists[o1s[0]]] - ballot[semifinalists[o1s[1]]]) for ballot in ballots)
            self.extraEvents["3beats2"] = upset3 > 0
            if len(o2s) > 3:
                fourth = o2s[-4]
                fourthNotLasts = sum(1 if s>1 else 0 for s in candScores[fourth])
                fourthWin = (fourthNotLasts > n1s[o1s[1]] and
                             sum(sign(ballot[fourth] - ballot[semifinalists[o1s[2]]])
                                    for ballot in ballots)
                                > 0)
                self.extraEvents["4beats1"] = fourthWin

        return [result.item() if hasattr(result, "item") else result for result in r2s]

    def stratBallotFor(self, polls):
        """Returns a function which takes utilities and returns a dict(
            isStrat=
        for the given "polling" info.


        >>> Irv().stratBallotFor([3,2,1,0])(Irv,Voter([3,6,5,2]))
        [1, 2, 3, 0]
        """
        len(polls)

        places = sorted(enumerate(polls),key=lambda x:-x[1]) #high to low
        top3 = [c for c,r in places[:3]]

        def stratBallot(cls, voter):
            stratGap = voter[top3[1]] - voter[top3[0]]
            myPrefs = [c for c,v in sorted(enumerate(voter),key=lambda x:-x[1])] #high to low
            my3order = [myPrefs.index(c) for c in top3]
            rating = 2
            ballot = [0] * len(voter)
            if my3order[0] == min(my3order): #agree on winner
                for i in range(my3order[0]+1):
                    ballot[myPrefs[i]] = 2
                if my3order[1] <= my3order[2]:
                    for i in range(my3order[0]+1,my3order[1]+1):
                        ballot[myPrefs[i]] = 1
                return dict(strat=ballot, isStrat=False, stratGap=stratGap)
            for c in myPrefs:
                ballot[c] = rating
                if rating and (c in top3):
                    if c == top3[0]:
                        rating = 0
                    else:
                        rating -= 1

            return dict(strat=ballot, isStrat=True, stratGap=stratGap)
        if self.extraEvents["3beats1"]:
            @rememberBallots
            def stratBallo2(cls, voter):
                stratGap = voter[top3[1]] - voter[top3[0]]
                myprefs = sorted(enumerate(voter),key=lambda x:-x[1]) #high to low
                rating = 2
                ballot = [None] * len(voter)
                isStrat=False
                stratGap = 0
                for c, _util in myprefs:
                    ballot[c] = rating
                    if rating and (c in top3):
                        if (c == top3[2]):
                            isStrat= (rating == 2)
                            rating = 0
                        else:
                            rating -= 1
                isStrat = (voter[top3[0]] == max(voter[c] for c in top3))
                return dict(strat=ballot, isStrat=isStrat, stratGap=stratGap)
            stratBallo2.__name__ = "stratBallot" #God, that's ugly.
            return stratBallo2

        if self.extraEvents["4beats1"]:
            fourth = places[3][1]
            first = top3[1]
            @rememberBallots
            def stratBallo3(cls, voter):
                stratGap = voter[top3[1]] - voter[top3[0]]
                myprefs = sorted(enumerate(voter),key=lambda x:-x[1]) #high to low

                rating = 2
                ballot = [None] * len(voter)
                if voter[fourth] > voter[first]:

                    for c, _util in myprefs:
                        ballot[c] = rating
                        if rating and (c == fourth):
                            rating -= 2
                        return dict(strat=ballot, isStrat=True, stratGap=stratGap)

                return stratBallot(cls,voter)
            stratBallo3.__name__ = "stratBallot" #God, that's ugly.
            return stratBallo3


        return rememberBallots(stratBallot)
