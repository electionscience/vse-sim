import csv
import os
import multiprocessing
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median

from methods import *
from mwmethods import *
from voterModels import *
from stratFunctions import noisyMedia


def simpleStratTest(
    electorate, method, strats, numWinners, electabilities, bgBallots, pollingError=0.2
):

    if numWinners > 1:
        stats = {"mean": mean, "max": max, "median": median}
    else:
        stats = {"": max}

    baseWinners = method.winnerSet(bgBallots, numWinners)
    polls = noisyMedia(method.results(bgBallots), pollingError)
    totalResults = []
    for strat, stratArgs in strats:
        results = {stat: 0 for stat in stats}
        for i, voter in enumerate(electorate):
            newBallot = strat(
                voter,
                electabilities=electabilities,
                polls=polls,
                numWinners=numWinners,
                **stratArgs,
            )
            winners = method.winnerSet(
                bgBallots[:i] + [newBallot] + bgBallots[i + 1 :], numWinners
            )
            if winners != baseWinners:
                for statName, statFunc in stats.items():
                    results[statName] += statFunc(
                        [voter[w] for w in winners]
                    ) - statFunc([voter[w] for w in baseWinners])
        totalResults.append(results)
    return totalResults


def st(args, kw):
    return simpleStratTest(*args, **kw)


def methodStratLoop(
    voterModel,
    methods,
    strats,
    nvot,
    ncand,
    numWinners,
    bgStrat=None,
    bgStratArgs=None,
    pollingError=0.2,
    pollingMethod=None,
    pollingStrat=Approval.zeroInfoBallot,
    pollingStratArgs=None,
):
    if bgStratArgs is None:
        bgStratArgs = {}
    if pollingStratArgs is None:
        pollingStratArgs = {"pickiness": 0.7}
    if not isinstance(methods, list):
        methods = [methods]
    electorate = voterModel(nvot, ncand)
    electabilities = noisyMedia(
        pollingMethod.results(
            pollingStrat(voter, **pollingStratArgs) for voter in electorate
        ),
        pollingError,
    )
    bgBallots = [
        bgStrat(v, electabilities=electabilities, numWinners=numWinners, **bgStratArgs)
        for v in electorate
    ]
    return {
        m.__name__: simpleStratTest(
            electorate,
            m,
            strats,
            numWinners,
            electabilities,
            bgBallots,
            pollingError,
        )
        for m in methods
    }


def keyTuple(method, strat, statName):
    return (method.__name__, strat[0].__name__, str(strat[1]), statName)


class StratTest:
    def __init__(
        self,
        niter,
        voterModel,
        methods,
        strats,
        nvot,
        ncand,
        numWinners=1,
        bgStrat=None,
        bgStratArgs=None,
        pollingError=0.2,
        pollingMethod=None,
        pollingStrat=Approval.zeroInfoBallot,
        pollingStratArgs=None,
    ):
        if bgStratArgs is None:
            bgStratArgs = {}
        if pollingStratArgs is None:
            pollingStratArgs = {"pickiness": 0.7}
        if pollingMethod is None:
            if numWinners == 1:
                pollingMethod, pollingStrat = Approval, Approval.zeroInfoBallot
            else:
                pollingMethod, pollingStrat = Plurality, Plurality.honBallot
        if not isinstance(methods, list):
            methods = [methods]
        if bgStrat is None:
            bgStrat = methods[0].honBallot
        for i, strat in enumerate(strats):
            if not isinstance(strat, tuple):
                strats[i] = (strat, {})
        self.strats = strats
        strats.append((methods[0].abstain, {}))
        self.statNames = [""] if numWinners == 1 else ["mean", "max", "median"]
        self.rowDicts = {
            keyTuple(m, strat, stat): []
            for m in methods
            for strat in strats
            for stat in self.statNames
        }
        with multiprocessing.Pool(processes=7) as pool:
            results = pool.starmap(
                methodStratLoop,
                [
                    (
                        voterModel,
                        methods,
                        strats,
                        nvot,
                        ncand,
                        numWinners,
                        bgStrat,
                        bgStratArgs,
                        pollingError,
                        pollingMethod,
                        pollingStrat,
                        pollingStratArgs,
                    )
                    for i in range(niter)
                ],
            )

        for row in results:
            for m in methods:
                for i, strat in enumerate(strats):
                    for stat in self.statNames:
                        self.rowDicts[keyTuple(m, strat, stat)].append(
                            row[m.__name__][i][stat]
                        )
        self.totals = {key: sum(rows) for key, rows in self.rowDicts.items()}
        # self.totals = [{stat: sum(oneDict[stat] for oneDict in methodResults) for stat in self.rows[0][0]} for methodResults in zip(*self.rows)]
        # self.stds = [{stat: std([oneDict[stat] for oneDict in methodResults])*len(methodResults)**0.5 for stat in self.rows[0][0]} for methodResults in zip(*self.rows)]
        # self.strats = kw['strats'] if 'strats' in kw else args[2]
        # abstainTotals = self.totals[-1]
        self.statNames = [""] if numWinners == 1 else ["mean", "max", "median"]
        self.esif = {
            keyTuple(m, strat, stat): -(
                self.totals[keyTuple(m, strat, stat)]
                - self.totals[keyTuple(m, (m.abstain, {}), stat)]
            )
            / self.totals[keyTuple(m, (m.abstain, {}), stat)]
            for m in methods
            for strat in self.strats
            for stat in self.statNames
        }
        # self.scores = [{stat: -(total - abstainTotals[stat])/abstainTotals[stat] for stat, total in m.items()} for m in self.totals[:-1]]
        self.multiMethods = len(methods) > 1
        self.multiStrats = any(s != strats[0][0] for s, a in strats)
        self.multiArgs = any(a != strats[0][1] for s, a in strats)

    def showESIF(self):
        for stat in self.statNames:
            print(f"{stat}:")
            print(
                ("Method\t" if self.multiMethods else "")
                + ("Strategy\t" if self.multiStrats else "")
                + ("Arguments\t" if self.multiArgs else "")
                + "ESIF"
            )
            for key, value in self.esif.items():
                if key[3] != stat or key[1] == "abstain":
                    continue
                print(
                    (key[0] + "\t" if self.multiMethods else "")
                    + (key[1] + "\t" if self.multiStrats else "")
                    + (key[2] + "\t" if self.multiArgs else "")
                    + f"{value:.4f}"
                )

    def getUncertainty(self, m1, m2, s1, s2, stat=None):
        if not isinstance(s1, tuple):
            s1 = (s1, {})
        if not isinstance(s2, tuple):
            s2 = (s2, {})
        if stat is None:
            stat = self.statNames[0]
        niter = len(self.rowDicts[keyTuple(m1, s1, stat)])
        devOfTotal = (
            std(
                [
                    a - b
                    for a, b in zip(
                        self.rowDicts[keyTuple(m1, s1, stat)],
                        self.rowDicts[keyTuple(m2, s2, stat)],
                    )
                ]
            )
            * niter**0.5
        )
        total = (
            self.totals[keyTuple(m1, s1, stat)] - self.totals[keyTuple(m2, s2, stat)]
        )
        return devOfTotal, total

    def showScores(self):
        """nonfunctional"""
        params = []
        stratNames = []
        stratArgs = []
        for s in self.strats:
            if isinstance(s, tuple):
                stratNames.append(s[0].__name__)
                stratArgs.append(s[1])
                for param in s[1]:
                    if param not in params:
                        params.append(param)
            else:
                stratNames.append(s.__name__)
                stratArgs.append({})
        multiNames = any(name != stratNames[0] for name in stratNames)
        print(("Strategy\t" if multiNames else "") + "\t".join(params + statNames))
        for key, value in self.esif:
            print(
                (key[1] + "\t" if multiNames else "")
                + "\t".join(
                    [str(stratArgs[i].get(p, "")) for p in params]
                    + [f"{row[stat]:.4f}" for stat in self.scores[0]]
                )
            )
