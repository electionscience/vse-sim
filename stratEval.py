import csv
import os
import multiprocessing
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median

from methods import *
from mwmethods import *
from voterModels import *
from stratFunctions import noisyMedia

def simpleStratTest(voterModel, method, strats, nvot, ncand, numWinners=1,
                    bgStrat=None, bgStratArgs={}, pollingError=0.2,
                    pollingMethod=None, pollingStrat=Approval.zeroInfoBallot, pollingStratArgs={'pickiness':0.7}):
    if pollingMethod is None:
        if numWinners == 1:
            pollingMethod, pollingStrat = Approval, Approval.zeroInfoBallot
        else:
            pollingMethod, pollingStrat = Plurality, Plurality.honBallot
    if bgStrat is None:
        bgStrat = method.honBallot
    for i, strat in enumerate(strats):
        if not isinstance(strat, tuple):
            strats[i] = (strat, {})
    if numWinners > 1:
        stats = {'mean': mean, 'max': max, 'median': median}
    else:
        stats = {'': max}

    electorate = voterModel(nvot, ncand)
    electabilities = noisyMedia(pollingMethod.results(pollingStrat(voter, **pollingStratArgs)
                                                        for voter in electorate), pollingError)
    bgBallots = [bgStrat(v, electabilities=electabilities, numWinners=numWinners, **bgStratArgs) for v in electorate]
    baseWinners = method.winnerSet(bgBallots, numWinners)
    polls = noisyMedia(method.results(bgBallots), pollingError)
    totalResults = []
    for strat, stratArgs in strats + [(method.abstain, {})]:
        results = {stat: 0 for stat in stats}
        for i, voter in enumerate(electorate):
            newBallot = strat(voter, electabilities=electabilities, polls=polls, numWinners=numWinners, **stratArgs)
            winners = method.winnerSet(bgBallots[:i] + [newBallot] + bgBallots[i+1:], numWinners)
            if winners != baseWinners:
                for statName, statFunc in stats.items():
                    results[statName] += statFunc([voter[w] for w in winners]) - statFunc([voter[w] for w in baseWinners])
        totalResults.append(results)
    return totalResults

def st(args, kw):
    return simpleStratTest(*args, **kw)

class StratTest:
    def __init__(self, niter=10, *args, **kw):
        self.rows = []
        with multiprocessing.Pool(processes=7) as pool:
            results = pool.starmap(st, [(args, kw) for i in range(niter)])
            self.rows.extend(results)
        self.totals = [{stat: sum(oneDict[stat] for oneDict in methodResults) for stat in self.rows[0][0]} for methodResults in zip(*self.rows)]
        self.stds = [{stat: std([oneDict[stat] for oneDict in methodResults])*len(methodResults)**0.5 for stat in self.rows[0][0]} for methodResults in zip(*self.rows)]
        self.strats = kw['strats'] if 'strats' in kw else args[2]
        abstainTotals = self.totals[-1]
        self.scores = [{stat: -(total - abstainTotals[stat])/abstainTotals[stat] for stat, total in m.items()} for m in self.totals[:-1]]

    def showScores(self):
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
        print(("Strategy\t" if multiNames else "")+"\t".join(params + [stat for stat in self.scores[0]]))
        for i, row in enumerate(self.scores):
            print((stratNames[i]+'\t' if multiNames else "")
                    +"\t".join([str(stratArgs[i].get(p, "")) for p in params]
                            + [f"{row[stat]:.4f}" for stat in self.scores[0]]))
