from methods import *
from voterModels import *


def pollPert(
    model,
    methodsAndStrats,
    nvot,
    ncand,
    niter,
    nwinners=1,
    perturbations=None,
    pollingMethod=Approval,
    pollingStrat=Approval.zeroInfoBallot,
    pollingStratArgs=None,
):
    if perturbations is None:
        perturbations = [-0.2, -0.02, 0.02, 0.2]
    if pollingStratArgs is None:
        pollingStratArgs = {"pickiness": 0.7}
    ms = []
    for m in methodsAndStrats:
        if isinstance(m, type):
            ms.append((m, m.vaBallot, {}))
        elif len(m) == 2:
            ms.append((m[0], m[1], {}))
        else:
            ms.append(m)
    totalResults = {p: {m[0].__name__: [0, 0, 0] for m in ms} for p in perturbations}
    condResults = {p: {m[0].__name__: [0, 0, 0] for m in ms} for p in perturbations}
    nonCondResults = {p: {m[0].__name__: [0, 0, 0] for m in ms} for p in perturbations}
    for _ in range(niter):
        electorate = model(nvot, ncand)
        basePolls = pollingMethod.results([pollingStrat(v) for v in electorate])
        condWinner = (
            None
            if Condorcet.scenarioType(electorate) == "cycle"
            else Condorcet.winnerSet([Condorcet.honBallot(v) for v in electorate])[0]
        )
        for method, strat, stratArgs in ms:
            baseWinners = set(
                method.winnerSet(
                    [
                        strat(v, polls=basePolls, electabilities=basePolls, **stratArgs)
                        for v in electorate
                    ],
                    numWinners=nwinners,
                )
            )
            for pert in perturbations:
                for cand in range(ncand):
                    newPolls = basePolls.copy()
                    newPolls[cand] = min(1, max(0, newPolls[cand] + pert))
                    newWinners = set(
                        method.winnerSet(
                            [
                                strat(
                                    v,
                                    polls=newPolls,
                                    electabilities=newPolls,
                                    **stratArgs
                                )
                                for v in electorate
                            ],
                            numWinners=nwinners,
                        )
                    )
                    if newWinners != baseWinners:
                        if cand in newWinners and cand not in baseWinners:
                            changeType = 0
                        elif cand not in newWinners and cand in baseWinners:
                            changeType = 1
                        else:
                            changeType = 2
                        totalResults[pert][method.__name__][changeType] += 1
                        if cand == condWinner:
                            condResults[pert][method.__name__][changeType] += 1
                        else:
                            nonCondResults[pert][method.__name__][changeType] += 1
    return totalResults, condResults, nonCondResults
