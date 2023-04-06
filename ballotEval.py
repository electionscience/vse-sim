from methods import *
from voterModels import *
from stratFunctions import *


def stratStats(
    model,
    strat,
    nvot,
    ncand,
    niter,
    stratArgs=None,
    numWinners=1,
    pollingError=0.2,
    usePolls=True,
    pickiness=0.7,
    pollFilter=None,
):
    if stratArgs is None:
        stratArgs = {}
    bulletCount, totalScore = 0, 0
    for _ in range(niter):
        electorate = model(nvot, ncand)
        if usePolls:
            pollBallots = [
                Approval.zeroInfoBallot(v, pickiness=pickiness) for v in electorate
            ]
            polls = noisyMedia(Approval.results(pollBallots), pollingError)
        else:
            polls = None
        ballots = [
            strat(voter, electabilities=polls, numWinners=numWinners, **stratArgs)
            for voter in electorate
        ]
        for ballot in ballots:
            ballotSum = sum(ballot)
            totalScore += ballotSum
            if ballotSum == max(ballot):
                bulletCount += 1
    return bulletCount / (nvot * niter), totalScore / (nvot * ncand * niter)
