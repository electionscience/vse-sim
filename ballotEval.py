from methods import *
from voterModels import *
from stratFunctions import *

import collections

def stratStats(model, strat, nvot, ncand, niter, stratArgs={}, numWinners=1, pollingError=0.2,
                usePolls=True, pickiness=0.7, pollFilter=None):
    bulletCount, totalScore = 0, 0
    scoreCounts = collections.Counter()
    for i in range(niter):
        electorate = model(nvot, ncand)
        if usePolls:
            pollBallots = [Approval.zeroInfoBallot(v, pickiness=pickiness) for v in electorate]
            polls = noisyMedia(Approval.results(pollBallots), pollingError)
        else: polls = None
        ballots = [strat(voter, electabilities=polls, numWinners=numWinners, **stratArgs) for voter in electorate]
        for ballot in ballots:
            ballotSum = sum(ballot)
            totalScore += ballotSum
            for score in ballot:
                scoreCounts[score] += 1
            if ballotSum == max(ballot):
                bulletCount += 1
    scoreList = [0]*(max(int(s) for s in scoreCounts.keys()) + 1)
    for score, count in scoreCounts.items():
        scoreList[int(score)] += count/(nvot*ncand*niter)
    return bulletCount/(nvot*niter), totalScore/(nvot*ncand*niter), scoreList
