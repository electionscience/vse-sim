"""Candidate Incentive Distribution"""
import random
import time
import csv
import os
import multiprocessing

from mydecorators import *
from methods import *
from mwmethods import *
from voterModels import *
from dataClasses import *
from debugDump import *

class candAffinity:
    """
    >>> c = candAffinity([[0,1,2]]*3+[[0,2,1]]*3+[[2,1,0]]*4)
    >>> c.score([0,1,2],0)
    0.2
    >>> c.score([0,1,2],1)
    0.55
    >>> c.score([0,1,2],2)
    0.85
    """
    def __init__(self, voters):
        self.candSupport = Plurality.results(Plurality.honBallot(v) for v in voters)
    def score(self, voter, candIndex):
        """
        Gives a number from 0 to 1; higher numbers show greater support for the candidate from the voter
        """
        myUtil = voter[candIndex]
        return sum((sign(myUtil - u) + 1)*s for u, s in zip(voter, self.candSupport))/2

def influentialBlocs(voters, method, numWinners=1, utilChange=0.1, numBuckets=5, sorter=candAffinity,
                    strat=None, stratArgs = {}, pollingMethod=Approval, pollingStrat=Approval.zeroInfoBallot,
                    pollingStratArgs={'pickiness':0.3}, media=noisyMedia, pollingError=0.2):
    """
    Uses voters as the base electorate to determine whether candidates have an incentive to appeal
    to various voting blocs, where the voting blocs are determined by ranking voters according to sorter
    and placing them in numBuckets equally (if numBuckets divides the number of voters) large blocs.

    If strat is None, the voters will use method.honBallot and no polling will be performed.
    Otherwise, voters will use strat with stratArgs and polls determined by the relevant arguments.
    Polling uses Approval Voting by default; it can also be set to None to use the voting method in the election
    or to an arbitrary voting method. Polling is then randomly perturbed according to pollingError.

    >>> influentialBlocs([Voter([0,1,2])]*3+[Voter([0,2,1])]*3+[Voter([2,1,0])]*4, Plurality, utilChange = 1.5)
    ([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0], [0, 0, 1, 0, 0]], [0], [0.4, 0.3, 0.3])
    """
    if strat is None:
        strat = method.honBallot
        polls = None
        baseBallots = [method.honBallot(v) for v in voters]
    else:
        if pollingMethod is None:
            pollingMethod = method
            pollingStrat = method.honBallot
        pollBallots = [pollingStrat(v, numWinners=numWinners, **pollingStratArgs) for v in voters]
        polls = media(method.results(pollBallots), pollingError) #method.results can't depend on numWinners; this may need to be changed
        baseBallots = [strat(v,  polls=polls, electabilities=polls, numWinners=numWinners, **stratArgs) for v in voters]
    if isinstance(sorter, type):
        sorter = sorter(voters)
    baseWinners = method.winnerSet(baseBallots, numWinners)
    baseResults = method.results(baseBallots)
    numCands = len(voters[0])
    numVoters = len(voters)
    isIncentive = [[] for i in range(numCands)]
    #isIncentive [c][b] is 1 if candidate c is incentivized to appeal to the bth bucket of voters, 0 otherwise
    for cand in range(numCands):
        utilShifts = [0]*numCands
        utilShifts[cand] = -utilChange if cand in baseWinners else utilChange
        sVoters, sBallots = zip(*sorted(zip(voters, baseBallots), key=lambda x: sorter.score(x[0], cand)))
        sortedVoters, sortedBallots = list(sVoters), list(sBallots)
        for b in range(numBuckets):
            newBallots = [strat(v.addUtils(utilShifts), polls=polls, electabilities=polls, numWinners=numWinners, **stratArgs)
                        for v in sortedVoters[int(b*numVoters/numBuckets):int((b+1)*numVoters/numBuckets)]]
            ballots = (sortedBallots[:int(b*numVoters/numBuckets)] + newBallots
                        + sortedBallots[int((b+1)*numVoters/numBuckets):])
            newWinners = method.winnerSet(ballots, numWinners=numWinners)
            if (cand in baseWinners and cand not in newWinners)\
                or (cand not in baseWinners and cand in newWinners):
                isIncentive[cand].append(1)
            else: isIncentive[cand].append(0)
    return isIncentive, baseWinners, baseResults


class CID:
    @autoassign
    def __init__(self, model, methodsAndStrats, nvot, ncand, niter, nwinners=1,
            numBuckets=5, sorter=candAffinity, utilChange=0.1,
            media=noisyMedia, pollingMethod=Approval, pollingError=0.2, seed=None, ):
        """methodsAndStrats is a list of (votingMethod, strat, stratArgs); stratArgs is optional.
        A voting method may be given in place of such a tuple, in which case honBallot will be used.
        nvot and ncand give the number of voters and candidates in each election, and niter is how many
        electorates will be generated and have all the methods and strategies run on them.

        >>> cid = CID(KSModel(), [Plurality, Irv, Approval, STAR, Minimax], 25, 6, 1)
        """
        if (seed is None):
            seed = time.time()
            self.seed = seed
        random.seed(seed)
        ms = []
        for m in methodsAndStrats:
            if isinstance(m, type):
                ms.append((m, m.honBallot, {}))
            elif len(m) == 2:
                ms.append((m[0], m[1], {}))
            else: ms.append(m)
        self.mNames = [m[0].__name__ + ':' + m[1].__name__ + str(m[2]) for m in ms]
        self.rows = []
        args = (model, nvot, ncand, ms, nwinners, utilChange, numBuckets, sorter, pollingMethod, pollingError, media)
        with multiprocessing.Pool(processes=7) as pool:
            results = pool.starmap(simOneElectorate, [args + (seed, i) for i in range(niter)])
            for result in results:
                self.rows.extend(result)

    def histograms(self):
        nb = len(self.rows[0]['incentives']) #number of buckets
        winnerIncents = {name:[0]*nb for name in self.mNames}
        loserIncents = {name:[0]*nb for name in self.mNames}
        allIncents = {name:[0]*nb for name in self.mNames}
        for row in self.rows:
            name = row['method'] + ':' + row['strat'] + str(row['stratArgs'])
            for i in range(nb):
                if row['isWinner']:
                    winnerIncents[name][i] += row['incentives'][i]
                else:
                    loserIncents[name][i] += row['incentives'][i]
                allIncents[name][i] += row['incentives'][i]
        incentFracts = {name: [i/sum(incents) for i in incents] for name, incents in allIncents.items()}
        return incentFracts, allIncents, loserIncents, winnerIncents

def simOneElectorate(model, nvot, ncand, ms, nwinners, utilChange, numBuckets, sorter,
                    pollingMethod, pollingError, media, baseSeed=None, i = 0):
    if i>0 and i%100 == 0: print('Iteration:', i)
    if baseSeed is not None:
        random.seed(baseSeed + i)

    electorate = model(nvot, ncand)
    results = []
    for method, strat, stratArgs in ms:
        allIncentives, baseWinners, baseResults = influentialBlocs(electorate, method, nwinners, utilChange, numBuckets,
                sorter, strat, stratArgs, pollingMethod, media=media, pollingError=pollingError)
        for i, candIncentives in enumerate(allIncentives):
            results.append(dict(incentives=candIncentives, isWinner=i in baseWinners,
                method=method.__name__, strat=strat.__name__, stratArgs=stratArgs, voterModel=str(model)))
    return results
