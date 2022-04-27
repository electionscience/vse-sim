"""Candidate Incentive Distribution"""
import random
import time
import csv
import os
import multiprocessing
import re
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean, std

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

class utilDeviation:
    def __init__(self, voters, **kw): pass
    def score(self, voter, candIndex):
        return voter[candIndex] - mean(voter)

class normalizedUtilDeviation:
    def __init__(self, voters, **kw): pass
    def score(self, voter, candIndex):
        return (voter[candIndex] - mean(voter))/std(voter)

class weightedUtilDev:
    """
    Like normalizedUtilDeviation, but weights the means and standard
    >>> chdh = [Voter(us) for us in [[10,9,1,0]]*10+[[9,10,1,0]]*9+[[0,1,10,2]]*15+[[0,1,2,20]]]
    >>> wcd = weightedUtilDev(chdh)
    >>> wcd.score(chdh[0], 0)
    0.23824257425742565
    >>> wcd.score(chdh[0], 1)
    0.18409653465346526
    >>> wcd.score(chdh[0], 2)
    -0.24907178217821788
    >>> wcd.score(chdh[0], 3)
    -0.30321782178217827
    >>> wcd.score(chdh[-1], 0)
    -0.15965671872583886
    >>> wcd.score(chdh[-1], 1)
    -0.06494510592237515
    >>> wcd.score(chdh[-1], 2)
    0.02976650688108858
    >>> wcd.score(chdh[-1], 3)
    1.7345755373434355
    >>> ch = [Voter(us) for us in [[10,9,1]]*10+[[9,10,1]]*9+[[0,1,10]]*15]
    >>> wc = weightedUtilDev(ch)
    >>> wc.score(ch[0],0)
    0.23448275862068957
    >>> wc.score(ch[0],2)
    -0.2637931034482759
    """
    def __init__(self, voters, strat=Plurality.honBallot, method=Plurality, stratArgs={}):
        unnormalizedSupport = method.results(strat(v, **stratArgs) for v in voters)
        normFactor = sum(unnormalizedSupport)
        self.candSupport = [s/normFactor for s in unnormalizedSupport]
        self.scores = {}
        for voter in voters:
            wmean = sum(util*self.candSupport[i] for i, util in enumerate(voter))
            wstd = sum(s*(u-wmean)**2 for s, u in zip(self.candSupport, voter))
            self.scores[voter.id] = [(util - wmean)/wstd for util in voter]

    def score(self, voter, candIndex):
        return self.scores[voter.id][candIndex]

def influentialBlocs(voters, method, numWinners=1, utilChange=0.1, numBuckets=5, sorter=normalizedUtilDeviation,
                    strat=None, stratArgs = {}, pollingMethod=Approval, pollingStrat=Approval.zeroInfoBallot,
                    pollingStratArgs={'pickiness':0.7}, media=noisyMedia, pollingError=0.2, sorterArgs={}):
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
        polls = media(pollingMethod.results(pollBallots), pollingError) #method.results can't depend on numWinners; this may need to be changed
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
            numBuckets=24, sorter=normalizedUtilDeviation, utilChange=0.1,
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

    def summarize(self):
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
        incentFracts = {name: [i*self.numBuckets/sum(incents) for i in incents] for name, incents in allIncents.items()}
        return incentFracts, allIncents, loserIncents, winnerIncents

    def chart(self, methodOnly=True):
        fig, ax = plt.subplots()
        incentFracts = self.summarize()[0]
        if methodOnly:
            incentFracts = {re.match(".*(?=:)", name).group(0): data for name, data in incentFracts.items()}
        for name, data in incentFracts.items():
            ax.plot([i/self.numBuckets for i in range(1, self.numBuckets+1)], data, label=name)
        #ax.set_xlim(1, self.numBuckets)
        ax.set_xlabel("Voter's support for candidate")
        ax.set_ylabel("Candidate's incentive to appeal to voter")
        ax.grid(True)
        ax.legend()
        plt.show()

    def saveFile(self, baseName="cidResults", newFile=True):
        """Prints the aggregated simulation results in an accessible format.
        """
        i = 1
        if newFile:
            while os.path.isfile(baseName + str(i) + ".csv"):
                i += 1

        universalInfo = {'nVoters': self.nvot, 'nCands': self.ncand, 'iterations': self.niter,
                    'winners': self.nwinners, 'model': str(self.model), 'sorter': self.sorter.__name__,
                    'numBuckets': self.numBuckets, 'utilChange': self.utilChange,
                    'pollingMethod': self.pollingMethod.__name__, 'pollingError': self.pollingError}
        fields = ['name', 'baseResult'] + list(range(self.numBuckets)) + list(universalInfo.keys())
        resultTypeIndices = {'all': 1, 'loss': 2, 'win': 3}

        myFile = open(baseName + (str(i) if newFile else "") + ".csv", "w")
        dw = csv.DictWriter(myFile, fields, restval="data missing")
        dw.writeheader()
        for outcome, index in resultTypeIndices.items():
            for name, results in self.summarize()[index].items():
                row = {'name': name, 'baseResult': outcome}
                row.update({i: result for i, result in enumerate(results)})
                row.update(universalInfo)
                dw.writerow(row)
        myFile.close()

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

def showChart(fileName, norm=1, methodOnly=True, forResult='all'):
    with open(fileName) as file:
        reader = csv.DictReader(file)
        fig, ax = plt.subplots()
        for row in reader:
            if row['baseResult'] != forResult: continue
            buckets = int(row['numBuckets'])
            rawData = [float(row[str(i)]) for i in range(buckets)]
            if norm == 1:
                normFactor = sum(rawData)/buckets
            elif norm == 'max':
                normFactor = max(rawData)
            data = [d/normFactor for d in rawData]
            name = re.match(".*(?=:)", row['name']).group(0) if methodOnly else row['name']
            ax.plot([(i+.5)/buckets for i in range(buckets)], data, label=name)
        ax.set_xlabel("Voter's support for candidate")
        ax.set_ylabel("Candidate's incentive to appeal to voter")
        ax.grid(True)
        ax.legend()
        plt.show()
