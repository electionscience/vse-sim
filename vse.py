
import csv
import os
import multiprocessing
import time
import random

from mydecorators import autoassign, cached_property, setdefaultattr, timeit
from methods import *
from voterModels import *
from dataClasses import *
from debugDump import *

def threeRoundResults(method, voters, backgroundStrat, foregrounds=[], bgArgs = {},
                      r1Media=noisyMedia, r2Media=noisyMedia,
                      pickiness=0.3, pollingError=0.2, r2PollingError=None):
    """
    Performs three elections: a single approval voting contest in which everyone
    votes honestly to give an intentionally crude estimate of electability
    (which is filtered by r1Media),
    then an election using no information beyond the first round of "polling" in which all voters
    use backgroundStrat, and a third round which may use the results of both the prior rounds.
    The third round is repeated for each choice of foreground.
    A foreground is a (foregroundStrat, targetSelectionFunction, foregroundSelectionFunction, fgArgs) tuple
    where targetSelectionFunction receives the input of (electabilities, media(round1Results)) and
    returns (candToHelp, candToHurt).
    foregroundSelectionFunction receives the input of
    (voter, candToHelp, candToHurt, electabilities, r2Media(round1Results)) and returns a positive float
    representing the voter's eagerness to be strategic if the voter will be part of the strategic foreground
    and 0 if the voter will just use backgroundStrat.
    foregroundSelectionFunction and fgArgs are optional in each tuple.
    bgArgs and fgArgs are both dictionaries containing additional keyword arguments for strategies.
    pollingError and r2PollingError are the margins of error (2-sigma) for the polls.
    """
    if isinstance(backgroundStrat, str):
        backgroundStrat = getattr(method, backgroundStrat)
    if isinstance(foregrounds, tuple):
        foregrounds = [foregrounds]
    for i, f in enumerate(foregrounds):
        #if foregroundSelectionFunction isn't provided, use default
        if len(f) == 2:
            foregrounds[i] = (f[0], f[1], wantToHelp, {})
        elif len(f) == 3:
            if isinstance(f[2], dict):
                foregrounds[i] = (f[0], f[1], wantToHelp, f[2])
            else:
                foregrounds[i] = (f[0], f[1], f[2], {})
    if r2PollingError is None:
        r2PollingError = pollingError

    r0Results = Approval.results([useStrat(voter, Approval.zeroInfoBallot, pickiness=pickiness)
    for voter in voters])
    r0Winner = method.winner(r0Results)
    electabilities = tuple(r1Media(r0Results, pollingError))
    backgroundBallots = [useStrat(voter, backgroundStrat, electabilities=electabilities, **bgArgs)
    for voter in voters]
    r1Results = method.results(backgroundBallots)
    r1Winner = method.winner(r1Results)
    totalUtils = voters.socUtils
    winProbs = pollsToProbs(r0Results, max(pollingError, 0.05))
    #The place of the first-place candidate is 1, etc.
    r0Places = [sorted(r0Results, reverse=True).index(result) + 1 for result in r0Results]
    r1Places = [sorted(r1Results, reverse=True).index(result) + 1 for result in r1Results]

    constResults = dict(method=method.__name__, electorate=voters.id, backgroundStrat=backgroundStrat.__name__,
    numVoters=len(voters), numCandidates=len(voters[0]), magicBestUtil=max(totalUtils),
    magicWorstUtil=min(totalUtils), meanCandidateUtil=mean(totalUtils), bgArgs=bgArgs,
    r0ExpectedUtil=sum(p*u for p, u in zip(winProbs,totalUtils)),#could use electabilities instead
    r0WinnerUtil=totalUtils[r0Winner], r1WinProb=winProbs[r1Winner], r1WinnerUtil=totalUtils[r1Winner])

    allResults = [makeResults(results=r0Results, totalUtil=totalUtils[r0Winner],
            probOfWin=winProbs[r0Winner], **constResults),
            makeResults(results=r1Results, totalUtil=totalUtils[r1Winner],
            probOfWin=winProbs[r1Winner],
            winnerPlaceInR0=r0Places[r1Winner], **constResults)]
    allResults[0]['method'] = 'ApprovalPoll'
    for foregroundStrat, targetSelect, foregroundSelect, fgArgs in foregrounds:
        polls = tuple(r2Media(r1Results, pollingError))
        candToHelp, candToHurt = targetSelect(electabilities=electabilities, polls=polls, r0polls=electabilities)
        pollOrder = [cand for cand, poll in sorted(enumerate(polls),key=lambda x: -x[1])]
        foreground = [] #(voter, ballot, eagernessToStrategize) tuples
        permbgBallots = []
        for id, voter in enumerate(voters):
            eagerness = foregroundSelect(voter, candToHelp=candToHelp, candToHurt=candToHurt,
            electabilities=electabilities, polls=polls)
            if eagerness > 0:
                foreground.append((voter,
                useStrat(voter, foregroundStrat, polls=polls, electabilities=electabilities,
                candToHelp=candToHelp, candToHurt=candToHurt, **fgArgs),
                eagerness))
            else:
                permbgBallots.append(backgroundBallots[id])

        #Everything below this just analyzes the result of the final round of voting
        foreground.sort(key=lambda v:-v[2]) #from most to least eager to use strategy
        fgSize = len(foreground)
        fgBallots = [ballot for _, ballot, _ in foreground]
        fgBaselineBallots = [useStrat(voter, backgroundStrat, electabilities=electabilities, **bgArgs)
                             for voter, _, _ in foreground]
        ballots = fgBallots + permbgBallots
        results = method.results(ballots)
        winner = method.winner(results)
        #foregroundBaseUtil = sum(voter[r1Winner] for voter, _, _ in foreground)/fgSize if fgSize else 0
        #foregroundStratUtil = sum(voter[winner] for voter, _, _ in foreground)/fgSize if fgSize else 0
        totalUtil = voters.socUtils[winner]
        fgHelped = []
        fgHarmed = []
        winnersFound = [(r1Winner, 0)]
        partialResults = constResults.copy()
        if winner != r1Winner:
            winnersFound.append((winner, fgSize - 1))
        i = 1
        deciderMargUtilDiffs = []
        if fgSize: #If not I should be quitting earlier than this but easier to just fake it.
            lastVoter = foreground[fgSize - 1][0]
        else: #zero-sized foreground
            lastVoter = [0.] * len(r1Results)
        deciderUtilDiffs = [(lastVoter[winner] - lastVoter[r1Winner] , nan, fgSize)]
        allUtilDiffs = [([voter[0][winner] - voter[0][r1Winner] for voter in foreground], fgSize)]
        while i < len(winnersFound):
            thisWinner = winnersFound[i][0]
            threshold = method.stratThresholdSearch(
            thisWinner, winnersFound[i][1], permbgBallots, fgBallots, fgBaselineBallots, winnersFound)
            minfg = [voter for voter, _, _ in foreground][:threshold]
            prevWinner = method.winner(method.results(
            permbgBallots + fgBallots[:threshold-1] + fgBaselineBallots[threshold-1:]))
            if thisWinner == winner:
                prefix = "min"
            elif r1Winner == prevWinner:
                prefix = "t1"
            else: prefix = "o"#+str(i)
            partialResults.update(makePartialResults(minfg, winner, r1Winner, prefix))
            deciderUtils = foreground[threshold][0] #The deciding voter
            if threshold == 0: #this shouldn't actually matter as we'll end up ignoring it anyway
                            #, so having the wrong utilities would be OK. But let's get it right.
                predeciderUtils = [0.] * len(r1Results)
            else:
                predeciderUtils = foreground[threshold - 1][0] #The one before the deciding voter
            deciderUtilDiffs.append((predeciderUtils[thisWinner] - predeciderUtils[r1Winner],
                                    deciderUtils[thisWinner] - deciderUtils[r1Winner],
                                    threshold))
            allUtilDiffs.append(([voter[0][thisWinner] - voter[0][r1Winner] for voter in foreground[:threshold+1]],
                                    threshold))
            deciderMargUtilDiffs.append((deciderUtils[thisWinner] - deciderUtils[prevWinner], threshold, max(deciderUtils)-mean(deciderUtils)))
            i += 1
        deciderMargUtilDiffs.sort(key=lambda x:x[1])
        partialResults['deciderMargUtilDiffs'] = deciderMargUtilDiffs

        totalStratUtilDiff = 0
        margStrategicRegret = 0
        avgStrategicRegret = 0
        deciderUtilDiffs = sorted(deciderUtilDiffs, key=lambda x:x[2])
        allUtilDiffs = sorted(allUtilDiffs, key=lambda x:x[1])
        for i in range(len(deciderUtilDiffs) - 1):
            totalStratUtilDiff += ((deciderUtilDiffs[i][1] + deciderUtilDiffs[i+1][0]) / 2 * #Use average over endpoints to interpolate average over range
                                    (deciderUtilDiffs[i+1][2] - deciderUtilDiffs[i][2]))
            margStrategicRegret += sum(allUtilDiffs[i+1][0][allUtilDiffs[i][1]:allUtilDiffs[i+1][1]])
            avgStrategicRegret += mean(allUtilDiffs[i+1][0]) * (allUtilDiffs[i+1][1] - allUtilDiffs[i][1])
        partialResults['totalStratUtilDiff'] = totalStratUtilDiff
        partialResults['margStrategicRegret'] = margStrategicRegret
        partialResults['avgStrategicRegret'] = avgStrategicRegret

        partialResults.update(makePartialResults([voter for voter, _, _ in foreground], winner, r1Winner, ""))
        allResults.append(makeResults(results=results, fgStrat = foregroundStrat.__name__,
        fgTargets=targetSelect.__name__, fgArgs=fgArgs,
        winnerPlaceInR0=r0Places[winner], winnerPlaceInR1=r1Places[winner],
        probOfWin=winProbs[winner], numWinnersFound=len(winnersFound), totalUtil=totalUtil,
        pivotalUtilDiff=deciderMargUtilDiffs[0][0]/deciderMargUtilDiffs[0][2] if deciderMargUtilDiffs else 0,
        factionSize=fgSize, factionFraction = (deciderMargUtilDiffs[0][1]+1)/fgSize if deciderMargUtilDiffs else None,
        deciderUtilDiffSum=sum(uDiff for uDiff, _, _ in deciderMargUtilDiffs), **partialResults))
    return allResults

class CsvBatch:
    #@timeit
    #@autoassign
    def __init__(self, model, methodsAndStrats,
            nvot, ncand, niter, r1Media=noisyMedia, r2Media=noisyMedia, seed=None,
            pickiness=0.4, pollingError=0.2):
        """methodsAndStrats is a list of (votingMethod, backgroundStrat, foregrounds, bgArgs).
        A voting method may be given in place of such a tuple, in which case backgroundSrat, foregrounds, and bgArgs
        will be determined automatically.
        foregrounds are (foregroundStrat, targetSelectionFunction, foregroundSelectionFunction, fgArgs) tuples.
        foregroundSelectionFunction is optional.
        nvot and ncand gives the number of voters and candidates in each election, and niter is how many
        electorates will be generated and have all the methods and strategies run on them.
        """
        self.rows = []
        if (seed is None):
            seed = time.time()
            self.seed = seed
        random.seed(seed)
        ms = []
        for m in methodsAndStrats:
            if isinstance(m, type) and issubclass(m, Method):
                fgs = m.defaultfgs()
                for bg in m.defaultbgs():
                    if isinstance(bg, tuple):
                        ms.append((m, bg[0], fgs, bg[1]))
                    else:
                        ms.append((m, bg, fgs, {}))
                #fgs = []
                #for targetFunc in [select21, select31, selectRand, select012]:
                    #fgs.extend([(m.diehardBallot, targetFunc, {'intensity':i}) for i in m.diehardLevels]
                    #+ [(m.compBallot, targetFunc, {'intensity':i}) for i in m.compLevels])
                    #fgs.append((m.vaBallot, targetFunc, {'info':'e'}))
                #for bg in [m.honBallot, m.vaBallot]:
                    #ms.append((m, bg, fgs, {'pollingUncertainty':0.4}))
            else:
                ms.append(m)
        args = (model, nvot, ncand, ms, pickiness, pollingError, r1Media, r2Media)
        with multiprocessing.Pool(processes=7) as pool:
            results = pool.starmap(oneStepWorker, [args + (seed, i) for i in range(niter)])
            for result in results:
                self.rows.extend(result)

        for row in self.rows: row['voterModel'] = str(model)

    def saveFile(self, baseName="SimResults", newFile=True):
        """print the result of doVse in an accessible format.
        for instance:

        csvs.saveFile()
        """
        i = 1
        if newFile:
            while os.path.isfile(baseName + str(i) + ".csv"):
                i += 1
        myFile = open(baseName + (str(i) if newFile else "") + ".csv", "w")
        dw = csv.DictWriter(myFile, self.rows[0].keys(), restval="NA")
        dw.writeheader()
        for r in self.rows:
            dw.writerow(r)
        myFile.close()

def oneStepWorker(model, nvot, ncand, ms, pickiness, pollingError, r1Media, r2Media, baseSeed=None, i = 0):

    if i>0 and i%50 == 0: print('Iteration:', i)
    if baseSeed is not None:
        random.seed(baseSeed + i)

    electorate = model(nvot, ncand)
    rows = []
    for method, bgStrat, fgs, bgArgs in ms:
        results = threeRoundResults(method, electorate, bgStrat, fgs, bgArgs=bgArgs,
                r1Media=r1Media, r2Media=r2Media, pickiness = pickiness, pollingError = pollingError)
        for result in results:
            result.update(dict(
                    seed = baseSeed + i,
                    pickiness = pickiness,
                    pollingError = pollingError,
                ))
        rows.extend(results)
    scenario = Condorcet.scenarioType(electorate)
    for row in rows:
        row["scenarioType"] = scenario
    return rows

class CsvBatches(CsvBatch):
    def __init__(self, model, methodsAndStrats,
            nvot, ncand, niter, r1Media=noisyMedia, r2Media=noisyMedia, seed=None,
            pickiness=0.4, pollingError=0.2):
        """Just like CsvBatch, but you can replace an argument such as nvot with a list of values for that
        argument to run a CsvBatch for every item of that list. The results appear in self.rows.
        """
        possibleListArgs = [model, nvot, ncand, r1Media, r2Media, pickiness, pollingError]
        listArgs = [arg if isinstance(arg, list) else [arg] for arg in possibleListArgs]
        argsList = listProduct(listArgs) #each entry is a list of arguments to be passed to one call of CsvBatch
        self.rows = []
        for a in argsList:
            self.rows.extend(CsvBatch(a[0], methodsAndStrats, a[1], a[2], niter,
                    r1Media=a[3], r2Media=a[4], seed=seed, pickiness=a[5], pollingError=a[6]).rows)

def listProduct(lists, index=0):
    """A Cartesian product for lists
    >>> listProduct([[1,2],[3,4]])
    [[1, 3], [2, 3], [1, 4], [2, 4]]
    """
    if len(lists) < 2:
        return [[i] for i in lists[0]]
    returnList = []
    for other in listProduct(lists[1:]):
        for item in lists[0]:
            returnList.append([item] + other)
    return returnList

if __name__ == "__main__":
    import doctest
    doctest.testmod()
