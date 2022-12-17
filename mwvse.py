import math
import functools
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median

from vse import *
from mwmethods import *
from methods import *
from stratFunctions import *
from voterModels import *

class ThreeRoundResults:
    @autoassign
    def __init__(self, method, voters, backgroundStrat, foregrounds=[], bgArgs = {},
                          r1Media=noisyMedia, r2Media=noisyMedia,
                          pickiness=0.3, pollingError=0.2,
                          pollingMethod=Approval, pollingStrat=None, pollingStratArgs={},
                          reportRound0=False, numWinners=1):
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
        self.configureInputs()
        self.numCands = len(self.voters[0])
        self.round0()
        self.electabilities = tuple(r1Media(self.r0Results, self.pollingError))
        #The place of the first-place candidate is 1, etc.
        self.r0Places = [sorted(self.r0Results, reverse=True).index(result) + 1 for result in self.r0Results]
        self.backgroundBallots = [useStrat(voter, self.backgroundStrat, electabilities=self.electabilities, **self.bgArgs)
                            for voter in self.voters]
        self.round1()

        self.totalUtils = self.voters.socUtils
        self.winProbs = pollsToProbs(self.r0Results, max(self.pollingError, 0.05))


        self.constResults = dict(method=self.method.__name__, electorate=self.voters.id, backgroundStrat=self.backgroundStrat.__name__,
                                numVoters=len(self.voters), bgArgs=self.bgArgs, numCandidates=self.numCands)

        self.addConstResults()
        self.allResults = []
        if self.reportRound0:
            self.addRound0Results()
        self.addRound1Results()
        for foregroundStrat, targetSelect, foregroundSelect, fgArgs in self.foregrounds:
            self.foregroundStrat, self.targetSelect, self.fgArgs = foregroundStrat, targetSelect, fgArgs
            polls = self.setR2Polls()
            candToHelp, candToHurt = targetSelect(electabilities=self.electabilities, polls=polls, r0polls=self.electabilities)
            pollOrder = [cand for cand, poll in sorted(enumerate(polls),key=lambda x: -x[1])]
            self.foreground = [] #(voter, ballot, eagernessToStrategize) tuples
            permbgBallots = []
            for id, voter in enumerate(voters):
                eagerness = foregroundSelect(voter, candToHelp=candToHelp, candToHurt=candToHurt,
                electabilities=self.electabilities, polls=polls)
                if eagerness > 0:
                    self.foreground.append((voter,
                            useStrat(voter, foregroundStrat, polls=polls, electabilities=self.electabilities,
                            candToHelp=candToHelp, candToHurt=candToHurt, **fgArgs),
                            eagerness))
                else:
                    permbgBallots.append(self.backgroundBallots[id])

            #Everything below this just analyzes the result of the final round of voting
            self.foreground.sort(key=lambda v:-v[2]) #from most to least eager to use strategy
            self.fgSize = len(self.foreground)
            fgBallots = [ballot for _, ballot, _ in self.foreground]
            fgBaselineBallots = [useStrat(voter, self.backgroundStrat, electabilities=self.electabilities, **bgArgs)
                                 for voter, _, _ in self.foreground]
            self.ballots = fgBallots + permbgBallots
            self.round2()

            #foregroundBaseUtil = sum(voter[r1Winner] for voter, _, _ in foreground)/fgSize if fgSize else 0
            #foregroundStratUtil = sum(voter[winner] for voter, _, _ in foreground)/fgSize if fgSize else 0

            if self.fgSize: #If not I should be quitting earlier than this but easier to just fake it.
                self.lastVoter = self.foreground[self.fgSize - 1][0]
            else: #zero-sized foreground
                self.lastVoter = [0.] * len(self.r1Results)
            self.initWinnerLoop()
            i = 1
            while i < len(self.winnersFound):
                self.thisWinner = self.winnersFound[i][0]
                self.threshold = self.stratThresholdSearch(method,
                        self.thisWinner, self.winnersFound[i][1], permbgBallots, fgBallots, fgBaselineBallots, self.winnersFound)
                minfg = [voter for voter, _, _ in foreground][:threshold]
                self.prevWinner = self.winnerFunc(
                    permbgBallots + fgBallots[:threshold-1] + fgBaselineBallots[threshold-1:])
                if self.thisWinner == self.winner:
                    prefix = "min"
                elif self.r1Winner == self.prevWinner:
                    prefix = "t1"
                else: prefix = "o"#+str(i)
                self.partialResults.update(self.makePartialResults(minfg, self.winner, self.r1Winner, prefix))
                self.deciderUtils = self.foreground[self.threshold][0] #The deciding voter
                if self.threshold == 0: #this shouldn't actually matter as we'll end up ignoring it anyway
                                #, so having the wrong utilities would be OK. But let's get it right.
                    self.predeciderUtils = [0.] * len(self.r1Results)
                else:
                    self.predeciderUtils = self.foreground[self.threshold - 1][0] #The one before the deciding voter
                self.winnerLoopStats()
                i += 1

            self.finalStats()



    def configureInputs(self):
        if isinstance(self.backgroundStrat, str):
            self.backgroundStrat = getattr(self.method, self.backgroundStrat)
        if isinstance(self.foregrounds, tuple):
            self.foregrounds = [self.foregrounds]
        for i, f in enumerate(self.foregrounds):
            #use defaults for foreground and target selection
            if not isinstance(f, tuple):
                self.foregrounds[i] = (f, selectRand, wantToHelp, {})
            elif len(f) == 2:
                if isinstance(f[1], dict):
                    self.foregrounds[i] = (f[0], selectRand, wantToHelp, f[1])
                else:
                    self.foregrounds[i] = (f[0], f[1], wantToHelp, {})
            elif len(f) == 3:
                if isinstance(f[2], dict):
                    self.foregrounds[i] = (f[0], f[1], wantToHelp, f[2])
                else:
                    self.foregrounds[i] = (f[0], f[1], f[2], {})
        if self.pollingStrat is None:
            self.pollingStrat = Approval.zeroInfoBallot
            self.pollingStratArgs = {'pickiness': self.pickiness}

    def round0(self):
        self.r0Results = self.pollingMethod.results([useStrat(voter, self.pollingStrat, **self.pollingStratArgs)
                                        for voter in self.voters])
        self.r0Winner = self.pollingMethod.winner(self.r0Results)
    def round1(self):
        self.r1Results = self.method.results(self.backgroundBallots)
        self.r1Winner = self.method.winner(self.r1Results)
        self.r1Places = [sorted(self.r1Results, reverse=True).index(result) + 1 for result in self.r1Results]

    def addConstResults(self):
        self.constResults.update(dict(magicBestUtil=max(self.totalUtils),
            magicWorstUtil=min(self.totalUtils), meanCandidateUtil=mean(self.totalUtils),
            r0ExpectedUtil=sum(p*u for p, u in zip(self.winProbs,self.totalUtils)),#could use electabilities instead
            r0WinnerUtil=self.totalUtils[self.r0Winner], r1WinProb=self.winProbs[self.r1Winner],
            r1WinnerUtil=self.totalUtils[self.r1Winner]))

    def addRound0Results(self):
        r = self.makeResults(results=self.r0Results, totalUtil=self.totalUtils[r0Winner],
                probOfWin=self.winProbs[r0Winner], **self.constResults)
        r['method'] = self.pollingMethod.__name__+'Poll'
        self.allResults.append(r)

    def addRound1Results(self):
        self.allResults.append(self.makeResults(results=self.r1Results, totalUtil=self.totalUtils[self.r1Winner],
                probOfWin=self.winProbs[self.r1Winner], winnerPlaceInR0=self.r0Places[self.r1Winner],
                **self.constResults))

    def setR2Polls(self):
        return tuple(self.r2Media(self.r1Results, self.pollingError))

    def round2(self):
        self.results = self.method.results(self.ballots)
        self.winner = self.method.winner(self.results)

    def initWinnerLoop(self):
        self.totalUtil = self.voters.socUtils[self.winner]
        self.fgHelped = []
        self.fgHarmed = []
        self.winnersFound = [(self.r1Winner, 0)]
        self.partialResults = self.constResults.copy()
        if self.winner != self.r1Winner:
            self.winnersFound.append((self.winner, self.fgSize - 1))
        self.deciderMargUtilDiffs = []
        self.deciderUtilDiffs = [(self.lastVoter[self.winner] - self.lastVoter[self.r1Winner] , nan, self.fgSize)]
        self.allUtilDiffs = [([voter[0][self.winner] - voter[0][self.r1Winner] for voter in self.foreground], self.fgSize)]

    def stratThresholdSearch(self, targetWinner, foundAt, bgBallots, fgBallots, fgBaselineBallots, winnersFound):
        """Returns the minimum number of strategist needed to elect targetWinner
        and modifies winnersFound to include any additional winners found during the search"""
        maxThreshold, minThreshold = foundAt, 0
        while maxThreshold > minThreshold: #binary search for min foreground size
            midpoint = int(floor((maxThreshold + minThreshold)/2))
            midpointBallots = bgBallots + fgBallots[:midpoint] + fgBaselineBallots[midpoint:]
            midpointWinner = self.winnerFunc(midpointBallots)
            if not any(midpointWinner == w for w, _ in winnersFound):
                winnersFound.append((midpointWinner, midpoint))
            if midpointWinner == targetWinner:
                maxThreshold = midpoint
            else:
                minThreshold = midpoint + 1
        return maxThreshold

    def winnerFunc(self, ballots):
        return self.method.winner(self.method.results(ballots))

    def resultFunc(self, ballots):
        return self.method.results(ballots)

    def winnerLoopStats(self):
        self.deciderUtilDiffs.append((self.predeciderUtils[self.thisWinner] - self.predeciderUtils[self.r1Winner],
                                self.deciderUtils[self.thisWinner] - self.deciderUtils[self.r1Winner],
                                self.threshold))
        self.allUtilDiffs.append(([voter[0][self.thisWinner] - voter[0][self.r1Winner] for voter in self.foreground[:self.threshold+1]],
                                self.threshold))
        self.deciderMargUtilDiffs.append((self.deciderUtils[self.thisWinner] - self.deciderUtils[self.prevWinner], self.threshold))

    def finalStats(self):
        totalStratUtilDiff = 0
        margStrategicRegret = 0
        avgStrategicRegret = 0
        self.deciderMargUtilDiffs.sort(key=lambda x:x[1])
        self.deciderUtilDiffs = sorted(self.deciderUtilDiffs, key=lambda x:x[2])
        self.allUtilDiffs = sorted(self.allUtilDiffs, key=lambda x:x[1])
        for i in range(len(self.deciderUtilDiffs) - 1):
            totalStratUtilDiff += ((self.deciderUtilDiffs[i][1] + self.deciderUtilDiffs[i+1][0]) / 2 * #Use average over endpoints to interpolate average over range
                                    (self.deciderUtilDiffs[i+1][2] - self.deciderUtilDiffs[i][2]))
            margStrategicRegret += sum(self.allUtilDiffs[i+1][0][self.allUtilDiffs[i][1]:self.allUtilDiffs[i+1][1]])
            avgStrategicRegret += mean(self.allUtilDiffs[i+1][0]) * (self.allUtilDiffs[i+1][1] - self.allUtilDiffs[i][1])
        self.partialResults['deciderMargUtilDiffs'] = self.deciderMargUtilDiffs
        self.partialResults['totalStratUtilDiff'] = totalStratUtilDiff
        self.partialResults['margStrategicRegret'] = margStrategicRegret
        self.partialResults['avgStrategicRegret'] = avgStrategicRegret

        self.partialResults.update(self.makePartialResults([voter for voter, _, _ in self.foreground], self.winner, self.r1Winner, ""))
        self.allResults.append(self.makeResults(results=self.results, fgStrat = self.foregroundStrat.__name__,
            fgTargets=self.targetSelect.__name__, fgArgs=self.fgArgs,
            winnerPlaceInR0=self.r0Places[self.winner], winnerPlaceInR1=self.r1Places[self.winner],
            probOfWin=self.winProbs[self.winner], numWinnersFound=len(self.winnersFound), totalUtil=self.totalUtil,
            pivotalUtilDiff=self.deciderMargUtilDiffs[0][0] if self.deciderMargUtilDiffs else 0,
            deciderUtilDiffSum=sum(uDiff for uDiff, _ in self.deciderMargUtilDiffs), **self.partialResults))

    def makeResults(self, **kw):
        results = {c: kw.get(c, None) for c in resultColumns}
        results.update(kw)
        return results

    def makePartialResults(self, fgVoters, winner, r1Winner, prefix=""):
        fgHelped = []
        fgHarmed = []
        numfg = len(fgVoters)
        if winner != r1Winner:
            for voter in fgVoters:
                if voter[winner] > voter[r1Winner]:
                    fgHelped.append(voter)
                elif voter[winner] < voter[r1Winner]:
                    fgHarmed.append(voter)

        tempDict = dict(fgUtil=sum(v[winner] for v in fgVoters)/numfg if numfg else 0,
        fgUtilDiff=sum(v[winner] - v[r1Winner] for v in fgVoters)/numfg if numfg else 0, fgSize=numfg,
        fgNumHelped=len(fgHelped), fgHelpedUtil=sum(v[winner] for v in fgHelped),
        fgHelpedUtilDiff= sum(v[winner] - v[r1Winner] for v in fgHelped),
        fgNumHarmed=len(fgHarmed), fgHarmedUtil=sum(v[winner] for v in fgHarmed),
        fgHarmedUtilDiff= sum(v[winner] - v[r1Winner] for v in fgHarmed))
        return{prefix+key:value for key, value in tempDict.items()}

@functools.lru_cache(maxsize=3)
def winnerSetSample(numCands, numWinners):
    """
    Returns a list of possible winner sets. Is exhaustive if the list would be short, random otherwise.
    winnerSetSample(5,2)
    [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}]
    """
    if math.comb(numCands, numWinners) < 100:
        winners = list(range(numWinners))
        winnerSets = [set(winners)]
        while winners[0] < numCands - numWinners:
            for i in range(numWinners-1, -1, -1):
                if winners[i] < numCands - (numWinners - i) and (i==numWinners-1 or winners[i+1] - winners[i] > 1):
                    winners[i] += 1
                    for j in range(i+1, numWinners):
                        winners[j] = winners[i] + j - i
                    break
            winnerSets.append(set(winners))
        return winnerSets
    else:
        return [set(random.sample(range(numCands), numWinners)) for i in range(50)]

mwResultColumns = ["method", "backgroundStrat", "fgStrat", "numVoters", "numCandidates",
        "numWinners", "magicBestUtil", "magicWorstUtil", "meanCandidateUtil",
        "results", "bgArgs", "fgArgs", "totalUtil", "deciderUtilDiffs", "fgTargets", "numWinnersFound"]
for stat in ["mean", "max", "median"]:
    for columnName in ["fgUtil", "fgUtilDiff", "fgSize",
            "fgNumHelped", "fgHelpedUtil", "fgHelpedUtilDiff",
            "fgNumHarmed", "fgHarmedUtil", "fgHarmedUtilDiff",
            "totalUtil", "deciderUtilDiffs",
            "highUtil", "randUtil",
            "pivotalUtilDiff", "deciderUtilDiffSum", "deciderMargUtilDiffs"]:
        resultColumns.append(stat + columnName)
class TRRMW(ThreeRoundResults):
    def configureInputs(self):
        if self.pollingStrat is None:
            self.pollingStrat = Plurality.honBallot
        super().configureInputs()

    def round1(self):
        self.round1Winners = self.method.winnerSet(self.backgroundBallots)

    def addConstResults(self):
        self.rwMeanUtil = mean(self.totalUtils) #rw = random winner
        possOutcomes = winnerSetSample(self.numCands, self.numWinners)
        self.constResults.update(dict(
            randMeanUtil = mean(self.totalUtils),
            randBestUtil = mean(max(v[c] for c in winnerSet) for winnerSet in possOutcomes for v in self.voters),
            randMedianUtil = mean(median(v[c] for c in winnerSet) for winnerSet in possOutcomes for v in self.voters),
            highMeanUtil = max(mean(mean(v[c] for c in winnerSet) for v in self.voters) for winnerSet in possOutcomes),
            highBestUtil = max(mean(max(v[c] for c in winnerSet) for v in self.voters) for winnerSet in possOutcomes),
            highMedianUtil = max(mean(median(v[c] for c in winnerSet) for v in self.voters) for winnerSet in possOutcomes),
            meanUtil = mean(v[w] for v in self.voters for w in self.round1Winners),
            maxUtil = mean(max(v[w] for w in self.round1Winners) for v in self.voters),
            medianUtil = mean(medan(v[w] for w in self.round1Winners) for v in self.voters)
        ))

    def addRound0Results(self): pass
    def addRound1Results(self): pass
    def setR2Polls(self):
        return self.round1Winners

    def round2(self):
        self.round2Winners = self.method.winnerSet(self.ballots)

    def initWinnerLoop(self):
        self.stats = {'mean': mean, 'max': max, 'median': median}
        self.totalUtil = self.voters.socUtils[self.winner]
        self.fgHelped = {stat: [] for stat in stats}
        self.fgHarmed = {stat: [] for stat in stats}
        self.winnersFound = [(set(self.round1Winners), 0)]
        self.partialResults = self.constResults.copy()
        if set(self.round2Winners) != self(self.round1Winners):
            self.winnersFound.append((set(self.round2Winners), self.fgSize - 1))
        self.deciderMargUtilDiffs = {stat: [] for stat in stats}

    def winnerFunc(self, ballots):
        return set(self.method.winnerSet(ballots))

    def winnerLoopStats(self):
        for stat in self.stats:
            self.deciderMargUtilDiffs.append((
                    stat(self.deciderUtils[w] for w in self.newWinner)
                    - stat(self.deciderUtils[w] for w in self.prevWinner),
                    self.threshold))

    def finalStats(self):
        self.deciderMargUtilDiffs.sort(key=lambda x:x[1])
        self.deciderUtilDiffs = sorted(self.deciderUtilDiffs, key=lambda x:x[2])
        self.allUtilDiffs = sorted(self.allUtilDiffs, key=lambda x:x[1])
        self.partialResults['deciderMargUtilDiffs'] = self.deciderMargUtilDiffs

        self.allResults.append(self.makeResults(results=self.results, fgStrat = self.foregroundStrat.__name__,
            fgTargets=self.targetSelect.__name__, fgArgs=self.fgArgs,
            numWinnersFound=len(self.winnersFound),
            pivotalUtilDiff=self.deciderMargUtilDiffs[0][0] if self.deciderMargUtilDiffs else 0,
            deciderUtilDiffSum=sum(uDiff for uDiff, _ in self.deciderMargUtilDiffs), **self.partialResults))
