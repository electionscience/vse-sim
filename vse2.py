
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

class CsvBatch:
    #@timeit
    #@autoassign
    def __init__(self, model, methodsAndStrats,
            nvot, ncand, niter, r1Media=truth, r2Media=truth, seed=None,
            pickiness=0.4, pollingError=0.3):
        """methodsAndStrats is a list of (votingMethod, backgroundStrat, foregrounds, bgArgs).
        A voting method my be given in place of such a tuple, in which case backgroundSrat, foregrounds, and bgArgs
        will be determined automatically.
        foregrounds are (targetSelectionFunction, foregroundStrat, foregroundSelectionFunction, fgArgs) tuples.
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
                fgs = []
                for targetFunc in [select21, select31, selectRand, select012]:
                    fgs.extend([(m.diehardBallot, targetFunc, {'intensity':i}) for i in m.diehardLevels]
                    + [(m.compBallot, targetFunc, {'intensity':i}) for i in m.compLevels])
                    fgs.append((m.lowInfoBallot, targetFunc, {'info':'e'}))
                for bg in [m.honBallot, m.lowInfoBallot]:
                    ms.append((m, bg, fgs, {'pollingUncertainty':0.4}))
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

    if i>0 and i%10 == 0: print('Iteration:', i)
    if baseSeed is not None:
        random.seed(baseSeed + i)

    electorate = model(nvot, ncand)
    rows = []
    for method, bgStrat, fgs, bgArgs in ms:
        results = method.threeRoundResults(electorate, bgStrat, fgs, bgArgs=bgArgs,
                r1Media=r1Media, r2Media=r2Media, pickiness = pickiness, pollingError = pollingError)
        for result in results:
            result.update(dict(
                    seed = baseSeed + i,
                    pickiness = pickiness,
                    pollingError = pollingError,
                ))
        rows.extend(results)
    return rows
