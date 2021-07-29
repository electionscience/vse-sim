from mydecorators import autoassign, cached_property, setdefaultattr, timeit
from methods import *
from voterModels import *
from dataClasses import *
from debugDump import *
import csv
import os



class CsvBatch:
    #@timeit
    #@autoassign
    def __init__(self, model, methodsAndStrats,
    nvot, ncand, niter, r1Media=truth, r2Media=truth, seed=None):
        """methodsAndStrats is a list of (votingMethod, backgroundStrat, foregrounds).
        A voting method my be given in place of such a tuple, in which case backgroundSrat and foregrounds
        will be determined automatically.
        foregrounds are (targetSelectionFunction, foregroundStrat, foregroundSelectionFunction) tuples.
        foregroundSelectionFunction is optional.
        nvot and ncand gives the number of voters and candidates in each election, and niter is how many
        electorates will be generated and have all the methods and strategies run on them.
        """
        self.rows = []
        if (seed is None):
            seed = str(niter)
            self.seed = seed
        random.seed(seed)
        ms = []
        for m in methodsAndStrats:
            if issubclass(m, Method):
                fgs = []
                for targetFunc in [select21, select31]:
                    fgs.extend([(paramStrat(m.diehardBallot, intensity=i), targetFunc) for i in m.diehardLevels]
                    + [(paramStrat(m.compBallot, intensity=i), targetFunc) for i in m.compLevels])
                    fgs.append((swapPolls(m.lowInfoBallot), targetFunc))
                for bg in [m.honBallot, m.lowInfoBallot]:
                    ms.append((m, bg, fgs))
            else:
                ms.append(m)
        for i in range(niter):
            electorate = model(nvot, ncand)
            for method, bgStrat, fgs in ms:
                result = method.threeRoundResults(electorate, bgStrat, fgs, r1Media=r1Media, r2Media=r2Media)
                self.rows.extend(result)
        for row in self.rows: row['voterModel'] = str(model)

    def saveFile(self, baseName="SimResults"):
        """print the result of doVse in an accessible format.
        for instance:

        csvs.saveFile()
        """
        i = 1
        while os.path.isfile(baseName + str(i) + ".csv"):
            i += 1
        myFile = open(baseName + str(i) + ".csv", "w")
        dw = csv.DictWriter(myFile, self.rows[0].keys(), restval="NA")
        dw.writeheader()
        for r in self.rows:
            dw.writerow(r)
        myFile.close()
