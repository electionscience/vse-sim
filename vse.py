
from imp import reload

from mydecorators import autoassign, cached_property, setdefaultattr, timeit
import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor
from test.test_binop import isnum
from debugDump import *


from voterModels import *
from stratFunctions import *
from methods import *
from uuid import uuid4
import csv
import os
join = os.path.join


def uniquify(seq):
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked


class CsvBatch:
    @timeit
    @autoassign
    def __init__(self, model, methods, nvot, ncand, niter,
                 baseName=None, media=truth, seed=None, force=False):
        """A harness function which creates niter elections from model and finds three kinds
        of utility for all methods given.

        for instance:

        >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
        >>> len(csvs.rows)
        54
        """
        rows = []
        emodel = str(model)
        if (seed is None):
            seed = (baseName or '') + str(niter)
            self.seed = seed
        random.seed(seed)
        try:
            from git import Repo
            repo = Repo(os.getcwd())
            if not force:
                assert not repo.is_dirty()
            self.repo_version = repo.head.commit.hexsha
        except:
            self.repo_version = 'unknown repo version'
        for i in range(niter):
            eid = uuid4()
            electorate = model(nvot, ncand)
            for method, chooserFuns in methods:
                results = method.resultsTable(
                    eid, emodel, ncand, electorate, chooserFuns, media=media)
                rows.extend(results)
            debug(i, results[1:3])
        self.rows = rows
        if baseName:
            self.saveFile(baseName)

    def saveFile(self, baseName="SimResults"):
        """print the result of doVse in an accessible format.
        for instance:

        csvs.saveFile()
        """
        i = 1
        while os.path.isfile(baseName + str(i) + ".csv"):
            i += 1
        keys = ["vse", "method", "chooser"]  # important stuff first
        keys.extend(list(self.rows[0].keys()))  # any other stuff I missed; dedup later
        for n in range(4):
            keys.extend(["tallyName"+str(n), "tallyVal"+str(n)])
        keys = uniquify(keys)
        myFile = open(baseName + str(i) + ".csv", "w")
        print("# " + str(dict(media=self.media.__name__,
                              version=self.repo_version,
                              seed=self.seed,
                              model=self.model,
                              methods=self.methods,
                              nvot=self.nvot,
                              ncand=self.ncand,
                              niter=self.niter)),
              file=myFile)
        dw = csv.DictWriter(myFile, keys, restval="NA")
        dw.writeheader()
        for r in self.rows:
            dw.writerow(r)
        myFile.close()


medianRuns = [
    OssChooser([beHon, ProbChooser([(1/2, beStrat), (1/2, beHon)])]),


    ProbChooser([(1/4, beX), (3/4, beHon)]),
    ProbChooser([(1/2, beX), (1/2, beHon)]),
    ProbChooser([(3/4, beX), (1/4, beHon)]),

    ProbChooser([(0.5, beStrat), (0.5, beHon)]),
    ProbChooser([(1/3, beStrat), (1/3, beHon), (1/3, beX)]),

    LazyChooser(),
    ProbChooser([(1/2, LazyChooser()), (1/2, beHon)]),

]

baseRuns = [
    OssChooser([beHon, ProbChooser([(1/2, beStrat), (1/2, beHon)])]),

    ProbChooser([(1/4, beStrat), (3/4, beHon)]),
    ProbChooser([(1/2, beStrat), (1/2, beHon)]),
    ProbChooser([(3/4, beStrat), (1/4, beHon)]),

]

allSystems = [[Score(1000), baseRuns],
              [Score(10), baseRuns],
              [Score(2), baseRuns],
              [Score(1), baseRuns],
              [BulletyApprovalWith(.6), baseRuns],
              [STAR(10), baseRuns],
              [STAR(2), baseRuns],
              [Plurality(), baseRuns],
              [Borda(), baseRuns],
              [Irv(), baseRuns],
              [Schulze(), baseRuns],
              [Rp(), baseRuns],
              [V321(), baseRuns],
              [Mav(), medianRuns],
              [Mj(), medianRuns],
              [IRNR(), baseRuns],
              ]

# request from Mark: "STAR0-2, STAR0-3, STAR0-4, STAR0-5, STAR0-6, STAR0-7, STAR0-8, STAR0-9, STAR0-10, Score0-10, 321, Approval, IRV and plurality"
markMethods = [
    [STAR(2), baseRuns],
    [STAR(3), baseRuns],
    [STAR(4), baseRuns],
    [STAR(5), baseRuns],
    [STAR(6), baseRuns],
    [STAR(7), baseRuns],
    [STAR(8), baseRuns],
    [STAR(9), baseRuns],
    [Score(10), baseRuns],
    [V321(), baseRuns],
    [BulletyApprovalWith(.6), baseRuns],
    [Irv(), baseRuns],
    [Plurality(), baseRuns],
]

# usage example:
# >>> from vse import *
# >>> vses = CsvBatch(KSModel(dcdecay=(1,3),wcdecay=(1.5,3), dccut = .2, wcalpha=1.5),
#           allSystems, nvot=40, ncand=6, niter=15000, baseName="target",
#           media=fuzzyMediaFor())

if __name__ == "__main__":
    import doctest
    setDebug(False)
    doctest.testmod()
