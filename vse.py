
from imp import reload

from mydecorators import autoassign, cached_property, setdefaultattr
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
import csv, os


##Outer duct tape

class VseBatch:

    def __init__(self, model, methods, nvot, ncand, niter):
        """A harness function which creates niter elections from model and finds three kinds
        of VSE for all methods given.

        for instance:

        >>> vses = VseBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
        >>> [[len(y) for y in x] for x in [vses.methods, vses.vses]]
        [[2, 2], [2, 2, 2]]
        """
        vses = []
        for i in range(niter):
            electorate = model(nvot, ncand)
            vse = []
            for method, chooserFuns in methods:
                vse.append(method.vseOn(electorate, chooserFuns))
            vses.append(vse)
            debug(i,vse)
        self.methods = methods
        self.vses = vses

    def printMe(self):
        """print the result of doVse in an accessible format.
        for instance:
        vses.printMe()

        """
        for i in range(len(self.methods)):
            print(self.methods[i][0])
            print([strat.getName() for strat in self.methods[i][1]],
                  [mean([result[i].results[j].result[0] for result in self.vses])
                      for j in range(len(self.methods[i][1]) - 1)],
                  mean(
                       [(0 if result[i].results[0].result[0]==result[i].results[2].result[0] else 1)
                            for result in self.vses]
                       )
                  )

    def save(self, fn="vseresults.txt"):

        out = open(fn, "wb")
        out.writeLn()
        head, body = self.methods, self.vses
        lines = []
        headItems = []
        for meth, choosers in head:
            mname = meth.__class__.__name__
            headItems.extend([mname + "_hon",
                             mname + "_strat"])
            for i, chooser in enumerate(choosers):
                print(mname)
                print(chooser.getName())
                print("hahah")
                headItems.append("_".join([mname, chooser.getName()]))
                for tallyKey in chooser.allTallyKeys:
                    headItems.append("_".join([mname, str(i), tallyKey]))
            headItems.append(mname + "_strat_push")
            print(meth)
            #for i, xtra in enumerate(meth[1]):
            #    headItems.append(mname + "_" + xtra.__name__ + str(i) + "_push")


        lines.append("\t".join([str(item) for item in headItems]) + "\n")

        for line in body:
            lineItems = []
            for methrun in line:
                lineItems.extend(methrun.results[:-1])
                pass
            lines.append("\t".join(str(item) for item in lineItems) + "\n")

        for line in lines:
            out.write(bytes(line, 'UTF-8'))
        out.close()

    def printElection(self, e, output=print):
        for voter in e:
            output(voter[0], voter[1], voter.cluster)

    def saveNElections(self, n,model=PolyaModel(),nvot=101,ncand=2,basePath="/Users/chema/mydev/br/election"):
        f = None
        def writeToFile(*args):
            f.write(bytes("\t".join(str(arg) for arg in args) + "\n", 'UTF-8'))
        for i in range(n):
            e = model(nvot,ncand)
            f = open(basePath + str(i) + ".txt", "wb")
            self.printElection(e,writeToFile)
            f.close()



def uniquify(seq):
    # order preserving
    checked = []
    for e in seq:
       if e not in checked:
           checked.append(e)
    return checked

class CsvBatch:
    @autoassign
    def __init__(self, model, methods, nvot, ncand, niter, baseName = None, media=truth):
        """A harness function which creates niter elections from model and finds three kinds
        of utility for all methods given.

        for instance:

        >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
        >>> len(csvs.rows)
        54
        """
        rows = []
        emodel = str(model)
        for i in range(niter):
            eid = uuid4()
            electorate = model(nvot, ncand)
            for method, chooserFuns in methods:
                results = method.resultsTable(eid, emodel, ncand, electorate, chooserFuns, media=media)
                rows.extend(results)
            debug(i,results[1:3])
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
        keys = ["vse","method","chooser"] #important stuff first
        keys.extend(list(self.rows[0].keys())) #any other stuff I missed; dedup later
        for n in range(4):
            keys.extend(["tallyName"+str(n),"tallyVal"+str(n)])
        keys = uniquify(keys)
        myFile = open(baseName + str(i) + ".csv", "w")
        print("# media = " + self.media.__name__, file=myFile)
        dw = csv.DictWriter(myFile, keys, restval = "NA")
        dw.writeheader()
        for r in self.rows:
            dw.writerow(r)
        myFile.close()



medianRuns = [OssChooser(),
               OssChooser([beHon,ProbChooser([(1/2, beStrat), (1/2, beHon)])]),


               ProbChooser([(1/4, beX), (3/4, beHon)]),
               ProbChooser([(1/2, beX), (1/2, beHon)]),
               ProbChooser([(3/4, beX), (1/4, beHon)]),

               ProbChooser([(0.5, beStrat), (0.5, beHon)]),
               ProbChooser([(1/3, beStrat), (1/3, beHon), (1/3, beX)]),

               LazyChooser(),
               ProbChooser([(1/2, LazyChooser()), (1/2, beHon)]),

               ]

baseRuns = [OssChooser(),
           OssChooser([beHon,ProbChooser([(1/2, beStrat), (1/2, beHon)])]),

           ProbChooser([(1/4, beStrat), (3/4, beHon)]),
           ProbChooser([(1/2, beStrat), (1/2, beHon)]),
           ProbChooser([(3/4, beStrat), (1/4, beHon)]),

           ]

allSystems = [[Score(1000), baseRuns],
                [Score(10), baseRuns],
                [Score(2), baseRuns],
                [Score(1), baseRuns],
                [Srv(10), baseRuns],
                [Srv(2), baseRuns],
                [Plurality(), baseRuns],
                [Irv(), baseRuns],
                [Schulze(), baseRuns],
                [V321(), baseRuns],
                [Mav(), medianRuns],
                [Mj(), medianRuns]
                 ]

if __name__ == "__main__":
    import doctest
    setDebug( False)
    doctest.testmod()
