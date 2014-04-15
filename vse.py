
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

        
##Outer duct tape
class VseBatch:
    
    def __init__(self, model, methods, nvot, ncand, niter):
        """A harness function which creates niter elections from model and finds three kinds
        of VSE for all methods given.
        
        for instance:
        vses = VseBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
        
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
                
    def printMe(self, comparisons):
        """print the result of doVse in an accessible format.
        for instance:
        printVse(vses)
        
        """
        for i in range(len(self.methods)):
            print(self.methods[i][0])
            print([strat for strat in self.methods[i][1]], 
                  [mean([result[i][j] for result in self.vses]) 
                      for j in range(len(self.methods[0]) - 1)],
                  mean(
                       [(0 if result[i][0]==result[i][2] else 1)
                            for result in self.vses]
                       )
                  )
            
    def save(self, fn="vseresults.txt"):
        
        out = open(fn, "wb")
        head, body = self.methods, self.vses
        lines = []
        headItems = []
        for meth, choosers in head:
            mname = meth.__class__.__name__
            headItems.extend([mname + "_hon",
                             mname + "_strat"])
            for i, chooser in enumerate(choosers):
                headItems.append("_".join(mname, chooser.getName()))
                for tallyKey in chooser.allTallyKeys():
                    headItems.append("_".join(mname, str(i), tallyKey))
            headItems.append(mname + "_strat_push")
            for i, xtra in enumerate(meth[1]):
                headItems.append(mname + "_" + xtra.__name__ + str(i) + "_push")
            
                
        lines.append("\t".join([str(item) for item in headItems]) + "\n")
        
        for line in body:
            lineItems = []
            for meth in line:
                lineItems.extend(meth[:-1])
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

if __name__ == "__main__":
    import doctest
    setDebug( False)
    doctest.testmod()