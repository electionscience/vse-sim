
from mydecorators import autoassign, cached_property, setdefaultattr
from voterModels import *
import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor
from numpy import std
from test.test_binop import isnum
from debugDump import *
from dataClasses import *




###################Choosers
class Chooser:
    tallyKeys = []

    @autoassign
    def __init__(self, choice, subChoosers=[]):
        """Subclasses should just copy/paste this logic because
        each will have its own parameters and so that's easiest."""
        pass

    def getName(self):
        if hasattr(self, "choice"): #only true for base class
            #print("base")
            return self.choice
        if not hasattr(self, "name") or not self.name:
            #print("generic")
            self.name = self.__class__.__name__[:-7] #drop the "Chooser"
        #print("specific")
        return self.name

    def __call__(self, cls, voter, tally):
        return self.choice

    def addTallyKeys(self, tally):
        for key in self.allTallyKeys:
            tally[key] = 0

    @cached_property
    def myKeys(self):
        prefix = self.getName() + "_"
        return [prefix + key for key in self.tallyKeys]

    @cached_property
    def allTallyKeys(self):
        keys = self.myKeys
        for subChooser in self.subChoosers:
            keys += subChooser.allTallyKeys
        return keys

    @cached_property
    def __name__(self):
        return self.__class__.__name__

beHon = Chooser("hon")
beStrat = Chooser("strat")
beX = Chooser("extraStrat")

class LazyChooser(Chooser):
    """Honest, if honest and strategic are the same. Otherwise, extra-strategic."""

    tallyKeys = [""]
    @autoassign
    def __init__(self, subChoosers=[beHon, beX]):
        pass


    def __call__(self, cls, voter, tally):
        if (getattr(voter, cls.__name__ + "_hon") ==
            getattr(voter, cls.__name__ + "_strat")):
            tally[self.myKeys[0]] += 0
            return self.subChoosers[0](cls, voter, tally) #hon
        tally[self.myKeys[0]] += 1
        return self.subChoosers[1](cls, voter, tally) #strat

class OssChooser(Chooser):
    tallyKeys = [""]
    """one-sided strategy:
    returns a 'strategic' ballot for those who prefer the honest runner-up,
    and an honest ballot for those who prefer the honest winner. Only works
    if honBallot and stratBallot have already been called for the voter.


    """
    @autoassign
    def __init__(self, subChoosers = [beHon, beStrat]):
        pass

    def __call__(self, cls, voter, tally):
        hon, strat = self.subChoosers
        if getattr(voter, cls.__name__ + "_isStrat", False):
            tally[self.myKeys[0]] += 1
            if callable(strat):
                #debug(strat)
                return strat(cls, voter, tally)
            return strat
        else:
            if callable(hon):
                #debug(hon)
                return hon(cls, voter, tally)
            return hon

    def getName(self):
        baseName = super(OssChooser, self).getName()
        return baseName + "." + "_".join(s.getName() for s in self.subChoosers) + "."

class ProbChooser(Chooser):
    @autoassign
    def __init__(self, probs):
        self.subChoosers = [chooser for (p, chooser) in probs]

    def __call__(self, cls, voter, tally):
        r = random.random()
        for (i, (p, chooser)) in enumerate(self.probs):
            r -= p
            if r < 0:
                if i > 0: #keep tally for all but first option
                    tally[self.getName() + "_" + chooser.getName()] += 1
                return chooser(cls, voter, tally)

    def getName(self):
        baseName = super(ProbChooser, self).getName()
        return baseName + "." + "_".join(s.getName() + str(round(p * 100)) for p,s in self.probs) + "."




###media

def truth(standings, tally=None):
    return standings

def topNMediaFor(n):
    def topNMedia(standings, tally=None):
        return list(standings[:n]) + [min(standings)] * (len(standings) - n)
    return topNMedia

def biaserAround(scale):
    def biaser(standings):
        return scale * std(standings,ddof=1)
    return biaser

def orderOf(standings):
    return [i for i,val in sorted(list(enumerate(standings)), key=lambda x:x[1], reverse=True)]

def fuzzyMediaFor(biaser = biaserAround(1)):
    def fuzzyMedia(standings, tally=None):
        if not tally:
            tally=SideTally()
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= [s + random.gauss(0,bias) for s in standings]
        tally["changed"] += 0 if orderOf(result)[0:2] == orderOf(standings)[0:2] else 1
        return result
    return fuzzyMedia

def biasedMediaFor(biaser=biaserAround(1),numerator=1):
    """
    if numerator is 1:
    0, 0, -1/2, -2/3, -3/4....
    if numerator is 1.5:
        0,0,-.25, -.5, -.625, -.7
    numerator shouldn't be over 2 unless you want strangeness.


    """
    def biasedMedia(standings, tally=None):
        if not tally:
            tally=SideTally()
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= (standings[0:2] +
                 [(standing - bias + numerator * (bias / max(i+2, 1)))
                        for i, standing in enumerate(standings[2:])])
        tally["changed"] += 0 if orderOf(result)[0:2] == orderOf(standings)[0:2] else 1
        return result
    return biasedMedia

def skewedMediaFor(biaser):
    """

    [0, -1/3, -2/3, -1]
    """
    def skewedMedia(standings, tally=None):
        if not tally:
            tally=SideTally()
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= [(standing - bias * i / (len(standings) - 1)) for i, standing in enumerate(standings)]

        tally["changed"] += 0 if orderOf(result)[0:2] == orderOf(standings)[0:2] else 1
        return result
    return skewedMedia
