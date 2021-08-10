
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






###media

def truth(standings, *args):
    return standings

def noisyMedia(standings, marginOfError):
    return [min(1, max(0, random.gauss(s,marginOfError/2))) for s in standings]

def topNMediaFor(n):
    def topNMedia(standings):
        return list(standings[:n]) + [min(standings)] * (len(standings) - n)
    return topNMedia

def biaserAround(scale):
    def biaser(standings):
        return scale * std(standings,ddof=1)
    return biaser

def orderOf(standings):
    return [i for i,val in sorted(list(enumerate(standings)), key=lambda x:x[1], reverse=True)]

def fuzzyMediaFor(biaser = biaserAround(1)):
    def fuzzyMedia(standings):
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= [s + random.gauss(0,bias) for s in standings]
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
    def biasedMedia(standings):
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= (standings[0:2] +
                 [(standing - bias + numerator * (bias / max(i+2, 1)))
                        for i, standing in enumerate(standings[2:])])
        return result
    return biasedMedia

def skewedMediaFor(biaser):
    """

    [0, -1/3, -2/3, -1]
    """
    def skewedMedia(standings):
        if callable(biaser):
            bias = biaser(standings)
        else:
            bias = biaser
        result= [(standing - bias * i / (len(standings) - 1)) for i, standing in enumerate(standings)]
        return result
    return skewedMedia
