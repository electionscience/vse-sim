from mydecorators import autoassign, cached_property, setdefaultattr

import random
from numpy.lib.scimath import sqrt
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
from numpy.ma.core import floor
from scipy.stats import beta
from test.test_binop import isnum
from uuid import uuid4


from debugDump import *

class Voter(tuple):
    """A tuple of candidate utilities.



    """
    @cached_property
    def id(self):
        """Get a repeatable uuid
        """
        return uuid4()
        
    @classmethod
    def rand(cls, ncand):
        """Create a random voter with an independent standard normal
        utility for each candidate.

        ncand determines the number of candidates a voter should have
        utilities for.
            >>> [len(Voter.rand(i)) for i in list(range(5))]
            [0, 1, 2, 3, 4]

        utilities should be in a standard normal distribution
            >>> v100 = Voter.rand(100)
            >>> -0.3 < mean(v100) < 0.3
            True
            >>> 0.8 < std(v100) < 1.2
            True
        """
        return cls(random.gauss(0,1) for i in range(ncand))


    def hybridWith(self, v2, w2):
        """Create a weighted average of two voters.

        The weight of v1 is always 1; w2 is the weight of v2 relative to that.

        If both are
        standard normal to start with, the result will be standard normal too.

        Length must be the same
            >>> Voter([1,2]).hybridWith(Voter([1,2,3]),1)
            Traceback (most recent call last):
              ...
            AssertionError

        A couple of basic sanity checks:
            >>> v2 = Voter([1,2]).hybridWith(Voter([3,2]),1)
            >>> [round(u,5) for u in v2.hybridWith(v2,1)]
            [4.0, 4.0]
            >>> Voter([1,2,5]).hybridWith(Voter([-0.5,-1,0]),0.75)
            (0.5, 1.0, 4.0)
        """
        assert len(self) == len(v2)
        return self.copyWithUtils(  ((self[i] / sqrt(1 + w2 ** 2)) +
                                    (w2 * v2[i] / sqrt(1 + w2 ** 2)))
                                 for i in range(len(self)))

    def copyWithUtils(self, utils):
        """create a new voter with attrs as self and given utils.

        This version is a stub, since this voter class has no attrs."""
        return self.__class__(utils)

    def mutantChild(self, muteWeight):
        """Returns a copy hybridized with a random voter of weight muteWeight.

        Should remain standard normal:
            >>> v100 = Voter.rand(100)
            >>> for i in range(30):
            ...     v100 = v100.mutantChild(random.random())
            ...
            >>> -0.3 < mean(v100) < 0.3 #3 sigma
            True
            >>> 0.8 < std(v100) < 1.2 #meh that's roughly 3 sigma
            True

        """
        return self.hybridWith(self.__class__.rand(len(self)), muteWeight)

class PersonalityVoter(Voter):

    cluster_count = 0

    def __init__(self, *args, **kw):
        super().__init__()#*args, **kw) #WTF, python?
        self.cluster = self.__class__.cluster_count
        self.__class__.cluster_count += 1
        self.personality = random.gauss(0,1) #probably to be used for strategic propensity
        #but in future, could be other clustering voter variability, such as media awareness

    #@classmethod
    #def rand(cls, ncand):
    #    voter = super().rand(ncand)
    #    return voter

    @classmethod
    def resetClusters(cls):
        cls.cluster_count = 0

    def copyWithUtils(self, utils):
        voter = super().copyWithUtils(utils)
        voter.copyAttrsFrom(self)
        return voter

    def copyAttrsFrom(self, model):
        self.personality = model.personality
        self.cluster = model.cluster

class Electorate(list):
    """A list of voters.
    Each voter is a list of candidate utilities"""

    @cached_property
    def id(self):
        """Get a repeatable uuid
        """
        return uuid4()

    @cached_property
    def socUtils(self):
        """Return mean utility across electorate for each candidate: their social utilities.

        >>> e = Electorate([[1,2],[3,4]])
        >>> e.socUtils
        [2.0, 3.0]
        """
        return list(map(mean,zip(*self)))


class RandomModel:
    """Empty base class for election models; that is, electorate factories.

    >>> e4 = RandomModel()(4,3)
    >>> [len(v) for v in e4]
    [3, 3, 3, 3]
    """

    def __str__(self):
        return self.__class__.__name__
    def __call__(self, nvot, ncand, vType=PersonalityVoter):
        return Electorate(vType.rand(ncand) for i in range(nvot))

class DeterministicModel(RandomModel):
    """Basically, a somewhat non-boring stub for testing.

        >>> DeterministicModel(3)(4, 3)
        [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 1, 2)]
    """

    @autoassign
    def __init__(self, modulo):
        pass

    def __call__(self, nvot, ncand, vType=PersonalityVoter):
        return Electorate(vType((i+j)%self.modulo for i in range(ncand))
                          for j in range(nvot))

class ReverseModel(RandomModel):
    """Creates an even number of voters in two diametrically-opposed camps
    (ie, opposite utilities for all candidates)

    >>> e4 = ReverseModel()(4,3)
    >>> [len(v) for v in e4]
    [3, 3, 3, 3]
    >>> e4[0].hybridWith(e4[3],1)
    (0.0, 0.0, 0.0)
    """
    def __call__(self, nvot, ncand, vType=PersonalityVoter):
        if nvot % 2:
            raise ValueError
        basevoter = vType.rand(ncand)
        return Electorate( ([basevoter] * (nvot//2)) +
                           ([vType(-q for q in basevoter)] * (nvot//2))
                        )

class QModel(RandomModel):
    """Adds a quality dimension to a base model,
    by generating an election and then hybridizing all voters
    with a common quality vector.

    Useful along with ReverseModel to create a poor-man's 2d model.

    Basic structure
        >>> e4 = QModel(sqrt(3), RandomModel())(100,1)
        >>> len(e4)
        100
        >>> len(e4.socUtils)
        1

    Reduces the standard deviation
        >>> 0.4 < std(list(zip(e4))) < 0.6
        True

    """
    @autoassign
    def __init__(self, qWeight=0.5, baseModel=ReverseModel()):
        pass

    def __call__(self, nvot, ncand, vType=PersonalityVoter):
        qualities = vType.rand(ncand)
        return Electorate([v.hybridWith(qualities,self.qWeight)
                for v in self.baseModel(nvot, ncand, vType)])

class PolyaModel(RandomModel):
    """This creates electorates based on a Polya/Hoppe/Dirichlet model, with mutation.
    You start with an "urn" of n=seedVoter voters from seedModel,
     plus alpha "wildcard" voters. Then you draw a voter from the urn,
     clone and mutate them, and put the original and clone back into the urn.
     If you draw a "wildcard", use voterGen to make a new voter.
     """
    @autoassign
    def __init__(self, seedVoters=2, alpha=1, seedModel=QModel(),
                 mutantFactor=0.2):
        pass

    def __call__(self, nvot, ncand, vType=PersonalityVoter):
        """Tests? Making statistical tests that would pass reliably is
        a huge hassle. Sorry, maybe later.
        """
        vType.resetClusters()
        election = self.seedModel(self.seedVoters, ncand, vType)
        while len(election) < nvot:
            i = random.randrange(len(election) + self.alpha)
            if i < len(election):
                election.append(election[i].mutantChild(self.mutantFactor))
            else:
                election.append(vType.rand(ncand))
        return election

class DimVoter(PersonalityVoter):
    """A voter in an n-dimensional model.


     """

    @classmethod
    def fromDims(cls, v, e, caring = None):
        if caring==None:
            caring = [1] * len(v)
            totCaring = e.totWeight
        else:
            totCaring = sum((c*w)**2 for c,w in zip(caring, e.dimWeights))
        me = cls(-sqrt(
            sum(((vd - cd)*w*cares)**2 for (vd, cd, w, cares) in zip(v,c,e.dimWeights,caring)) /
                            totCaring)
          for c in e.cands)
        me.copyAttrsFrom(v)
        me.dims = v
        me.elec = e
        return me


class DimElectorate(Electorate):

    def asDims(self, v, *args):
        return v

    def fromDims(self, dimvoters, vType):
        for v in dimvoters:
            self.append(vType.fromDims(v,self))

    def calcTotWeight(self):
        self.totWeight = sum([w**2 for w in self.dimWeights])

class DimModel(RandomModel):
    """

    >>> dm = DimModel(2,baseElectorate=DeterministicModel(3))
    >>> dm(2,4)
    [(4.25, 0.0, 1.25, 4.25), (2.0, 1.25, 0.0, 2.0)]
    >>> dm.dimWeights
    [1, 0.5]


    """
    builtElectorate = DimElectorate

    @autoassign
    def __init__(self, ndims=3, dimWeights=None, baseElectorate=RandomModel()):
        if self.dimWeights is None:
            self.dimWeights = [2**(-n) for n in range(ndims)]
        assert(len(self.dimWeights) == self.ndims)

    def __call__(self, nvot, ncand, vType=DimVoter):
        elec = self.builtElectorate()
        elec.dimWeights = self.dimWeights
        return self.makeElectorate(elec, nvot, ncand, vType)

    def makeElectorate(self, elec, nvot, ncand, vType):
        elec.calcTotWeight()
        votersncands = self.baseElectorate(nvot + ncand, len(elec.dimWeights), vType)
        elec.base = [elec.asDims(v,i) for i,v in enumerate(votersncands[:nvot])]
        elec.cands = [elec.asDims(v,nvot+i) for i,v in enumerate(votersncands[nvot:])]
        elec.fromDims(elec.base, vType)
        return elec

def rbeta(a,b):
    return lambda: beta.rvs(a,b)

unishdist = rbeta(1,.8)

caresDist = rbeta(3,1.5)

class KSElectorate(DimElectorate):

    def chooseClusters(self, n, alpha, caring):
        """
        Sets up the crosscat structure.
        args:
            n: number of voters to create
            alpha: global alpha for subcluster dirichlet processes

        preconditions:
            .numSubclusters is a len-.numViews array of zeros


        Side-effects include setting the following attributes on self:
            .views: a
        """
        self.views = []
        for i in range(n):
            item = []
            for c in range(self.numViews):
                r = (i+alpha) * random.random()
                if r > i:
                    item.append(self.numSubclusters[c])
                    self.numSubclusters[c] += 1
                else:
                    item.append(self.views[int(r)][c])
            self.views.append(item)
        self.clusterMeans = []
        self.clusterCaring = []
        for c in range(self.numViews):
            subclusterMeans = []
            subclusterCaring = []
            for i in range(self.numSubclusters[c]):
                cares = caring()

                subclusterMeans.append([random.gauss(0,sqrt(cares)) for i in range(self.dimsPerView[c])])
                subclusterCaring.append(caring())
            self.clusterMeans.append(subclusterMeans)
            self.clusterCaring.append(subclusterCaring)

    def asDims(self, v, i):
        result = []
        dim = 0
        cares = []
        for c in range(self.numViews):
            clusterMean = self.clusterMeans[c][self.views[i][c]]
            for m in clusterMean:
                acare = self.clusterCaring[c][self.views[i][c]]
                result.append(m + (v[dim] * sqrt(1-acare)))
                cares.append(acare)
            dim += 1
        v = PersonalityVoter(result) #TODO: do personality right
        v.cares = cares
        return v

    def fromDims(self, dimvoters, vType):
        for v in dimvoters:
            self.append(vType.fromDims(v,self,v.cares))

class KSModel(DimModel): #Kitchen sink

    builtElectorate = KSElectorate
    baseElectorate = RandomModel()

    @autoassign
    #dc = dimensional cluster; vc = voter cluster
    def __init__(self, dcdecay=(1,1), dccut = .2,
            wcdecay=(1,1), wccut = .2,
            wcalpha=1, vccaring=(6,3)):
        pass

    def __str__(self):
        return "_".join(str(x) for x in (self.__class__.__name__,self.wcalpha) + self.dcdecay + self.wcdecay + self.vccaring)

    def __call__(self, nvot, ncand, vType=DimVoter):
        """Create an electorate.
        Args:
            nvot: number of voters
            ncand: number of cands
            vType: type of voter. Defaults to DimVoter.

        Directly responsible for side-effects:
            Choose V (# views) and D_v (# dims per view)
            Use two-level stick-breaking to assign non-normalized (!!!) dimension weights to each dimension.
            len(e.dimsPerView) = e.numViews
            len(e.dimWeights) = sum(e.dimsPerView) = num dims

            e.numSubclusters: init to array of e.numViews zeros; let chooseClusters go from there


        TODO: Tests? Making statistical tests that would pass reliably is
        a huge hassle. Sorry, maybe later.
        """
        vType.resetClusters()
        e = self.builtElectorate()
        e.dimsPerView = [] #number of dimensions in each view
        e.dimWeights = [] #raw importance of each dimension, regardless of view
        viewWeight = 1
        while viewWeight > self.dccut:
            dimweight = viewWeight
            dimnum = 0
            while dimweight > self.wccut:
                e.dimWeights.append(dimweight)
                dimnum += 1
                dimweight *= beta.rvs(*self.wcdecay)
            e.dimsPerView.append(dimnum)
            viewWeight *= beta.rvs(*self.dcdecay)
        e.numViews = len(e.dimsPerView)
        e.numSubclusters = [0] * e.numViews
        e.chooseClusters(nvot + ncand, self.wcalpha, lambda:beta.rvs(*self.vccaring))
        return self.makeElectorate(e, nvot, ncand, vType)
