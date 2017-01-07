import inspect
import functools
import numpy as np
import random

#from stackexchange...
def autoargs(*include,**kwargs):   
    def _autoargs(func):
        attrs,varargs,varkw,defaults=inspect.getargspec(func)
        def sieve(attr):
            if kwargs and attr in kwargs['exclude']: return False
            if not include or attr in include: return True
            else: return False            
        @functools.wraps(func)
        def wrapper(self,*args,**kwargs):
            # handle default values
            if defaults:
                for attr,val in zip(reversed(attrs),reversed(defaults)):
                    if sieve(attr): setattr(self, attr, val)
            # handle positional arguments
            positional_attrs=attrs[1:]            
            for attr,val in zip(positional_attrs,args):
                if sieve(attr): setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args=args[len(positional_attrs):]
                if sieve(varargs): setattr(self, varargs, remaining_args)                
            # handle varkw
            if kwargs:
                for attr,val in list(kwargs.items()):
                    if sieve(attr): setattr(self,attr,val)            
            return func(self,*args,**kwargs)
        return wrapper
    return _autoargs

#Hah: http://www.pydanny.com/cached-property.html
class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value






##actual code
DEBUG = True
arrayType = type(np.array([1]))

class ElectionCounts():
    
    @autoargs()
    def __init__(self, delg, appr, prefs, order, cantWin = [], oldSmith = None):
        """
        delg: A list of n delegation counts
        appr: A list of n approval counts
        prefs: A list of n preference lists counts
        order: delegation order
        """
        self.n = len(delg)
        if type(self.appr) != arrayType:
            self.appr = np.matrix(self.appr)
        if DEBUG:
            n = self.n
            assert(self.appr.size==n)
            assert(len(prefs)==n)
            for pref in prefs:
                assert(len(pref) == n)
                for i in range(n):
                    assert(i in pref)
            noDelg = list(range(n))
            for i in order:
                noDelg.remove(i)
            for i in noDelg:
                assert(delg[i] == 0)
                
    def __repr__(self):
        return ("ElectionCounts(%s,%s,%s,%s)"%
               (self.delg, self.appr.tolist()[0], self.prefs, self.order))
    
    def oneMatrix(self, pref, size=1):
        n = self.n
        mat = np.tril(np.ones((n,n)),-1) * size
        inverse_pref = np.array(pref)
        inverse_pref[pref] = range(n)
        for i in range(n):
            mat[:,i] = mat[inverse_pref,i]
        for i in range(n):
            mat[i,:] = mat[i,inverse_pref]
        return mat
    
    def appMatrix(self, appr = None):
        n = self.n
        if appr is None:
            appr = self.appr
        return np.matrix(np.ones((n,1))) * appr
            
    @cached_property        
    def matrix(self):
        mat = self.appMatrix()
        for i in self.order:
            mat += self.oneMatrix(self.prefs[i], self.delg[i])
        return mat
    
    def beaters(self, loser, candidates, minwin = [None], rival = [None], private = False):
        """a generator which, using the matrix m, gives any members of candidates who loser doesn't majority beat.
        
        NOTE: THIS MODIFIES candidates AS A SIDE-EFFECT, AND NOTICES IF IT"S MODIFIED BY OTHERS.
        Also modifies `by` as a side effect
        """
        m = self.matrix
        best = np.argmax(m[loser])
        if private:
            outer = candidates
            candidates = list(candidates) #local copy
        while len(candidates):
            c = candidates.pop(0)
            if private and (c not in outer):
                continue
            if (m[best,loser] > m[c,loser]) and (m[best,c] > m[loser,c]):
                #print("a",c,loser)
                if rival[0] is not None:
                    toWin = max((m[best,loser] - m[c,loser]), (m[best,c] - m[loser,c]))
                    if rival[0][0] < toWin:
                        rival[0] = (toWin,best,loser,c)
                        
                if private:
                    outer.remove(c)
                yield c
            else:
                if m[loser,c] >= m[c,loser]:
                    #print("b",c,loser)
                    if minwin[0]:
                        if minwin[0][0] > m[loser,c]:
                            minwin[0] = (m[loser,c],loser,c)
                    if private:
                        outer.remove(c)
                    yield c
                    
    def oneWinner(self, m):
        start = np.argmax(m[0])
        theRest = list(range(self.n))
        theRest.remove(start)
        #print(theRest)
        return self.climbFrom(start, theRest)
    
    def climbFrom(self, start, theRest):
        """Find the first leaf in a depth-first search up through theRest starting from start.
        
        Watch out! Modifies theRest as side effect!"""
        for c in self.beaters(start, theRest):
            return self.climbFrom(c, theRest)
        #Nobody beats start, so just return.
        return start
    
    @cached_property  
    def majSmith(self):
        m = self.matrix
        winners = [self.oneWinner(m)]
        remaining = list(range(self.n))
        remaining.remove(winners[0])
        minWin = [(1e6,)]
        rival = [(0,)]
        self.growFrom(winners[0], winners, remaining, minWin, rival)
        self.minWin = minWin[0]
        self.rival = rival[0]
        return winners
        
    def growFrom(self, seed, plant, soil, minwin = [None], rival = [None]):
        #print(seed, plant, soil)
        """as a SIDE-EFFECT, recursively fill out the set of winners, starting from seed"""
        for w in self.beaters(seed,soil, minwin, rival, private=True):
            #print(w,"grows on",seed)
            plant.append(w)
            self.growFrom(w, plant, soil, minwin, rival)
        
    def delegated(self, amounts, cantWin=None):
        delegator = self.order[0]
        appr = np.matrix(np.zeros(self.n))
        dprefs = self.prefs[delegator]
        #print(dprefs)
        appr[:,dprefs] = amounts
        if DEBUG:
            for i in range(self.n-1):
                assert appr[0,dprefs[i]] >= appr[0,dprefs[i+1]],"bullshit %i %i %s ... %s" % (appr[0,dprefs[i]],appr[0,dprefs[i+1]],appr,dprefs)
            
        #print(appr)
        delg = list(self.delg)
        delg[delegator] = 0
        result = ElectionCounts(delg,appr + self.appr,self.prefs,self.order[1:],
                                cantWin or self.cantWin, self.majSmith)
        result.matrix = self.matrix - self.oneMatrix(dprefs, self.delg[delegator]) + self.appMatrix(appr)
        return result
        
        
    def winner(self, verbose = 0):
        if not len(self.order): #delegation tree leaf
            #print(self.matrix)
            if verbose > 2:
                print("leafed out", self.matrix)
            return np.argmax(self.matrix[0])
        smith = self.majSmith
        if len(smith) <= 1: #Clear winner, not worth finishing
            #print(self.matrix)
            if verbose > 2:
                print("crystal ball", smith[0], self.matrix)
            return smith[0]
        if self.oldSmith:
            if verbose and len(smith) > len(self.oldSmith): #Check if smith set has grown. Surprising, but not decisive
                print("Smith set expanded!")#,self.oldSmith, smith, self.matrix)
                #print("old", self.oldSmith)
                #print("new", smith, self.matrix)
        
        if self.cantWin:
            badWinners = True
            for possibility in smith:
                badWinners = badWinners and (possibility in self.cantWin)
            if badWinners:
                #print("badwinners", smith, self.cantWin)
                if verbose > 2:
                    print("giving up", self.matrix)
                return None #This is a shortcut. We don't know that this cand will win, but it will be ignored anyway.
            
        #figure out reasonable bounds for whom to approve, who might win.
        idealWinnerIndex = bestHopeIndex = self.n
        bestHope = None
        worstWinnerIndex = 0
        curPrefs = self.prefs[self.order[0]]
        for w in smith:
            i = curPrefs.index(w)
            if i > worstWinnerIndex:
                worstWinnerIndex = i
            if i < idealWinnerIndex:
                idealWinnerIndex = i
        idealWinner = curPrefs[idealWinnerIndex]
                
        cantWin = self.cantWin or set()
        
        #print("looping",len(self.order))
        for amounts in self.possibleDelegations(worstWinnerIndex, idealWinnerIndex):
            #print(".")
            #print(self.delegated(np.array([10,10,0,0,0])))
            dec =  self.delegated(amounts,cantWin)
            w = dec.winner(verbose)
            if verbose and len(self.order) > 2:
                print(w,len(self.order),"amounts",amounts,bestHope, bestHopeIndex,"and",worstWinnerIndex, idealWinnerIndex,"with",np.trace(dec.matrix))
            #print("    " * (5 - len(self.order)), "winner?", w, bestHope, curPrefs )
            if w == idealWinner:
                if verbose > 1.5:
                    print("love it", w, dec.matrix)
                return(w)
            if w is None:
                #print("nothing for",amounts)
                continue
            i = curPrefs.index(w)
            #if len(self.order) == 3: #print(i)
            if i < bestHopeIndex:
                if verbose > 2-len(self.order)*1.0/10:
                    print("updating w,len(self.order),amounts",w,i,len(self.order),amounts,curPrefs,bestHopeIndex)
                    #print(,amounts)
                    print()
                bestHopeIndex = i
                bestHope = w
                for l in range(i+1,self.n):
                    cantWin.add(curPrefs[l])
        return bestHope
    
    def possibleDelegations(self, worstWinnerIndex, idealWinnerIndex):
        #first, full thresholding
        curPrefs = self.prefs[self.order[0]]
        size = self.delg[self.order[0]]
        delegations = np.zeros(self.n)
        for i in range(max(1,idealWinnerIndex + 1)):
            delegations[i] = size
        dcopy = np.array(delegations)
        #print("hi",i,worstWinnerIndex + 1)
        for i in range(i,worstWinnerIndex + 1):
            delegations[i] = size
            yield np.array(delegations)
            #print("there")
        #print("you")
            
        #Now, try to be clever
        if self.minWin:
            for i in range(idealWinnerIndex + 1, worstWinnerIndex): #if i==3 and idealWinnerIndex==1 we want [max, max, mid, mid, 0, 0]
                delegations = np.array(dcopy)
                needed = max(0,
                             min(size,
                                 self.minWin[0] - (max(self.matrix[:,curPrefs[i]][range(i+1,worstWinnerIndex+1),:])[0,0] - size) + 0.1))
                for j in range(idealWinnerIndex + 1, i + 1):
                    delegations[j] = needed
                yield np.array(delegations)
                    
                
        #self.appMatrixcurPrefs[idealWinnerIndex]
            
    
    def scores(self):
        scores = np.zeros(self.n)
        for i in range(len(self.delg)):
            for j in range(self.n):
                scores[self.prefs[i][j]] = scores[self.prefs[i][j]] + self.delg[i] * (self.n-j-1) 
        for j in range(self.n):
            scores[j] = scores[j] + self.appr.tolist()[0][j] * self.n-1
        return scores
            
            
            
    
myEc = ElectionCounts([4,3,2,0],[0,0,0,1],[[0,1,2,3],[1,2,0,3],[2,0,1,3],[3,2,1,0]],[0,1,2,3])

myEc2 = ElectionCounts([5,30,20,0],[35,0,0,1],[[0,1,2,3],[1,2,0,3],[2,1,0,3],[3,2,1,0]],[0,1,2,3])



myEc3 = ElectionCounts([4,3,2,0],[0,0,0,0],[[0,1,2,3],[2,3,0,1],[3,0,1,2],[3,2,1,0]],[0,1,2,3])

def shuffled(n):
    l = list(range(n))
    random.shuffle(l)
    return l

def randomElection(ncand):
    ec = ElectionCounts([random.randrange(4,20,3) for _ in range(ncand)],[round(random.random(),3)*10 for _ in range(ncand)],
                        [shuffled(ncand) for _ in range(ncand)],list(range(ncand)))
    return ec

def monteCarlo(n):
    funky = []
    for i in range(n):
        if i % 50 == 0:
            print("tick",i)
        re = randomElection(4 + random.randrange(4))
        w = re.winner()
        if w not in re.majSmith:
            print("Unsmith!!!",i)
            funky.append(re)
            print(re.delg,re.appr)  
            print(re.prefs)
            print(re.matrix)
            print(w,re.majSmith)
            print("funny, huh?")
    return funky
     
     