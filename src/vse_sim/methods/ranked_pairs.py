from ..voter_models import DeterministicModel  # noqa: F401
from .schulze import Schulze


class Rp(Schulze):
    def resolveCycle(self, cmat, n):
        """Note: mutates cmat destructively.

        >>> Rp().resultsFor(DeterministicModel(3)(5,3),Rp().honBallot,isHonest=True)["results"]
        [1, 2, 0]
        """
        matches = [(i, j, cmat[i][j]) for i in range(n) for j in range(i,n) if i != j]
        rps = sorted(matches,key=lambda x:-abs(x[2]))
        for (i, j, margin) in rps:
            if margin < 0:
                i, j = j, i
            if cmat[j][i] is not True:
                cmat[i][j] = True
                for k in range(n):
                    if k not in (i, j):
                        if cmat[j][k] is True:
                            cmat[i][k] = True
                        if cmat[k][i] is True:
                            cmat[k][j] = True

        return [sum(cmat[i][j] is True for j in range(n)) for i in range(n)]
