
import csv
import hashlib
import os
import random
from uuid import uuid4

import numpy as np

from .decorators import autoassign, timeit
from .diagnostics import debug, setDebug
from .methods import (
    IRNR,
    V321,
    Borda,
    BulletyApprovalWith,
    Irv,
    IrvPrime,
    Mav,
    Mj,
    Plurality,
    Rp,
    Schulze,
    Score,
    Srv,
)
from .strategies import (
    Chooser,
    LazyChooser,
    OssChooser,
    ProbChooser,
    beHon,
    beStrat,
    beX,
    biasedMediaFor,
    biaserAround,
    fuzzyMediaFor,
    orderOf,
    skewedMediaFor,
    topNMediaFor,
    truth,
)
from .voter_models import (
    DeterministicModel,
    DimElectorate,
    DimModel,
    DimVoter,
    Electorate,
    KSElectorate,
    KSModel,
    PersonalityVoter,
    PolyaModel,
    QModel,
    RandomModel,
    ReverseModel,
    Voter,
    rbeta,
)

join = os.path.join

__all__ = [
    "Borda",
    "BulletyApprovalWith",
    "Chooser",
    "CsvBatch",
    "DeterministicModel",
    "DimElectorate",
    "DimModel",
    "DimVoter",
    "Electorate",
    "IRNR",
    "Irv",
    "IrvPrime",
    "KSElectorate",
    "KSModel",
    "LazyChooser",
    "Mav",
    "Mj",
    "OssChooser",
    "PersonalityVoter",
    "Plurality",
    "PolyaModel",
    "ProbChooser",
    "QModel",
    "RandomModel",
    "ReverseModel",
    "Rp",
    "Schulze",
    "Score",
    "Srv",
    "V321",
    "Voter",
    "allSystems",
    "baseRuns",
    "beHon",
    "beStrat",
    "beX",
    "biasedMediaFor",
    "biaserAround",
    "debug",
    "fuzzyMediaFor",
    "markMethods",
    "medianRuns",
    "orderOf",
    "rbeta",
    "seedRandomGenerators",
    "setDebug",
    "skewedMediaFor",
    "topNMediaFor",
    "truth",
    "uniquify",
]




def uniquify(seq):
    # order preserving
    checked = []
    for e in seq:
       if e not in checked:
           checked.append(e)
    return checked


def seedRandomGenerators(seed):
    """Seed the Python and NumPy global generators deterministically."""
    random.seed(seed)
    numpy_seed = int.from_bytes(
        hashlib.sha256(str(seed).encode()).digest()[:4], byteorder="little"
    )
    np.random.seed(numpy_seed)


class CsvBatch:
    @timeit
    @autoassign
    def __init__(self, model, methods, nvot, ncand, niter,
            baseName = None, media=truth, seed=None, force=False,
            retain_rows=True):
        """A harness function which creates niter elections from model and finds three kinds
        of utility for all methods given.

        for instance:

        >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3) # doctest: +ELLIPSIS
        >>> len(csvs.rows)
        60

        ``force=True`` permits provenance collection from a dirty Git working
        tree. It does not control output-file replacement; ``saveFile`` always
        chooses the next available numbered filename.
        """
        if (seed is None):
            seed = (baseName or '') + str(niter)
            self.seed = seed
        seedRandomGenerators(seed)
        try:
            from git import Repo
            repo = Repo(os.getcwd())
            if not force:
                assert not repo.is_dirty()
            self.repo_version = repo.head.commit.hexsha
        except Exception:
            self.repo_version = 'unknown repo version'
        generated_rows = self._generateRows()
        if baseName and not retain_rows:
            self.rows = []
            self.saveFile(baseName, generated_rows)
        else:
            self.rows = list(generated_rows)
            if baseName:
                self.saveFile(baseName)

    def _generateRows(self):
        emodel = str(self.model)
        for i in range(self.niter):
            eid = uuid4()
            electorate = self.model(self.nvot, self.ncand)
            last_results = None
            for method, chooserFuns in self.methods:
                results = method.resultsTable(
                    eid,
                    emodel,
                    self.ncand,
                    electorate,
                    chooserFuns,
                    media=self.media,
                )
                yield from results
                last_results = results
            if last_results is not None:
                debug(i, last_results[1:3])

    def saveFile(self, baseName="SimResults", rows=None):
        """Print the result of doVse in an accessible format.
        for instance:

        csvs.saveFile()
        """
        i = 1
        while os.path.isfile(baseName + str(i) + ".csv"):
            i += 1
        rows = iter(self.rows if rows is None else rows)
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError("Cannot save a CSV batch with no result rows")
        keys = ["vse", "method", "chooser", *list(first_row.keys())]
        for n in range(4):
            keys.extend([f"tallyName{str(n)}", f"tallyVal{str(n)}"])
        keys = uniquify(keys)
        output_file = baseName + str(i) + ".csv"
        with open(output_file, "w", newline="") as myFile:
            print(
                f"# {dict(media=self.media.__name__, version=self.repo_version, seed=self.seed, model=self.model, methods=self.methods, nvot=self.nvot, ncand=self.ncand, niter=self.niter)}",
                file=myFile,
            )

            dw = csv.DictWriter(myFile, keys, restval = "NA")
            dw.writeheader()
            dw.writerow(first_row)
            for r in rows:
                dw.writerow(r)
        self.output_file = output_file
        return output_file



medianRuns = [
               OssChooser([beHon,ProbChooser([(1/2, beStrat), (1/2, beHon)])]),


               ProbChooser([(1/4, beX), (3/4, beHon)]),
               ProbChooser([(1/2, beX), (1/2, beHon)]),
               ProbChooser([(3/4, beX), (1/4, beHon)]),

               ProbChooser([(0.5, beStrat), (0.5, beHon)]),
               ProbChooser([(1/3, beStrat), (1/3, beHon), (1/3, beX)]),

               LazyChooser(),
               ProbChooser([(1/2, LazyChooser()), (1/2, beHon)]),

               ]

baseRuns = [
           OssChooser([beHon,ProbChooser([(1/2, beStrat), (1/2, beHon)])]),

           ProbChooser([(1/4, beStrat), (3/4, beHon)]),
           ProbChooser([(1/2, beStrat), (1/2, beHon)]),
           ProbChooser([(3/4, beStrat), (1/4, beHon)]),

           ]

allSystems = [[Score(1000), baseRuns],
                [Score(10), baseRuns],
                [Score(2), baseRuns],
                [Score(1), baseRuns],
                [BulletyApprovalWith(.6), baseRuns],
                [Srv(10), baseRuns],
                [Srv(2), baseRuns],
                [Plurality(), baseRuns],
                [Borda(), baseRuns],
                [Irv(), baseRuns],
                [IrvPrime(), baseRuns],
                [Schulze(), baseRuns],
                [Rp(), baseRuns],
                [V321(), baseRuns],
                [Mav(), medianRuns],
                [Mj(), medianRuns],
                [IRNR(), baseRuns],
                 ]

markMethods = [
                [Srv(2), baseRuns],
                [Srv(3), baseRuns],
                [Srv(4), baseRuns],
                [Srv(5), baseRuns],
                [Srv(6), baseRuns],
                [Srv(7), baseRuns],
                [Srv(8), baseRuns],
                [Srv(9), baseRuns],
                [Score(10), baseRuns],
                [V321(), baseRuns],
                [BulletyApprovalWith(.6), baseRuns],
                [Irv(), baseRuns],
                [Plurality(), baseRuns],
                 ]

if __name__ == "__main__":
    import doctest
    setDebug( False)
    doctest.testmod()
