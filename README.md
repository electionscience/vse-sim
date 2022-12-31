# Voter Satisfaction Efficiency


# Voter Satisfaction Efficiency
These are some methods for running VSE (Voter Satisfaction Efficiency)
simulations for various voting systems. 

See [Voter Satisfaction Efficiency FAQ](http://electionscience.github.io/vse-sim/) for an explanation of the methods and results.

# Installing the code
Requirements: python3, scipy, pydoc

Testing uses pydoc, which should make most things pretty self-documenting.

E.g.:

    python3 -m doctest methods.py
    python3 -m doctest voterModels.py
    python3 -m doctest dataClasses.py
    python3 vse.py

# Running simulations

Try

    $ python3
    >>> from vse import CsvBatch, baseRuns, Mav, medianRuns, Score
    >>> from voterModels import PolyaModel
    >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
    >>> csvs.saveFile()

and look for the results in `SimResults1.csv`
* Install python dependencies (eg, scipy) and run "run sims for paper.ipynb"
* Install R dependencies (eg, data.table) and run "create graphs for paper.R". Working directory must be here.
* Install shell dependency (Inkscape for Mac — or, if other platform, edit script) and run "convert_pngs_for_paper.sh"

# Recreating figures for STAR voting paper

To reproduce the figures in the paper:

* Install python dependencies (eg, scipy) and run "run sims for paper.ipynb"
* Install R dependencies (eg, data.table) and run "create graphs for paper.R". Working directory must be here.
* Install shell dependency (Inkscape for Mac — or, if other platform, edit script) and run "convert_pngs_for_paper.sh"