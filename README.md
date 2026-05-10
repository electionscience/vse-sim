# Voter Satisfaction Efficiency

These are some methods for running VSE (Voter Satisfaction Efficiency)
simulations for various voting systems.

See [Voter Satisfaction Efficiency FAQ](http://electionscience.github.io/vse-sim/) for an explanation of the methods and results.

## Installing the code

Requirements: Python 3.10+, NumPy, SciPy.

Testing uses doctests, which should make most things pretty self-documenting.

E.g.:

    python3 -m doctest methods.py
    python3 -m doctest voterModels.py
    python3 -m doctest dataClasses.py
    python3 vse.py

Or, using the development dependencies from the Pipfile:

    pipenv install --dev
    pipenv run python -m pytest --doctest-modules

To generate local coverage artifacts:

    pipenv install --dev
    pipenv run coverage

The GitHub Actions workflow runs the same coverage check on pushes, pull requests,
and manual dispatches. It uploads the HTML coverage report plus machine-readable
coverage and JUnit XML files as workflow artifacts.

## Security automation

GitHub Actions also runs CodeQL code scanning for Python on pushes, pull
requests, a weekly schedule, and manual dispatches. Dependabot checks Python and
GitHub Actions dependencies weekly, and the dependency review workflow blocks
pull requests that introduce dependencies with moderate or higher vulnerabilities.

## Running simulations

Try

    $ python3
    >>> from vse import CsvBatch, baseRuns, Mav, medianRuns, Score
    >>> from voterModels import PolyaModel
    >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
    >>> csvs.saveFile()

and look for the results in `SimResults1.csv`
