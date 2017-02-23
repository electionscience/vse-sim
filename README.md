# Voter Satisfaction Efficiency
These are some methods for running VSE (Voter Satisfaction Efficiency)
simulations for various voting systems. 

See [Voter Satisfaction Efficiency](http://electology.github.io/vse-sim/VSE/) for an explanation of the methods and results.

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
    >>> csvs = CsvBatch(PolyaModel(), [[Score(), baseRuns], [Mav(), medianRuns]], nvot=5, ncand=4, niter=3)
    >>> csvs.saveFile()

and look for the results in `SimResults1.csv`
