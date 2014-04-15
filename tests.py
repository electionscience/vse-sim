import unittest
import doctest
import vse, voterModels, stratFunctions, methods, dataClasses

from debugDump import *

debug("Hi")
setDebug(False)
debug("High 5")

def load_tests(loader, tests, ignore):
    
    setDebug(False)
    tests.addTests(doctest.DocTestSuite(vse))
    tests.addTests(doctest.DocTestSuite(voterModels))
    tests.addTests(doctest.DocTestSuite(stratFunctions))
    tests.addTests(doctest.DocTestSuite(methods))
    tests.addTests(doctest.DocTestSuite(dataClasses))
    return tests