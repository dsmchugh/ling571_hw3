#!/opt/python-2.6/bin/python2.6
__author__ = 'David McHugh'

import ling571
import sys

trainInput = sys.argv[1]
trainResults = sys.argv[2]

g = ling571.CnfGrammar()
p = ling571.ProbCkyParser(g)

p.readParses(trainInput)
g.writeToFile(trainResults, includeProbabilities=True)