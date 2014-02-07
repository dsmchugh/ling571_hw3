#!/opt/python-2.6/bin/python2.6
__author__ = 'David McHugh'

import ling571
import ling571_baseline
import sys

trainInput = sys.argv[1]
trainResults = sys.argv[2]
sentences = sys.argv[3]
parses_baseline = sys.argv[4]
parses_improved = sys.argv[5]

def run(module, output):
    g_baseline = module.CnfGrammar()
    p_baseline = module.ProbCkyParser(g_baseline)

    p_baseline.train(trainInput)
    g_baseline.writeToFile(trainResults, includeProbabilities=True)

    p_baseline.load(sentences)
    p_baseline.writeParsesToFile(output, includeSentence=False, includeCounts=False)


run(ling571_baseline, parses_baseline)
run(ling571_baseline, parses_improved)
