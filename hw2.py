#!/opt/python-2.6/bin/python2.6
__author__ = 'David McHugh'

import ling571
import sys

cfFile = sys.argv[1]
cnfFile = sys.argv[2]
sntcFile = sys.argv[3]
parseFile = sys.argv[4]

#   load context free grammar
g = ling571.Grammar()
g.load(cfFile)
#   convert grammar to chomsky normal form
cnfG = ling571.CnfGrammar()
cnfG.convertFromCfg(g)
#    output converted grammar to file
cnfG.writeToFile(cnfFile)

#   load sentences to parse
p = ling571.CkyParser(cnfG)
p.load(sntcFile)
#   print parses to file
p.writeParsesToFile(parseFile)