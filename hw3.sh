#!/bin/sh

/opt/python-2.6/bin/python2.6 hw3.py parses.train trained.pcfg sents.test parses.hyp parses.improved.hyp

$1/evalb -p $1/COLLINS.prm parses.test parses.hyp > parses.hyp.eval
$1/evalb -p $1/COLLINS.prm parses.test parses.improved.hyp > parses.improved.hyp.eval
