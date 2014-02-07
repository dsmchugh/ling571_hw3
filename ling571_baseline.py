#!/opt/python-2.6/bin/python2.6
__author__ = 'David McHugh'

import copy
import math
import nltk

class Rule:
    splitter = '->'

    def __init__(self, left = None, right = None):

        if not left is None and not right is None:
            self.buildRule(left, right)
        else:
            self._string = ''
            self.left = None
            self.right = []
            self.logProbability = 0

    def __eq__(self,other):
        other.toString() == self.toString()

    def getProbability(self):
        return math.pow(10,self.logProbability)

    def buildRule(self, left, right):
            self.left = left

            if isinstance(right, Tag):
                self.right = [right]
            else:
                self.right = right

    def _buildString(self):
        self._string = self.left.name + ' ' + self.splitter
        for tag in self.right:
            self._string += ' ' + tag.name

    def containsTerminal(self):
        #    if rule string contains apostrophe, signifies a terminal is present
        #    does not rule out hybrid
        return Tag(self.toString()).isTerminal()

    def isBinary(self):
        if len(self.right) == 2:
            return True
        return False

    def isUnitProduction(self):
        if not self.containsTerminal() and len(self.right) == 1:
            return True
        return False

    def toString(self):
        self._buildString()
        return self._string

    def firstRight(self):
        return self.right[0]

    def setFirstRight(self, tag):
        self.right[0] = tag
        self._string = ''

    def secondRight(self):
        if len(self.right) > 1:
            return self.right[1]

    def setSecondRight(self, tag):
        self.right[1] = tag
        self._string = ''

    def addToRule(self, tag):
        if self.left is None:
            self.left = tag
        else:
            self.right.append(tag)
        self._string = ''


class Tag:
    def __init__(self, name):
        self.name = name.strip()
        self.pointers = []
        self.logProbability = 0

    def __eq__(self, other):
        return self.name == other.name

    def isTerminal(self):
        if self.name.count('\'') >= 2:
            return True
        return False

    def getProbability(self):
        return math.pow(10,self.logProbability)


class Pointer:
    def __init__(self, i, j, tagIdx):
        self.i = i
        self.j = j
        self.tagIdx = tagIdx


class Grammar:
    start = 'TOP'

    def __init__(self):
        self.rules = []
        self._newTags = []

    def _generateNewTag(self):
        newTagCnt = len(self._newTags)
        tag = Tag('X' + str(newTagCnt + 1))

        while self._newTagExists(tag):
            newTagCnt += 1
            tag = Tag('X' + str(newTagCnt))

        self._newTags.append(tag)
        return tag

    def _newTagExists(self, newTag):
        if newTag in self._newTags:
            return True
        return False

    def _getNewTag(self, oldTag):
        newTag = Tag(oldTag.name.replace('\'', '').upper())

        if not newTag.name.isalnum():
            return self._generateNewTag()

        self._newTags.append(newTag)
        return newTag

    @staticmethod
    def parseRule(line):
        split = line.split(Rule.splitter)

        leftTag = Tag(split[0])
        rightStrings = split[1].split()

        rightTags = []
        for string in rightStrings:
            rightTags.append(Tag(string))

        return Rule(leftTag, rightTags)

    def load(self, inputFile):
        grammarFile = open(inputFile)
        grammarLines = grammarFile.readlines()
        grammarFile.close()

        del self.rules[:]
        del self._newTags[:]

        for line in grammarLines:
            if not Rule.splitter in line:
                continue

            self.rules.append(self.parseRule(line.replace('\n', '').replace('  ', ' ')))

    def writeToFile(self, outputFile, includeProbabilities = False):
        nonterminals = []
        terminals = []

        #    split terminals off for sorting purposes only
        for rule in self.rules:
            if len(rule.right) == 1:
                terminals.append(rule)
            else:
                nonterminals.append(rule)

        nonterminals.sort(key=lambda tup: tup.toString())
        terminals.sort(key=lambda tup: tup.toString())

        grammarFile = open(outputFile, 'w')
        grammarFile.write('#\t' + __author__ + '\n\n')

        for rule in nonterminals:
            self._writeLine(grammarFile, rule, includeProbabilities)

        for rule in terminals:
            self._writeLine(grammarFile, rule, includeProbabilities)

        grammarFile.close()

    def _writeLine(self, grammarFile, rule, writeProb):
        grammarFile.write(rule.toString())
        if writeProb:
            grammarFile.write(' [' + str(rule.getProbability()) + ']')
        grammarFile.write('\n')


class CnfGrammar(Grammar):
    def __init__(self):
        Grammar.__init__(self)
        self._todo = []
        self._done = []

    #    precondition: all cnf rules already removed
    def _convertHybrids(self):
        for i, rule in enumerate(self._todo):
            #    if contains terminal, must be hybrid
            if rule.containsTerminal():
                #    will be fixed, remove from to-do list and fix
                self._todo.pop(i)
                newRules = self._convertHybrid(rule)
                self._sortRules(newRules)

    def _convertHybrid(self, rule):
        newRules = []
        newRights = []

        #    iterate over tags in right side of rule
        #    if tag is terminal, create new left side rule
        for rightTag in rule.right:
            if rightTag.isTerminal():
                newTag = self._getNewTag(rightTag)
                newRights.append(newTag)
                newRule = Rule(newTag, rightTag)
                newRules.append(newRule)
            else:
                newRights.append(rightTag)

        #    return newly created rules
        newRules.append(Rule(rule.left, newRights))
        return newRules

    #    precondition all cnf and hybrid rules are removed
    def _convertUnitProductions(self):
        for i, rule in enumerate(self._todo):
            if rule.isUnitProduction():
                self._todo.pop(i)
                newRules = self._convertUnitProduction(rule)
                self._sortRules(newRules)
                self._convertUnitProductions()


    def _convertUnitProduction(self, rule):
        children = []

        #    right side of unit production rule
        fromTag = rule.firstRight()
        #    left side of unit production rule
        toTag = rule.left

        #    find all rules (fixed and non) where left side of rule
        #    matches right side of unit production rule
        for cnfRule in self._done:
            if cnfRule.left == fromTag:
                children.append(cnfRule)

        for cfRule in self._todo:
            if cfRule.left == fromTag:
                children.append(cfRule)

        newRules = []

        #    create new rules where left of unit prod rule
        #    is left of rules where right side of unit prod
        #    rule is left
        #    A -> B and B-> C D to A -> C D
        for child in children:
            newRule = Rule(toTag, child.right)
            newRules.append(newRule)

        return newRules

    #    precondition: all cnf, hybrid, unit removed
    def _convertNonBinaries(self):
        for i, rule in enumerate(self._todo):
            if not rule.isBinary():
                self._todo.pop(i)
                newRules = self._convertNonBinary(rule)
                self._sortRules(newRules)
                self._convertNonBinaries()

    def _convertNonBinary(self, rule):
        newRules = []
        #    create new tag to represent first two tags in right side
        #    of non binary rule
        newLeft = self._generateNewTag()
        #    take first two tags in right side of nonbinary rule for new rule
        newRight = [rule.firstRight(), rule.secondRight()]
        #    create new rule
        newRule = Rule(newLeft, newRight)
        newRules.append(newRule)

        #    create new rule from old nonbinary rule with new tag for first two tags on right
        #    remove old rule from todo list and add updated rule to new list
        oldLeft = rule.left
        oldRight = rule.right
        rule.right.pop(1)
        rule.setFirstRight(newLeft)
        newRules.append(Rule(oldLeft, oldRight))

        return newRules

    @staticmethod
    def _isCnf(rule):
        if not rule.containsTerminal():
            if rule.isBinary():
                return True
            return False
        if len(rule.right) == 1:
            return True
        return False

    def _sortRules(self, rules):
        for rule in rules:
            if self._isCnf(rule):
                #if any(i.right == rule.right for i in self._done):
                #    continue
                self._done.append(copy.deepcopy(rule))
            else:
                self._todo.append(copy.deepcopy(rule))

    #    convert a context free grammar to chomsky normal form
    def convertFromCfg(self, grammar):
        self.rules = grammar.rules

        self._sortRules(self.rules)

        self._convertHybrids()

        self._convertUnitProductions()

        self._convertNonBinaries()

        self.rules = self._done

class Parser:
    start = '('
    end = ')'

    def __init__(self):
        self._sentences = []
        self._grammar = Grammar

    def load(self, fileName):
        sentenceFile = open(fileName)
        sentenceLines = sentenceFile.readlines()
        sentenceFile.close()

        del self._sentences[:]

        for line in sentenceLines:
            if '' == line:
                continue

            self._sentences.append(line.replace('\n', ''))

    def writeParsesToFile(self, fileName, includeSentence = True, includeCounts = True):
        #    total number of parses
        totalCnt = 0
        #    total number of sentences
        sentenceCnt = 0

        #    clear parse output file
        output = open(fileName, 'w')
        #output.write('#\t' + __author__ + '\n\n')
        output.close()

        #    generate and print trees for each sentence
        for sentence in self._sentences:
            parses = self._parse(sentence)
            parseCnt = len(parses)

            output = open(fileName, 'a')

            if includeSentence:
                output.write(sentence + '\n')

            if len(parses) == 0:
                output.write('\n')
            else:
                for parse in parses:
                    output.write(parse + '\n')

            if includeCounts:
                output.write('\n' + str(parseCnt) + '\n\n')

            totalCnt += parseCnt
            sentenceCnt += 1

        #    calculate and print average number of parses per sentence
        output = open(fileName, 'a')
        if includeCounts:
            output.write(str(float(totalCnt) / float(sentenceCnt)) + '\n')
        output.close()


class CkyParser(Parser):

    def __init__(self, cnfGrammar):
        Parser.__init__(self)
        assert isinstance(cnfGrammar, CnfGrammar)
        self._grammar = cnfGrammar


    def _parse(self, sentence):
        table = self._buildTable(sentence)
        strings = self._buildStrings(table, sentence)
        return strings

    def _buildTable(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        length = len(tokens)

        table = [[Cell() for i in range(length + 1)] for i in range(length + 1)]

        j = 1
        while j <= length:
            #    fill cell with left side of rules where right side is correct terminal
            table[j - 1][j].addTags(
                [rule.left for rule in self._grammar.rules if rule.firstRight().name == '\'' + tokens[j - 1] + '\''])

            i = j - 2
            while i >= 0:
                k = i + 1
                while k <= j - 1:
                    #    iterate all rules
                    for rule in self._grammar.rules:
                        #    iterate tags in "b" cell
                        for bIdx, bTag in enumerate(table[i][k].tags):
                            #    if position "b" in rule matches tag in "b" cell
                            if rule.firstRight().name == bTag.name:
                                #    iterate tags in "c" cell
                                for cIdx, cTag in enumerate(table[k][j].tags):
                                    #    if position "c" in rule matches tag in "c" cell
                                    if rule.secondRight().name == cTag.name:
                                        bPointer = Pointer(i, k, bIdx)
                                        cPointer = Pointer(k, j, cIdx)
                                        tag = Tag(rule.left.name)
                                        tag.pointers.append(bPointer)
                                        tag.pointers.append(cPointer)
                                        table[i][j].addTag(tag)
                    k += 1
                i -= 1
            j += 1

        return table

    def _buildStrings(self, table, sentence):
        strings = []

        rootCell = table[0][len(table) - 1]

        tokens = nltk.word_tokenize(sentence)

        for tag in rootCell.tags:
            tokenCopy = copy.deepcopy(tokens)
            #    if any tags are start tag in root cell, begin to descent into tree
            if tag.name == self._grammar.start:
                string = ''
                depth = 0
                string = self._addNodeToString(table, tag, string, depth, tokenCopy)
                strings.append(string)

        return strings

    def _addNodeToString(self, table, tag, string, depth, tokens):
        string += '\n'
        for x in range(depth):
            string += '  '

        string += self.start + tag.name
        depth += 1

        for pointer in tag.pointers:
            childTag = table[pointer.i][pointer.j].tags[pointer.tagIdx]
            string = self._addNodeToString(table, childTag, string, depth, tokens)

        if len(tag.pointers) == 0:
            if len(tokens) > 0:
                string += ' ' + tokens[0]
                tokens.pop(0)

        string += self.end
        return string


class ProbCkyParser(Parser):
    _rules = []

    def __init__(self, cnfGrammar):
        Parser.__init__(self)
        self._grammar = cnfGrammar

    def train(self, fileName):
        inputFile = open(fileName)
        parseLines = inputFile.readlines()
        inputFile.close()

        for line in parseLines:
            self._readLine(line)

        self._calculateProbabilities()

    def _calculateProbabilities(self):
        ruleCounts = {}
        leftCounts = {}

        for rule in self._rules:
            string = rule.toString()
            left = rule.left.name

            if ruleCounts.has_key(string):
                ruleCnt = ruleCounts[string]
                ruleCnt += 1
                ruleCounts[string] = ruleCnt
            else:
                ruleCounts[string] = 1

            if leftCounts.has_key(left):
                leftCnt = leftCounts[left]
                leftCnt += 1
                leftCounts[left] = leftCnt
            else:
                leftCounts[left] = 1

        addedToGrammar = []

        for rule in self._rules:
            string = rule.toString()
            if not string in addedToGrammar:
                leftCount = leftCounts[rule.left.name]
                ruleCount = ruleCounts[string]
                rule.logProbability = math.log10(float(ruleCount)) - math.log10(float(leftCount))

                if len(rule.right) == 1:
                    rule.firstRight().logProbability = rule.logProbability

                self._grammar.rules.append(rule)
                addedToGrammar.append(string)

    def _readLine(self, line):
        tokens = nltk.word_tokenize(line)

        #   nltk tokenizer separates apostrophe--reattach
        for i,token in enumerate(tokens):
            if token == '\'':
                tokens[i-1] += '\''
                tokens.pop(i)

        newRules = []

        for token in tokens:

            if token == self.start:
                newRules.append(Rule())
            elif token == self.end:
                ruleIdx = len(newRules) - 1
                rule = newRules[ruleIdx]
                if ruleIdx > 0:
                    parentRule = newRules[ruleIdx - 1]
                    parentRule.addToRule(rule.left)

                if rule.isUnitProduction():
                    rule.firstRight().isPreterminal = True
                    rule.firstRight().name = '\'' + rule.firstRight().name + '\''

                self._rules.append(newRules[ruleIdx])
                newRules.pop(ruleIdx)

            else:
                ruleIdx = len(newRules) - 1
                newRules[ruleIdx].addToRule(Tag(token))

    def _buildTable(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        length = len(tokens)

        table = [[Cell() for i in range(length + 1)] for i in range(length + 1)]

        j = 1
        while j <= length:
            #    fill cell with left side of rules where right side is correct terminal
            table[j - 1][j].addTags(
                [rule.left for rule in self._grammar.rules if rule.firstRight().name == '\'' + tokens[j - 1] + '\''])

            i = j - 2
            while i >= 0:
                k = i + 1
                while k <= j - 1:
                    #    iterate all rules
                    for rule in self._grammar.rules:
                        #    iterate tags in "b" cell
                        for bIdx, bTag in enumerate(table[i][k].tags):
                            #    if position "b" in rule matches tag in "b" cell
                            if rule.firstRight().name == bTag.name:
                                #    iterate tags in "c" cell
                                for cIdx, cTag in enumerate(table[k][j].tags):
                                    #    if position "c" in rule matches tag in "c" cell
                                    if rule.secondRight().name == cTag.name:
                                        cell = table[i][j]
                                        aTag = Tag(rule.left.name)
                                        aProb = rule.logProbability
                                        bProb = bTag.logProbability
                                        cProb = cTag.logProbability
                                        tmpProb = aProb + bProb + cProb

                                        cellTag = None
                                        for tag in cell.tags:
                                            if tag.name == rule.left.name:
                                                cellTag = tag

                                        if cellTag != None:
                                            if cellTag.logProbability > tmpProb:
                                                cell.tags.remove(cellTag)
                                            else:
                                                continue

                                        bPointer = Pointer(i, k, bIdx)
                                        cPointer = Pointer(k, j, cIdx)
                                        aTag.pointers.append(bPointer)
                                        aTag.pointers.append(cPointer)
                                        table[i][j].addTag(aTag)
                    k += 1
                i -= 1
            j += 1

        return table

    def _parse(self, sentence):
        table = self._buildTable(sentence)
        strings = self._buildStrings(table, sentence)
        return strings

    def _buildStrings(self, table, sentence):
        strings = []

        rootCell = table[0][len(table) - 1]

        tokens = nltk.word_tokenize(sentence)

        for tag in rootCell.tags:
            tokenCopy = copy.deepcopy(tokens)
            #    if any tags are start tag in root cell, begin to descent into tree
            if tag.name == self._grammar.start:
                string = ''
                string = self._addNodeToString(table, tag, string, tokenCopy)
                strings.append(string.strip())

        return strings

    def _addNodeToString(self, table, tag, string, tokens):
        string += ' ' + self.start + tag.name

        for pointer in tag.pointers:
            childTag = table[pointer.i][pointer.j].tags[pointer.tagIdx]
            string = self._addNodeToString(table, childTag, string, tokens)

        if len(tag.pointers) == 0:
            if len(tokens) > 0:
                string += ' ' + tokens[0]
                tokens.pop(0)

        string += self.end
        return string


class Cell:
    def __init__(self):
        self.tags = []

    def addTag(self, tag):
        self.tags.append(tag)

    def addTags(self, tags):
        for tag in tags:
            self.addTag(tag)