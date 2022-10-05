import collections
import copy
import optparse

from ling.Tree import Tree
import ling.Trees as Trees
import parsers.EnglishPennTreebankParseEvaluator as EnglishPennTreebankParseEvaluator
#from parser import EnglishPennTreebankParseEvaluator
import ios.PennTreebankReader as PennTreebankReader
import ios.MASCTreebankReader as MASCTreebankReader
import numpy as np

class Parser:

    def train(self, train_trees):
        pass

    def get_best_parse(self, sentence):
        """
        Should return a Tree
        """
        pass

def mlog(n):
    return np.log(n) if n > 0 else float('Inf')

from graphviz import Digraph

class PCFGParser(Parser):

    def train(self, train_trees):
        # TODO: before you generate your grammar, the training
        #       trees need to be binarized so that rules are at
        #       most binary

        #for train_tree in train_trees:
        #    TreeAnnotations.vertical_markovization(train_tree)
            #print str(train_tree)
        
        self.lexicon = Lexicon(train_trees)
        #self.grammar = Grammar(train_trees)
        binary_trees = []
        for train_tree in train_trees:
            binary_trees.append(TreeAnnotations.annotate_tree(train_tree))
        #for train_tree in binary_trees:
            #print train_tree
        self.grammar = Grammar(binary_trees)
        #print self.grammar
        '''
        nonterms = set(self.lexicon.get_all_tags())
        nnonterms = set([])
        for NT in nonterms:
            rules = self.grammar.get_unary_rules_by_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
            rules = self.grammar.get_binary_rules_by_left_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
            rules = self.grammar.get_binary_rules_by_right_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
        nonterms = nonterms.union(nnonterms)
        for NT in nonterms:
            rules = self.grammar.get_unary_rules_by_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
                #print rule.parent, rule.child
            rules = self.grammar.get_binary_rules_by_left_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
                #print rule.parent, rule.left_child, rule.right_child
            rules = self.grammar.get_binary_rules_by_right_child(NT)
            for rule in rules:
                nnonterms.add(rule.parent)
                #print rule.parent, rule.left_child, rule.right_child
        self.nonterms = nonterms.union(nnonterms)
        '''
        
        print('\nLexicons learnt:\n')
        for word in self.lexicon.word_to_tag_counters:
            for A in self.lexicon.word_to_tag_counters[word]:
				if self.lexicon.word_to_tag_counters[word][A] > 0:
					print A + '->' + word + ' prob: ' + str(float(self.lexicon.word_to_tag_counters[word][A])  / sum([self.lexicon.word_to_tag_counters[word][A] for word in self.lexicon.word_counter]))
                
        print('\nGrammar learnt:\n')
        print('\nBinary Rules Learnt:\n')
        self.nonterms = set([])
        for left_child in self.grammar.binary_rules_by_left_child:
            self.nonterms.add(left_child)
            for binary_rule in self.grammar.get_binary_rules_by_left_child(left_child):
                self.nonterms.add(binary_rule.parent)
                self.nonterms.add(binary_rule.right_child)
                if binary_rule.score > 0:
					print binary_rule.parent + '->' + left_child + ' ' + binary_rule.right_child + ' prob: ' + str(binary_rule.score)
        print('\nUnary Rules Learnt:\n')
        for child in self.grammar.unary_rules_by_child:
            self.nonterms.add(child)
            for unary_rule in self.grammar.get_unary_rules_by_child(child):
                self.nonterms.add(unary_rule.parent)
                if unary_rule.score > 0:
					print unary_rule.parent + '->' + child + ' prob: ' + str(unary_rule.score)
                #print self.nonterms
        print('\n')

    def get_best_parse(self, sentence):
        """
        Should return a Tree.
        'sentence' is a list of strings (words) that form a sentence.
        """
        # TODO: implement this method (CKY)
        n = len(sentence)
        #print sentence, n, self.nonterms, len(self.nonterms)        
        #print sentence, 'S' in self.nonterms, ',' in self.nonterms, n, self.nonterms, len(self.nonterms) 
                           
        score = [[collections.defaultdict(lambda: 0.0) for _ in range(n+1)] for _ in range(n+1)]
        back = [[collections.defaultdict(lambda: '') for _ in range(n+1)] for _ in range(n+1)]
        
        for i in range(n):
            
            # handle terminals        
            for A in self.nonterms:
                #prob = self.lexicon.score_tagging(sentence[i], A)
                prob = self.lexicon.word_to_tag_counters[sentence[i]][A] 
                if prob > 0: #if A -> words[i] in grammar
                    prob = float(prob) / sum([self.lexicon.word_to_tag_counters[word][A] for word in self.lexicon.word_counter])
                    score[i][i+1][A] = prob #mlog(prob)

            # handle unaries
            added = True
            while added:
                added = False
                for B in self.nonterms: # score[i][i+1]:
                    rules = self.grammar.get_unary_rules_by_child(B)
                    for rule in rules:
                        A = rule.parent
                        prob = rule.score * score[i][i+1][B] #P(A->B)*score[i][i+1][B]
                        if prob > score[i][i+1][A]:
                            score[i][i+1][A] = prob
                            back[i][i+1][A] = (B,)
                            added = True
        
        #print 'Done CKY termnials'
            
        # handle binaries
        for span in range(2, n + 1):
            for begin in range(n - span + 1):
                
                end = begin + span
                for split in range(begin + 1, end):
                    for B in score[begin][split]: #self.nonterms:
                       rules = self.grammar.get_binary_rules_by_left_child(B)
                       for rule in rules:
                           A = rule.parent
                           C = rule.right_child
                           prob = score[begin][split][B] * score[split][end][C] * rule.score #P(A->BC)
                           if prob > score[begin][end][A]:
                               score[begin][end][A] = prob
                               back[begin][end][A] = (split,B,C)
                    for C in score[split][end]: #self.nonterms:
                       rules = self.grammar.get_binary_rules_by_right_child(C)
                       for rule in rules:
                           A = rule.parent
                           B = rule.left_child                            
                           #if rule.parent == A:
                           prob = score[begin][split][B] * score[split][end][C] * rule.score #P(A->BC)
                           #prob = mlog(score[begin][split][B]) + mlog(score[split][end][C]) + mlog(rule.score) #P(A->BC)
                           if prob > score[begin][end][A]:
                               score[begin][end][A] = prob
                               back[begin][end][A] = (split,B,C)
                               
                # handle unaries
                added = True
                while added:
                    added = False
                    for B in self.nonterms:
                        rules = self.grammar.get_unary_rules_by_child(B)
                        for rule in rules:
                            A = rule.parent
                            prob = rule.score * score[begin][end][B]
                            if prob > score[begin][end][A]:
                                score[begin][end][A] = prob
                                back[begin][end][A] = (B,)
                                added = True
        
        '''
		table = [['' for _ in range(n+1)] for _ in range(n+1)]
        for i in range(n+1):
            for j in range(n+1):
                for nt in self.nonterms:
                    if score[i][j][nt] == 0: continue
                    #print i,j, nt, back[i][j][nt]
                    table[i][j] += (str(nt) + '->' + str(back[i][j][nt]) + ':' + str(score[i][j][nt]) + ' | ')
        np.savetxt('C:\courses\Coursera\Past\NLP\Stanford_NLP\Assignment\pa6-cky-v2\python\CKY.txt', table, delimiter=';', fmt='%s')
        '''
        
        #print 'Done CKY!'
        
        return self.build_tree(back, 0, n, 'ROOT', sentence) #, tree) #'S' if 'S' in back[0][n] else 'ROOT') # 'S' 'ROOT' 
        
        
    def build_tree(self, back, i, j, A, sentence):
        
        #print NT
        if back[i][j][A] != '': # a non-terminal
            syms = back[i][j][A]
            #print 'here', A, syms, len(syms)
            if len(syms) == 1: # unary
                B = syms[0]
                #print 'here_unary', A, i, j, B
                if A in [',', '.', ':', '-', '_']:
                    return Tree(A, [Tree(sentence[i])])
                else:
                    child = self.build_tree(back, i, j, B, sentence)
                    return Tree(A, [child])
            elif len(syms) == 3: # binary
                split, B, C = list(syms)
                #print 'here_binary', A, i, j, split, B, C
                left_child = self.build_tree(back, i, split, B, sentence)
                right_child = self.build_tree(back, split, j, C, sentence)
                return Tree(A, [left_child, right_child])
        else:
                #print 'here_terminal', A, i, j, sentence[i]
                return Tree(A, [Tree(sentence[i])])

    def build_tree1(self, back, i, j, A, sentence, tree):
        
        #print NT
        if back[i][j][A] != '': # a non-terminal
            syms = back[i][j][A]
            #print 'here', A, syms, len(syms)
            if len(syms) == 1: # unary
                B = syms[0]
                tree.node(A+str(i)+'_'+str(j), label=A)
                tree.node(B+str(i)+'_'+str(j), label=B)
                tree.edge(A+str(i)+'_'+str(j), B+str(i)+'_'+str(j))
                #print 'here_unary', A, i, j, B
                if A in [',','.']:
                    return Tree(A, [Tree(B)])
                else:
                    child = self.build_tree(back, i, j, B, sentence, tree)
                    return Tree(A, [child])
            elif len(syms) == 3: # binary
                split, B, C = list(syms)
                #print 'here_binary', A, i, j, split, B, C
                left_child = self.build_tree(back, i, split, B, sentence, tree)
                right_child = self.build_tree(back, split, j, C, sentence, tree)
                tree.node(A+str(i)+'_'+str(j), label=A)
                tree.node(B+str(i)+'_'+str(split), label=B)
                tree.node(C+str(split)+'_'+str(j), lael=C)
                tree.edge(A+str(i)+'_'+str(j), B+str(i)+'_'+str(split))
                tree.edge(A+str(i)+'_'+str(j), C+str(split)+'_'+str(j))
                return Tree(A, [left_child, right_child])
        else:
                #print 'here_terminal', A, i, j, sentence[i]
                tree.node(A+str(i)+'_'+str(j), label=A)
                tree.node(sentence[i]+str(i), label=sentence[i])
                tree.edge(A+str(i)+'_'+str(j), sentence[i]+str(i))
                return Tree(A, [Tree(sentence[i])])
                
class BaselineParser(Parser):

    def train(self, train_trees):
        self.lexicon = Lexicon(train_trees)
        self.known_parses = {}
        self.span_to_categories = {}
        for train_tree in train_trees:
            tags = train_tree.get_preterminal_yield()
            tags = tuple(tags)  # because lists are not hashable, but tuples are
            if tags not in self.known_parses:
                self.known_parses[tags] = {}
            if train_tree not in self.known_parses[tags]:
                self.known_parses[tags][train_tree] = 1
            else:
                self.known_parses[tags][train_tree] += 1
            self.tally_spans(train_tree, 0)
        
        '''
        binary_trees = []
        for train_tree in train_trees:
            binary_trees.append(TreeAnnotations.annotate_tree(train_tree))
        self.grammar = Grammar(binary_trees)
        for rule in self.grammar.get_binary_rules_by_left_child('NP'):
            print rule.parent, rule.left_child, rule.right_child
        for rule in self.grammar.get_binary_rules_by_right_child('VP'):
            print rule.parent, rule.left_child, rule.right_child # rule
        #print self.grammar.unary_rules_by_child
        for rule in self.grammar.get_unary_rules_by_child('N'):
            print rule.parent, rule.child
        print self.lexicon.word_to_tag_counters['crabs']['N']
        '''
        
    def get_best_parse(self, sentence):
        tags = self.get_baseline_tagging(sentence)
        tags = tuple(tags)
        if tags in self.known_parses:
            return self.get_best_known_parse(tags, sentence)
        else:
            return self.build_right_branch_parse(sentence, list(tags))

    def build_right_branch_parse(self, words, tags):
        cur_position = len(words) - 1
        right_branch_tree = self.build_tag_tree(words, tags, cur_position)
        while cur_position > 0:
            cur_position -= 1
            right_branch_tree = self.merge(
                    self.build_tag_tree(words, tags, cur_position),
                    right_branch_tree)
        right_branch_tree = self.add_root(right_branch_tree)
        return right_branch_tree

    def merge(self, left_tree, right_tree):
        span = len(left_tree.get_yield()) + len(right_tree.get_yield())
        maxval = max(self.span_to_categories[span].values())
        for key in self.span_to_categories[span]:
            if self.span_to_categories[span][key] == maxval:
                most_freq_label = key
                break
        return Tree(most_freq_label, [left_tree, right_tree])

    def add_root(self, tree):
        return Tree("ROOT", [tree])

    def build_tag_tree(self, words, tags, cur_position):
        leaf_tree = Tree(words[cur_position])
        tag_tree = Tree(tags[cur_position], [leaf_tree])
        return tag_tree

    def get_best_known_parse(self, tags, sentence):
        maxval = max(self.known_parses[tags].values())
        for key in self.known_parses[tags]:
            if self.known_parses[tags][key] == maxval:
                parse = key
                break
        parse = copy.deepcopy(parse)
        parse.set_words(sentence)
        return parse

    def get_baseline_tagging(self, sentence):
        tags = [self.get_best_tag(word) for word in sentence]
        return tags

    def get_best_tag(self, word):
        best_score = 0
        best_tag = None
        for tag in self.lexicon.get_all_tags():
            score = self.lexicon.score_tagging(word, tag)
            if best_tag is None or score > best_score:
                best_score = score
                best_tag = tag
        return best_tag

    def tally_spans(self, tree, start):
        if tree.is_leaf() or tree.is_preterminal():
            return 1
        end = start
        for child in tree.children:
            child_span = self.tally_spans(child, end)
            end += child_span
        category = tree.label
        if category != "ROOT":
            if end-start not in self.span_to_categories:
                self.span_to_categories[end-start] = {}
            if category not in self.span_to_categories[end-start]:
                self.span_to_categories[end-start][category] = 1
            else:
                self.span_to_categories[end-start][category] += 1
        return end - start

class TreeAnnotations:

    @classmethod
    def annotate_tree(cls, unannotated_tree):
        """
        Currently, the only annotation done is a lossless binarization
        """

        # TODO: change the annotation from a lossless binarization to a
        # finite-order markov process (try at least 1st and 2nd order)
        # mark nodes with the label of their parent nodes, giving a second
        # order vertical markov process
        #TreeAnnotations.vertical_markovization(unannotated_tree)
        #print str(unannotated_tree)
        return TreeAnnotations.binarize_tree(unannotated_tree)

    @classmethod
    def vertical_markovization(cls, tree, parent_label = None):
        for child in tree.children:
            if not child.is_leaf():
                TreeAnnotations.vertical_markovization(child, tree.label)
        if parent_label:
            tree.label += '^' + parent_label
        
    @classmethod
    def binarize_tree(cls, tree):
        label = tree.label
        if tree.is_leaf():
            return Tree(label)
        if len(tree.children) == 1:
            return Tree(label, [TreeAnnotations.binarize_tree(tree.children[0])])

        intermediate_label = "@%s->" % label
        intermediate_tree = TreeAnnotations.binarize_tree_helper(
                tree, 0, intermediate_label)
        return Tree(label, intermediate_tree.children)

    @classmethod
    def binarize_tree_helper(cls, tree, num_children_generated,
            intermediate_label):
        left_tree = tree.children[num_children_generated]
        children = []
        children.append(TreeAnnotations.binarize_tree(left_tree))
        if num_children_generated < len(tree.children) - 1:
            right_tree = TreeAnnotations.binarize_tree_helper(
                    tree, num_children_generated + 1,
                    intermediate_label + "_" + left_tree.label)
            children.append(right_tree)
        return Tree(intermediate_label, children)


    @classmethod
    def at_filter(cls, string):
        if string.startswith('@'):
            return True
        else:
            return False

    @classmethod
    def unannotate_tree(cls, annotated_tree):
        """
        Remove intermediate nodes (labels beginning with "@")
        Remove all material on node labels which follow their base
        symbol (cuts at the leftmost -, ^, or : character)
        Examples: a node with label @NP->DT_JJ will be spliced out,
        and a node with label NP^S will be reduced to NP
        """
        debinarized_tree = Trees.splice_nodes(annotated_tree,
                TreeAnnotations.at_filter)
        unannotated_tree = Trees.FunctionNodeStripper.transform_tree(
                debinarized_tree)
        return unannotated_tree


class Lexicon:
    """
    Simple default implementation of a lexicon, which scores word,
    tag pairs with a smoothed estimate of P(tag|word)/P(tag).

    Instance variables:
    word_to_tag_counters
    total_tokens
    total_word_types
    tag_counter
    word_counter
    type_tag_counter
    """

    def __init__(self, train_trees):
        """
        Builds a lexicon from the observed tags in a list of training
        trees.
        """
        self.total_tokens = 0.0
        self.total_word_types = 0.0
        self.word_to_tag_counters = collections.defaultdict(lambda: \
                collections.defaultdict(lambda: 0.0))
        self.tag_counter = collections.defaultdict(lambda: 0.0)
        self.word_counter = collections.defaultdict(lambda: 0.0)
        self.type_to_tag_counter = collections.defaultdict(lambda: 0.0)

        for train_tree in train_trees:
            words = train_tree.get_yield()
            tags = train_tree.get_preterminal_yield()
            for word, tag in zip(words, tags):
                self.tally_tagging(word, tag)


    def tally_tagging(self, word, tag):
        if not self.is_known(word):
            self.total_word_types += 1
            self.type_to_tag_counter[tag] += 1
        self.total_tokens += 1
        self.tag_counter[tag] += 1
        self.word_counter[word] += 1
        self.word_to_tag_counters[word][tag] += 1


    def get_all_tags(self):
        return self.tag_counter.keys()


    def is_known(self, word):
        return word in self.word_counter


    def score_tagging(self, word, tag):
        p_tag = float(self.tag_counter[tag]) / self.total_tokens
        c_word = float(self.word_counter[word])
        c_tag_and_word = float(self.word_to_tag_counters[word][tag])
        if c_word < 10:
            c_word += 1
            c_tag_and_word += float(self.type_to_tag_counter[tag]) \
                    / self.total_word_types
        p_word = (1.0 + c_word) / (self.total_tokens + self.total_word_types)
        p_tag_given_word = c_tag_and_word / c_word
        return p_tag_given_word / p_tag * p_word


class Grammar:
    """
    Simple implementation of a PCFG grammar, offering the ability to
    look up rules by their child symbols.  Rule probability estimates
    are just relative frequency estimates off of training trees.

    self.binary_rules_by_left_child
    self.binary_rules_by_right_child
    self.unary_rules_by_child
    """

    def __init__(self, train_trees):
        self.unary_rules_by_child = collections.defaultdict(lambda: [])
        self.binary_rules_by_left_child = collections.defaultdict(
                lambda: [])
        self.binary_rules_by_right_child = collections.defaultdict(
                lambda: [])

        unary_rule_counter = collections.defaultdict(lambda: 0)
        binary_rule_counter = collections.defaultdict(lambda: 0)
        symbol_counter = collections.defaultdict(lambda: 0)

        for train_tree in train_trees:
            self.tally_tree(train_tree, symbol_counter,
                    unary_rule_counter, binary_rule_counter)
        for unary_rule in unary_rule_counter:
            unary_prob = float(unary_rule_counter[unary_rule]) \
                    / symbol_counter[unary_rule.parent]
            unary_rule.score = unary_prob
            self.add_unary(unary_rule)
        for binary_rule in binary_rule_counter:
            binary_prob = float(binary_rule_counter[binary_rule]) \
                    / symbol_counter[binary_rule.parent]
            binary_rule.score = binary_prob
            self.add_binary(binary_rule)


    def __unicode__(self):
        rule_strings = []
        for left_child in self.binary_rules_by_left_child:
            for binary_rule in self.get_binary_rules_by_left_child(
                    left_child):
                rule_strings.append(str(binary_rule))
        for child in self.unary_rules_by_child:
            for unary_rule in self.get_unary_rules_by_child(child):
                rule_strings.append(str(unary_rule))
        return "%s\n" % "".join(rule_strings)


    def add_binary(self, binary_rule):
        self.binary_rules_by_left_child[binary_rule.left_child].\
                append(binary_rule)
        self.binary_rules_by_right_child[binary_rule.right_child].\
                append(binary_rule)


    def add_unary(self, unary_rule):
        self.unary_rules_by_child[unary_rule.child].append(unary_rule)


    def get_binary_rules_by_left_child(self, left_child):
        return self.binary_rules_by_left_child[left_child]


    def get_binary_rules_by_right_child(self, right_child):
        return self.binary_rules_by_right_child[right_child]


    def get_unary_rules_by_child(self, child):
        return self.unary_rules_by_child[child]


    def tally_tree(self, tree, symbol_counter, unary_rule_counter,
            binary_rule_counter):
        if tree.is_leaf():
            return
        if tree.is_preterminal():
            return
        if len(tree.children) == 1:
            unary_rule = self.make_unary_rule(tree)
            symbol_counter[tree.label] += 1
            unary_rule_counter[unary_rule] += 1
        if len(tree.children) == 2:
            binary_rule = self.make_binary_rule(tree)
            symbol_counter[tree.label] += 1
            binary_rule_counter[binary_rule] += 1
        if len(tree.children) < 1 or len(tree.children) > 2:
            raise Exception("Attempted to construct a Grammar with " \
                    + "an illegal tree: " + str(tree))
        for child in tree.children:
            self.tally_tree(child, symbol_counter, unary_rule_counter,
                    binary_rule_counter)


    def make_unary_rule(self, tree):
        return UnaryRule(tree.label, tree.children[0].label)


    def make_binary_rule(self, tree):
        return BinaryRule(tree.label, tree.children[0].label,
                tree.children[1].label)


class BinaryRule:
    """
    A binary grammar rule with score representing its probability.
    """

    def __init__(self, parent, left_child, right_child):
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.score = 0.0


    def __str__(self):
        return "%s->%s->%s %% %s" % (self.parent, self.left_child, self.right_child, self.score)


    def __hash__(self):
        result = hash(self.parent)
        result = 29 * result + hash(self.left_child)
        result = 29 * result + hash(self.right_child)
        return result


    def __eq__(self, o):
        if self is o:
            return True

        if not isinstance(o, BinaryRule):
            return False

        if (self.left_child != o.left_child):
            return False
        if (self.right_child != o.right_child):
            return False
        if (self.parent != o.parent):
            return False
        return True


class UnaryRule:
    """
    A unary grammar rule with score representing its probability.
    """

    def __init__(self, parent, child):
        self.parent = parent
        self.child = child
        self.score = 0.0

    def __str__(self):
        return "%s->%s %% %s" % (self.parent, self.child, self.score)

    def __hash__(self):
        result = hash(self.parent)
        result = 29 * result + hash(self.child)
        return result

    def __eq__(self, o):
        if self is o:
            return True

        if not isinstance(o, UnaryRule):
            return False

        if (self.child != o.child):
            return False
        if (self.parent != o.parent):
            return False
        return True


MAX_LENGTH = 20

def save_tree(root, tree, level=0, index=0, edges=set([])):
    
    root_label = root.label + '_' + str(level) + '_' + str(index)
    for i in range(len(root.children)):
         child = root.children[i]
         tree.node(root_label, label=root.label)
         child_label = child.label + '_' + str(level+1) + '_' + str(i)
         tree.node(child_label, label=child.label)
         if not ((root_label, child_label) in edges):
             tree.edge(root_label, child_label)
             edges.add((root_label, child_label))
         save_tree(child, tree, level+1, i, edges)

import string

def test_parser(parser, test_trees):
    evaluator = EnglishPennTreebankParseEvaluator.LabeledConstituentEval(
            ["ROOT"], set(["''", "``", ".", ":", ","]))
    #rx = re.compile(':"')
    #rx.sub(r'\\\1', 'abc:cdef_ij"')
    for test_tree in test_trees:
        test_sentence = test_tree.get_yield()
        if len(test_sentence) > 20:
            continue
        guessed_tree = parser.get_best_parse(test_sentence)
        
        #TreeAnnotations.vertical_markovization(test_tree)

        print "Guess:\n%s" % Trees.PennTreeRenderer.render(guessed_tree)
        print "Gold:\n%s" % Trees.PennTreeRenderer.render(test_tree)
        
        try:

            tree = Digraph('gold_tree', node_attr={'shape': 'plaintext'}, format='png')
            save_tree(test_tree, tree, level=0, index=0, edges=set([]))
            tree.render('trees/pcfg_gold_' + "".join(filter(lambda x: ord(x)<128, '_'.join(test_sentence[:6]).translate(None, string.punctuation))), view=False)
            #tree.render('trees/pcfg_gold_' + '_'.join('_'.join(test_sentence[:6]).replace(':', '_')), view=False)
            #print()
            tree = Digraph('guessed_tree', node_attr={'shape': 'plaintext'}, format='png')
            save_tree(guessed_tree, tree, level=0, index=0, edges=set([]))
            tree.render('trees/pcfg_guessed_' + "".join(filter(lambda x: ord(x)<128, '_'.join(test_sentence[:6]).translate(None, string.punctuation))), view=False)
            #tree.render('trees/pcfg_guessed_' + '_'.join('_'.join(test_sentence[:6]).replace(':', '_')), view=False)

        except: 
            pass        
        
        evaluator.evaluate(guessed_tree, test_tree)
    print ""
    return evaluator.display(True)


def read_trees(base_path, low=None, high=None):
    print "Reading MASC from %s" % base_path
    trees = PennTreebankReader.read_trees(base_path, low, high)
    return [Trees.StandardTreeNormalizer.transform_tree(tree) \
        for tree in trees]


def read_masc_trees(base_path, low=None, high=None):
    print "Reading MASC from %s" % base_path
    trees = MASCTreebankReader.read_trees(base_path, low, high)
    return [Trees.StandardTreeNormalizer.transform_tree(tree) \
        for tree in trees]


if __name__ == '__main__':
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--path", dest="path",
            default="../data/parser")
    opt_parser.add_option("--data", dest="data", default = "masc")
    opt_parser.add_option("--parser", dest="parser",
            default="BaselineParser")
    opt_parser.add_option("--maxLength", dest="max_length",
            default="20")
    opt_parser.add_option("--testData", dest="test_data", default="")

    (options, args) = opt_parser.parse_args()
    options = vars(options)

    print "PCFGParserTest options:"
    for opt in options:
        print "  %-12s: %s" % (opt, options[opt])
    print ""
    MAX_LENGTH = int(options['max_length'])

    parser = globals()[options['parser']]()
    print "Using parser: %s" % parser.__class__.__name__

    base_path = options['path']
    pre_base_path = base_path
    data_set = options['data']
    if not base_path.endswith('/'):
        base_path += '/'

    print "Data will be loaded from: %s" % base_path

    train_trees = []
    validation_trees = []
    test_trees = []

    if data_set == 'miniTest':
        base_path += 'parser/%s' % data_set

        # training data: first 3 of 4 datums
        print "Loading training trees..."
        train_trees = read_trees(base_path, 1, 3)
        print "done."

        # test data: last of 4 datums
        print "Loading test trees..."
        test_trees = read_trees(base_path, 4, 4)
        print "done."

    if data_set == "masc":
        base_path += "parser/"

        # training data: MASC train
        print "Loading MASC training trees... from: %smasc/train" % base_path
        train_trees.extend(read_masc_trees("%smasc/train" % base_path, 0, 38))
        print "done."
        print "Train trees size: %d" % len(train_trees)
        print "First train tree: %s" % \
                Trees.PennTreeRenderer.render(train_trees[0])
        print "Last train tree: %s" % \
                Trees.PennTreeRenderer.render(train_trees[-1])

        # test data: MASC devtest
        print "Loading MASC test trees..."
        test_trees.extend(read_masc_trees("%smasc/devtest" % base_path, 0, 11))
        #test_trees.extend(read_masc_trees("%smasc/blindtest" % base_path, 0, 8))
        print "done."
        print "Test trees size: %d" % len(test_trees)
        print "First test tree: %s" % \
                Trees.PennTreeRenderer.render(test_trees[0])
        print "Last test tree: %s" % \
                Trees.PennTreeRenderer.render(test_trees[-1])


    if data_set not in ["miniTest", "masc"]:
        raise Exception("Bad data set: %s: use miniTest or masc." % data_set)

    print ""
    print "Training parser..."
    parser.train(train_trees)

    print "Testing parser"
    test_parser(parser, test_trees)

# python PCFGParserTester.py --parser PCFGParser --path ../data/ --data masc/miniTest