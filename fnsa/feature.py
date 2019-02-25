# -*- coding: utf-8 -*-

from fnsa.graph import make_graph
from fnsa.util import build_lex2tokens, DEFAULT_LEX


def project(doc):
    for token in doc:
        if token._.lex in ['dr', 'en', 'if']: token._.lps = '%s_%s' % (token._.lex, token._.category)
        elif token._.lex in ['fi', 'lm']: token._.lps = '%s_%s_%s' % (token._.lex, token._.category, token._.influence)
        elif token._.lex == 'fe': token._.lps = '%s_%s_%s_%s' % (token._.lex, token._.category, token._.direction, token._.influence)

            
def compute_feature(token): 
    if token._.lex == DEFAULT_LEX: return "%s_%s" % (token._.lex, token.text)
    return token._.lps


class FeatureExtractor(object):

    def __init__(self, lexicon, detectors):
        self.lexicon = lexicon
        self.detectors = detectors

    def __call__(self, text):
        doc = self.lexicon(text)
        graph = make_graph(doc)
        lex2tokens = build_lex2tokens(doc)
        for detector in self.detectors: detector(doc, graph, lex2tokens)
        project(doc)
        return doc, [compute_feature(token) for token in doc]
