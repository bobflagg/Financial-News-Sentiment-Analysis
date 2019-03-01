# -*- coding: utf-8 -*-

from fnsa.graph import get_distance
import math
import numpy as np


def score_candidate(graph, source, candidate):
    f = get_distance(graph, source, candidate)
    b = get_distance(graph, candidate, source)
    d = min(f, b)
    return d


class ScopeDetector(object):

    def __init__(self, name, max_delta=100):
        self.name = name
        self.max_delta = max_delta

    def __call__(self, doc, graph, lex2tokens):
        sources = self.get_sources(doc, lex2tokens)
        targets = self.get_targets(doc, lex2tokens)
        scores = []
        for source in sources:
            candidates = [target for target in targets if np.abs(target.i - source.i) < self.max_delta]
            for candidate in candidates:
                scores.append((source, candidate, score_candidate(graph, source, candidate), np.abs(candidate.i - source.i)))
        scores = sorted(scores, key=lambda item: item[-2:])
        assigned = set([])
        for source, candidate, *rest in scores:
            if candidate.i in assigned: continue
            self.apply(source, candidate)
            assigned.add(candidate.i)

    def get_sources(self, doc, lex2tokens):
        raise NotImplementedError("Subclasses must implement this method!")

    def get_targets(self, doc, lex2tokens):
        raise NotImplementedError("Subclasses must implement this method!")

    def apply(self, source, target):
        raise NotImplementedError("Subclasses must implement this method!")

        
class SimpleScopeDetector(ScopeDetector):

    def __init__(self, source_lex, target_lex, max_delta=100):
        super(SimpleScopeDetector, self).__init__("%s-%s" % (source_lex, target_lex), max_delta)
        self.source_lex = source_lex
        self.target_lex = target_lex

    def get_sources(self, doc, lex2tokens):
        return lex2tokens.get(self.source_lex, [])

    def get_targets(self, doc, lex2tokens):
        return lex2tokens.get(self.target_lex, [])

    
class DRScopeDetector(SimpleScopeDetector):

    def __init__(self, max_delta=100):
        super(DRScopeDetector, self).__init__('dr', 'fe', max_delta)

    def apply(self, source, target): target._.direction = source._.category

        
class IFScopeDetector(ScopeDetector):

    def __init__(self, max_delta=100):
        super(IFScopeDetector, self).__init__("if", max_delta)

    def get_sources(self, doc, lex2tokens):
        sources = lex2tokens.get('if', [])
        return sources

    def get_targets(self, doc, lex2tokens):
        targets = lex2tokens.get('fe', []) + lex2tokens.get('fi', []) + lex2tokens.get('lm', [])
        return targets

    def apply(self, source, target):
        if target._.influence == 'Reversal' and source._.category == 'Reversal': target._.influence = '='
        else: target._.influence = source._.category



