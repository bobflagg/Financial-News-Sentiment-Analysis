# -*- coding: utf-8 -*-

from fnsa.graph import make_graph
from fnsa.lexicon import get_en2scope
from fnsa.util import build_lex2tokens, DEFAULT_LEX


def project(doc):
    for token in doc:
        if token._.lex in ['dr', 'en', 'if']: token._.lps = '%s_%s' % (token._.lex, token._.category)
        elif token._.lex in ['fi', 'lm']: token._.lps = '%s_%s_%s' % (token._.lex, token._.category, token._.influence)
        elif token._.lex == 'fe': token._.lps = '%s_%s_%s_%s' % (token._.lex, token._.category, token._.direction, token._.influence)

            
def compute_feature(token): 
    if token._.lex == DEFAULT_LEX: 
        return token.text
    return token._.lps

            
def append_features(features, token): 
    if token._.lex == DEFAULT_LEX: 
        features.append(token.text)
    else:
        if token._.lex != 'en':
            features.extend([w.lower() for w in token.text.split()])
        features.append(token._.lps)


class FeatureExtractor(object):

    def __init__(self, lexicon, detectors, include_words=False):
        self.lexicon = lexicon
        self.detectors = detectors
        self.include_words = include_words

    def __call__(self, text):
        doc = self.lexicon(text)
        graph = make_graph(doc)
        lex2tokens = build_lex2tokens(doc)
        for detector in self.detectors: detector(doc, graph, lex2tokens)
        project(doc)
        return doc, self.compute_features(doc, lex2tokens)

    def compute_features(self, doc, lex2tokens):
        if self.include_words:
            features = []
            for token in doc: append_features(features, token)
            return features
        return [compute_feature(token) for token in doc]

            
class EntityFeatureExtractor(FeatureExtractor):

    def compute_features(self, doc, lex2tokens):
        en_scope_map = get_en2scope(doc)
        entities = []
        features = []
        for entity in lex2tokens.get('en', []):
            if entity._.accepted:
                entities.append(entity)
                tokens = [doc[j] for j in en_scope_map.get(entity.i, set([]))]
                tokens = sorted(tokens, key=lambda t: t.i)
                entity_features = [compute_feature(token) for token in tokens]
                features.append(entity_features)
        return entities, features

