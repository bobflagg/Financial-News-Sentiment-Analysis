# -*- coding: utf-8 -*-

from fnsa.graph import make_graph
from fnsa.lexicon import get_en2scope
from fnsa.util import build_lex2tokens, DEFAULT_LEX, EXTRACTED_ENTITY_LEX, STRICT_LEXICONS

VERY_STRICT_F_TYPE = 0
STRICT_F_TYPE = 1
REGULAR_F_TYPE = 2
FLUSH_F_TYPE = 3

def project(doc):
    for token in doc:
        if token._.lex in ['dr', EXTRACTED_ENTITY_LEX, 'if']: token._.lps = '%s_%s' % (token._.lex, token._.category)
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
        if token._.lex != EXTRACTED_ENTITY_LEX:
            features.extend([w.lower() for w in token.text.split()])
        features.append(token._.lps)


class FeatureExtractor(object):

    def __init__(self, lexicon, detectors, ftype=FLUSH_F_TYPE):
        self.lexicon = lexicon
        self.detectors = detectors
        self.type = ftype

    def __call__(self, text):
        doc = self.lexicon(text)
        graph = make_graph(doc)
        lex2tokens = build_lex2tokens(doc)
        for detector in self.detectors: detector(doc, graph, lex2tokens)
        project(doc)
        return doc, self.compute_features(doc, lex2tokens)

    def compute_features(self, doc, lex2tokens):
        if self.type == FLUSH_F_TYPE:
            features = []
            for token in doc: append_features(features, token)
            return features
        if self.type == STRICT_F_TYPE:
            return [token._.lps for token in doc if token._.lex in STRICT_LEXICONS]
        if self.type == VERY_STRICT_F_TYPE:
            return [token._.lps for token in doc if token._.lex in STRICT_LEXICONS and token._.lps.split('_')[-2] in ['-', '+']]
        return [compute_feature(token) for token in doc]

            
class EntityFeatureExtractor(FeatureExtractor):

    def compute_features(self, doc, lex2tokens):
        en_scope_map = get_en2scope(doc)
        entities = []
        features = []
        for entity in lex2tokens.get(EXTRACTED_ENTITY_LEX, []):
            if entity._.accepted:
                entities.append(entity)
                tokens = [doc[j] for j in en_scope_map.get(entity.i, set([]))]
                tokens = sorted(tokens, key=lambda t: t.i)
                entity_features = [compute_feature(token) for token in tokens]
                features.append(entity_features)
        return entities, features

