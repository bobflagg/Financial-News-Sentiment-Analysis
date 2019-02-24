# -*- coding: utf-8 -*-

from fnsa.graph import get_distance, get_path
from fnsa.lexicon import get_en2scope
from fnsa.scope import ScopeDetector
import math
import numpy as np
import re

class ENScopeDetector(ScopeDetector):
    def __init__(self, max_delta=100, window=4):
        super(ENScopeDetector, self).__init__("en", max_delta=max_delta)
        self.window = window

    def __call__(self, doc, graph, lex2tokens):
        # The basic strategy is to build the context of a mention with the tokens (or phrases) which are either associated
        # with the mention by the appropriate scope detector or are very close to the mention in the dependency parse
        # tree. This strategy is refined with the following two adjustments:
        #   1. Tokens that don't get assigned by the basic procedure will be assigned to the closest entity.
        #   2. Some tokens should normally be assigned to only one entity.  For those we keep track of the strength of
        #      assignment and keep only the strongest one.
        assigned = set([])
        closest_entity_map = {}
        optimal_entity_map = {}

        entities = [e for e in lex2tokens.get('en', []) if e._.accepted]
        for entity in entities:
            # Limit to reasonable close tokens for efficiency:
            targets = [target for target in doc if np.abs(target.i - entity.i) < self.max_delta]
            for target in targets:
                j = target.i
                # include in context if very close and track closest entities:
                distance, _ = self.distance(graph, entity, target)
                if distance < self.window: self.apply(doc, entity.i, j)
                if j in closest_entity_map: indices, d = closest_entity_map[j]
                else:
                    indices = set([entity.i])
                    d = distance
                if distance < d:
                    d = distance
                    indices = set([entity.i])
                elif distance == d:
                    indices.add(entity.i)
                closest_entity_map[j] = (indices, d)
                # track optimal entity:
                include = False
                if target._.lex in ['fe']:
                    include, strength = self.include_fe(graph, entity, target)
                elif target._.lex in ['lm', 'fi']:
                    include, strength = self.include_lm_or_fi(graph, entity, target)
                if include:
                    if j not in optimal_entity_map: optimal_entity_map[j] = (entity.i, strength)
                    else:
                        best_position, best_strength = optimal_entity_map[j]
                        if strength > best_strength: optimal_entity_map[j] = (entity.i, strength)
        for j, data in optimal_entity_map.items():
            i, _ = data
            self.apply(doc, i, j)
            assigned.add(j)
        for j, data in closest_entity_map.items():
            if j in assigned: continue
            indices, _ = data
            for i in indices: self.apply(doc, i, j)
        merge_conjunctive_pairs(doc, graph, entities)

    def distance(self, graph, source, target):
        f = get_distance(graph, source, target)
        b = get_distance(graph, target, source)
        d = min(f, b)
        if d == 0: strength = math.inf
        else: strength = 1 / d
        return d, strength

    def include_fe(self, graph, source, target):
        d, strength = self.distance(graph, source, target)
        return d < self.window, strength

    def include_lm_or_fi(self, graph, source, target):
        d, strength = self.distance(graph, source, target)
        return d < self.window, strength

    def apply(self, doc, i, j):
        en_scope_map = get_en2scope(doc)
        if not i in en_scope_map: en_scope_map[i] = set([])
        en_scope_map[i].add(j)

        
def merge_conjunctive_pairs(doc, graph, entities):
    entities = sorted(entities, key=lambda t: t.i)
    pairs = {(first.i, second.i) for first in entities for second in entities if first.i < second.i}
    for i, j in pairs:
        merge = False
        path = get_path(graph, doc, doc[i], doc[j])
        if path.startswith('<-conj->'): merge = True
        if {rel for rel in re.split(r'<\-|\->', path) if rel} == {'conj'}: merge = True
        if merge:
            en_scope_map = get_en2scope(doc)
            en_scope_map[i] |= en_scope_map[j]
            en_scope_map[j] |= en_scope_map[i]

