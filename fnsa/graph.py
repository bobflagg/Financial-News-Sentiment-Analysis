# -*- coding: utf-8 -*-

import math
import networkx as nx

def build_lex2tokens(doc):
    lex2tokens = {}
    for token in [token for token in doc if token._.lex != DEFAULT_LEX]:
        lex = token._.lex
        if lex not in lex2tokens: lex2tokens[lex] = []
        lex2tokens[lex].append(token)
    return lex2tokens

def get_distance(graph, source, target):
    head = source.head
    if source.i == head.i: return math.inf
    try:
        path_list = nx.dijkstra_path(graph, str(head.i), str(target.i))
        return len(path_list) - 1
    except nx.NetworkXNoPath:
        return math.inf
    except KeyError:
        return math.inf

def get_path(graph, doc, source, target):
    try:
        path_list = nx.dijkstra_path(graph, str(target.i), str(source.i))
        prev_node = str(target.i)
        dep_path = ""
        for node in path_list[1:]:
            direction = graph[prev_node][node]['dir']
            dep_path += direction
            if direction == '->':
                dep_path += doc[int(node)].dep_
            else:
                dep_path += doc[int(prev_node)].dep_
            prev_node = node
        return dep_path
    except nx.NetworkXNoPath:
        return 'null'
    except KeyError:
        return 'null'

def make_graph(doc, include_punctuation=False):
    graph = nx.DiGraph()
    for token in doc:
        if token.head.i == token.i: continue
        if include_punctuation or token.pos_ != 'PUNCT':
            graph.add_edge(str(token.head.i), str(token.i), **{'dir': '->', 'dep':token.dep_})
            graph.add_edge(str(token.i), str(token.head.i), **{'dir': '<-', 'dep':token.dep_})
    return graph

