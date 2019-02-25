# -*- coding: utf-8 -*-

from nltk.tree import Tree

ACCEPTABLE_TYPES = ['ORG']
DEFAULT_LEX = 'nd'
DEFAULT_LEXICONS = "fe fi dr if lm".split()

def build_lex2tokens(doc):
    lex2tokens = {}
    for token in [token for token in doc if token._.lex != DEFAULT_LEX]:
        lex = token._.lex
        if lex not in lex2tokens: lex2tokens[lex] = []
        lex2tokens[lex].append(token)
    return lex2tokens

def to_nltk_tree(node, include_punctuation=False):
    # https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    if node.n_lefts + node.n_rights > 0:
        return Tree(
            node.orth_,
            [to_nltk_tree(child) for child in node.children if include_punctuation or child.pos_ != 'PUNCT'])
    else:
        return node.orth_

def show_tree(doc):
    sent = list(doc.sents)[0]
    to_nltk_tree(sent.root).pretty_print()

def annotate_token(token):
    if token._.lex != "nd":
        return "[%s/%s:%s]" % (token.text, token._.lex, token._.category)
    return token.text

def annotate(doc):
    return [annotate_token(token) for token in doc]

def show(doc, index=None, include_text=False, include_tree=False):
    prefix = " "
    if include_text: 
        text = doc.text
        if index: 
            prefix = " " * len("%d. " % (index, ))
            text = "%d. %s" % (index, text)
        print(text)
    annotations = annotate(doc)
    print("%s%s" % (prefix, " ".join(annotations)))
    if include_tree: show_tree(doc)
    
def show_tokens(doc, include_annotations=False, include_text=False):
    if include_annotations: show(doc, include_text=include_text)
    elif include_text: print(doc.text)
    for token in doc:
        prefix = "[%s||%s]" % (token._.lex, token._.category[:3])
        print("%10s %20s -->> %s||%s||%s" % (prefix, token.text, token._.direction, token._.influence, token._.lps))
        
def load_sentences(path="./sentences.txt"):
    sentences = []
    with open(path) as fp: 
        for line in fp:
            line = line.strip()
            if line: sentences.append(line)
    return sentences