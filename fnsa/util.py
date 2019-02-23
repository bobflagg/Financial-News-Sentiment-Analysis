# -*- coding: utf-8 -*-

def annotate(doc):
    annotations = []
    for token in doc:
        annotation = token.text
        if token._.lex != "nd":
            annotation = "[%s/%s:%s]" % (token.text, token._.lex, token._.category)
        annotations.append(annotation)
    return annotations

def show(doc, index=None, include_text=False):
    prefix = " "
    if include_text: 
        text = doc.text
        if index: 
            prefix = " " * len("%d. " % (index, ))
            text = "%d. %s" % (index, text)
        print(text)
    annotations = annotate(doc)
    print("%s%s" % (prefix, " ".join(annotations)))