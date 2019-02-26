# -*- coding: utf-8 -*-

import os

def load(fname='all-agree.tsv', directory='./data'):
    path = os.path.join(directory, fname)
    corpus = []
    with open(path, mode='r', encoding='UTF-8') as ifp:
        for line in ifp:
            line = line.strip()
            if line:
                code, text, features = line.split('\t')
                corpus.append((code, text, features))
    corpus = corpus[1:]
    X = [record[-1] for record in corpus]
    sentences = [record[1] for record in corpus]
    y = [int(record[0]) for record in corpus]
    return sentences, X, y

