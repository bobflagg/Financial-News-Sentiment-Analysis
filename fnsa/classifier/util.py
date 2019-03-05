# -*- coding: utf-8 -*-

from fnsa.classifier.classifier import NBSVM

import itertools

import matplotlib.pyplot as plt

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    xsize=8,
    ysize=8,
    path=None,
    show=False,
    store=False
):
    """
    Visualizes a given confusion matrix with optional normalization.
    """
    plt.figure(figsize=(xsize,ysize))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = "%d" % cm[i, j]
        if normalize: text = "%0.3f" % cm[i, j]
        plt.text(j, i, text, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if store and path is not None: plt.savefig(path, format='png')
    if show: plt.show()

def split(sentences, X, y, test_size=0.2, shuffle=True, random_state=None):
    indices = range(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y
    )
    sentences_train = [sentences[i] for i in train_indices]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    sentences_test = [sentences[i] for i in test_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return sentences_train, X_train, y_train, sentences_test, X_test, y_test

def kfold(X, y, n_splits=10, shuffle=True, random_state=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in skf.split(X, y):
        X_train = [X[i] for i in train_index]
        y_train = [y[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_test = [y[i] for i in test_index]
        yield X_train, y_train, X_test, y_test
        
def fit(X, y, ctype='nb'):
    if ctype == 'nb':
        alpha=0.025
        binary = True
        ngram_range = (1,2)
        use_fs = False
        use_idf = True
        tokenizer = lambda sentence: sentence.split()
        pipeline = [('vectorizer', CountVectorizer(ngram_range=ngram_range, tokenizer=tokenizer, binary=binary))]
        if use_fs: pipeline.append(('feature-selector', SelectFromModel(ExtraTreesClassifier())))
        if use_idf: pipeline.append(('transformer', TfidfTransformer()))
        pipeline.append(('estimator', MultinomialNB(alpha)))
    classifier = Pipeline(pipeline)
    classifier = classifier.fit(X, y)
    return classifier
    
def evaluate(classifier, X, y):
    y_predicted = classifier.predict(X)
    accuracy = accuracy_score(y, y_predicted)
    cm = confusion_matrix(y, y_predicted, labels=[-1, 0, 1])
    return accuracy, cm

def cross_validate(classifier, X, y, n_splits=10, shuffle=True, random_state=None):
    cms = []
    accuracies = []
    fold = 0
    for X_train, y_train, X_test, y_test in kfold(X, y, n_splits, shuffle, random_state):
        classifier.fit(X_train, y_train)
        accuracy, cm = evaluate(classifier, X_test, y_test)
        cms.append(cm)
        accuracies.append(accuracy)
        fold += 1
        print("Fold %02d accuracy = %0.3f." % (fold, accuracy))
    return np.mean(accuracies), sum(cms)

