# -*- coding: utf-8 -*-

from fnsa.classifier.classifier import NBSVM
from fnsa.classifier.data import load as load_data
from fnsa.classifier.util import cross_validate, evaluate, plot_confusion_matrix

import optparse
import os

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


BASE_DIRECTORY = '/opt/code/github/Financial-News-Sentiment-Analysis'
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--dataset", help="The dataset to train on", default='all-agree')
    parser.add_option("--include-words", help="Include words among training features", action='store_true', default=False)
    parser.add_option("--classifier", help="The algorithm to use", default='nb')
    (opts, args) = parser.parse_args()
    print("Training %s classifier on %s data." % (opts.classifier, opts.dataset))
    
    fname = opts.dataset
    if opts.include_words: fname = '%s-with-words' % opts.dataset
    path = os.path.join(BASE_DIRECTORY, 'data', fname + ".tsv")
    print("Loading data from %s." % (path, ))
    sentences, X, y = load_data(path)
    classes = [-1, 0, 1]
    
    alpha = 0.025
    beta = 0.25
    C = 1.0
    binary = True
    ngram_range = (1,2)
    use_fs = False
    use_idf = True

    pipeline = [('vectorizer', CountVectorizer(ngram_range=ngram_range, binary=binary))]
    if use_fs: pipeline.append(('feature-selector', SelectFromModel(ExtraTreesClassifier())))
    if use_idf: pipeline.append(('transformer', TfidfTransformer()))
    if opts.classifier == "nb": pipeline.append(('estimator', MultinomialNB(alpha=alpha)))
    else: pipeline.append(('estimator', NBSVM(alpha=alpha, C=C, beta=beta)))
    
    classifier = Pipeline(pipeline)

    accuracy, cm = cross_validate(classifier, X, y, n_splits=10, shuffle=True, random_state=None)
    plot_confusion_matrix(cm, classes)
    print("Average Accuracy = %0.3f." % (accuracy,))

    classifier = classifier.fit(X, y)
    path = os.path.join(BASE_DIRECTORY, 'model', '%s-%s.pkl' % (opts.classifier, fname))
    print("Saving trained model to %s." % (path, ))
    joblib.dump(classifier, path);


