# -*- coding: utf-8 -*-

from fnsa.feature import VERY_STRICT_F_TYPE, STRICT_F_TYPE, REGULAR_F_TYPE, FLUSH_F_TYPE
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


class Tokenizer(object):

    def __call__(self, text): return text.split()

BASE_DIRECTORY = '/opt/code/github/Financial-News-Sentiment-Analysis'
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--dataset", help="The dataset to train on", default='all-agree')
    parser.add_option("--ftype", help="The type of feature extraction to use", default='regular') 
    # options are: strict, regular and flush
    parser.add_option("--use-tf-idf", help="Use TF-IDF when computing features", action='store_true', default=False)
    parser.add_option("--use-fs", help="Use feature selection", action='store_true', default=False)
    parser.add_option("--binary", help="Make count vectorizer binary", action='store_true', default=False)
    parser.add_option("--classifier", help="The algorithm to use", default='nb')
    parser.add_option("--n-gram-bound", help="Upper bound on n-grams to use in features", type=int, default=2)
    parser.add_option("--alpha", help="The alpha value to use in training", type=float, default=0.025)
    parser.add_option("--beta", help="The beta value to use in training", type=float, default=0.25)
    parser.add_option("-C", help="The C  value to use in training", type=float, default=1.0)
    (opts, args) = parser.parse_args()
    if opts.ftype == 'very-strict': ftype = VERY_STRICT_F_TYPE
    if opts.ftype == 'strict': ftype = STRICT_F_TYPE
    if opts.ftype == 'regular': ftype = REGULAR_F_TYPE
    if opts.ftype == 'flush': ftype = FLUSH_F_TYPE
    print('\n###############################################################################')
    print("Training %s classifier on %s data with %s feature extraction." % (opts.classifier, opts.dataset, opts.ftype))
    
    fname = '%s-%d' % (opts.dataset, ftype)    
    path = os.path.join(BASE_DIRECTORY, 'data', fname + ".tsv")
    print("Loading data from %s." % (path, ))
    sentences, X, y = load_data(path)
    classes = [-1, 0, 1]
    tokenizer = Tokenizer()
    pipeline = [('vectorizer', CountVectorizer(ngram_range=(1,opts.n_gram_bound), tokenizer=tokenizer, binary=opts.binary))]
    if opts.use_fs: pipeline.append(('feature-selector', SelectFromModel(ExtraTreesClassifier())))
    if opts.use_tf_idf: pipeline.append(('transformer', TfidfTransformer()))
    if opts.classifier == "nb": pipeline.append(('estimator', MultinomialNB(alpha=opts.alpha)))
    else: pipeline.append(('estimator', NBSVM(alpha=opts.alpha, C=opts.C, beta=opts.beta)))
    
    classifier = Pipeline(pipeline)

    accuracy, cm = cross_validate(classifier, X, y, n_splits=10, shuffle=True, random_state=None)
    plot_confusion_matrix(cm, classes)
    print("Average Accuracy = %0.3f." % (accuracy,))

    classifier = classifier.fit(X, y)
    path = os.path.join(BASE_DIRECTORY, 'model', '%s-%s.pkl' % (opts.classifier, fname))
    print("Saving trained model to %s." % (path, ))
    joblib.dump(classifier, path);
    print('###############################################################################\n')



