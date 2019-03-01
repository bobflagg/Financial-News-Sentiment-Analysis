# -*- coding: utf-8 -*-

from fnsa.feature import FeatureExtractor
from fnsa.lexicon import get_en2scope, Lexicon
from fnsa.scope import DRScopeDetector, IFScopeDetector
from fnsa.graph import make_graph
from fnsa.util import *
import optparse
import os
import spacy
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


BASE_DIRECTORY = '/opt/code/github/Financial-News-Sentiment-Analysis'
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--dataset", help="The dataset to train on", default='all-agree')
    parser.add_option("--include-words", help="Include words among training features", action='store_true', default=False)
    (opts, args) = parser.parse_args()
    ext = 'without including words'
    if opts.include_words: ext = 'including words'
    print("Preparing %s classifcation data %s." % (opts.dataset, ext))
    
    nlp = spacy.load('en_core_web_sm')
    lexicon = Lexicon(nlp)
    dr_detector = DRScopeDetector()
    if_detector = IFScopeDetector()
    extractor = FeatureExtractor(lexicon, detectors=[dr_detector, if_detector], include_words=opts.include_words)
    
    if opts.dataset == 'all-agree':
        directory = '/opt/code/sentiment-analysis/data/fpb/FinancialPhraseBank-v1.0'
        fname = 'Sentences_AllAgree.txt'
        path = os.path.join(directory, fname)
        sentiment2code = {'negative':-1, 'neutral':0, 'positive':1}
        records = []
        with open(path, mode='r', encoding='Windows-1252') as ifp:
            cnt = 0
            for line in ifp:
                line = line.strip()
                if line:
                    text, sentiment = line.split('@')
                    records.append((text, sentiment2code[sentiment]))
                    cnt += 1
        print("Loaded %d records." % cnt)

        fname = '%s.tsv' % opts.dataset
        if opts.include_words: fname = '%s-with-words.tsv' % opts.dataset
        path = os.path.join(BASE_DIRECTORY, 'data', fname)
        print("Dumping classification data to %s." % path)
        with open(path, mode='w', encoding='UTF-8') as ofp:
            ofp.write("Sentiment\tSentence\tFeatures\n")
            for record in tqdm(records):
                text, code = record
                doc, features = extractor(text)
                features = " ".join([feature.lower() for feature in features])
                ofp.write("%d\t%s\t%s\n" % (code, text, features))

    if opts.dataset == 'ad-hoc':
        path = '/opt/code/github/SentenceLevelSentimentFinancialNews/adhoc_test.tsv'
        with open(path) as ifp: lines = ifp.readlines()
        records = []
        cnt = -1
        for line in lines:
            cnt += 1
            if cnt == 0: continue
            line = line.strip()
            if line:
                sentence, code = line.split('\t')
                sentence = sentence.strip()
                code = int(code)
                if code == 0: code = -1
                records.append((sentence, code))
        print("Loaded %d records." % cnt)        
    
        fname = '%s.tsv' % opts.dataset
        if opts.include_words: fname = '%s-with-words.tsv' % opts.dataset
        path = os.path.join(BASE_DIRECTORY, 'data', fname)
        print("Dumping classification data to %s." % path)
        with open(path, mode='w', encoding='UTF-8') as ofp:
            ofp.write("Sentiment\tSentence\tFeatures\n")
            for record in tqdm(records):
                text, code = record
                doc, features = extractor(text)
                features = " ".join([feature.lower() for feature in features])
                ofp.write("%d\t%s\t%s\n" % (code, text, features))



