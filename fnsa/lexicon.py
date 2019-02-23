# -*- coding: utf-8 -*-

from spacy.tokens import Doc, Token
from spacy.matcher import PhraseMatcher

DEFAULT_LEX = 'nd'

class Lexicon(object):

    def __init__(self, nlp, lexicons="fe fi dr if lm".split(), prune=False):
        extend_token()
        self.nlp = nlp
        self.lexicons = lexicons
        self.prune = prune
        seen = set([])
        self.matchers = [Matcher(nlp, lexicon, seen) for lexicon in lexicons]

    def __call__(self, text):
        doc = self.nlp(text)
        prepare(doc, self.lexicons)
        for e in doc.ents: store_match(doc, 'en', e.label_, e.start, e.end, -1, True)
        self.match(doc)
        self.merge(doc)
        return doc

    def match(self, doc):
        for matcher in self.matchers: matcher(doc)

    def merge(self, doc):
            lex_entries = list(get_entries(doc))
            lex_entries = sorted(lex_entries, key=lambda item: (-(item[3] - item[2]), item[0]))
            seen = set([])
            entries = []
            for item in lex_entries:
                lex, iid, start, end, index, accepted = item
                if start in seen or end - 1 in seen: continue
                seen |= set(range(start, end))
                doc[start]._.lex = lex
                doc[start]._.category = iid
                if lex == 'en':
                    doc[start]._.index = index
                    doc[start]._.accepted = accepted
                entries.append(item)
            entries = sorted(entries, key=lambda item: item[2])
            set_entries(doc, entries)
            delta = 0
            for item in entries:
                lex, iid, start, end, index, accepted = item
                start -= delta
                end -= delta
                doc[start:end].merge()
                delta += end - start - 1
            if self.prune:
                previous_lex = None
                entries = []
                for token in doc:
                    lex = token._.lex
                    if lex == DEFAULT_LEX:
                        if previous_lex != DEFAULT_LEX: start = token.start
                        stop = token.end
                    else:
                        if previous_lex == DEFAULT_LEX: entries.append((start, stop))
                    previous_lex = lex
                delta = 0
                for start, stop in entries:
                    start -= delta
                    stop -= delta
                    doc[start:stop].merge()
                    delta += stop - start - 1
        
class CallBack():
    def __init__(self, nlp, lexicon, category):
        self.nlp = nlp
        self.lexicon = lexicon
        self.category = category

    def store_matches(self, doc, matches):
        for ent_id, start, end in matches:
            store_match(doc, self.lexicon, self.nlp.vocab.strings[ent_id], start, end)

    def set_matched(self, doc):
        set_matched(doc, self.lexicon)

    def is_matched(self, doc):
        return is_matched(doc, self.lexicon)

    def __call__(self, matcher, doc, id, matches):
        if not self.is_matched(doc):
            self.set_matched(doc)
            self.store_matches(doc, matches)

class Matcher():

    def __init__(self, nlp, lexicon, seen):
        self.nlp = nlp
        self.lexicon = lexicon
        path = "./fnsa/lexicons/%s.tsv" % lexicon
        detection_map = self.load(path)
        self.matcher = PhraseMatcher(nlp.vocab)
        for category, phrases in detection_map.items():
            new_phrases = []
            for phrase in phrases:
                if phrase in seen: continue
                seen.add(phrase)
                new_phrases.append(phrase)
            self.matcher.add(category, CallBack(nlp, lexicon, category), *new_phrases)

    def __call__(self, doc): self.matcher(doc)

    def load(self, path, seen=set([])):
        detection_map = {}
        fp = open(path, mode='r', encoding='UTF-8')
        for line in fp:
            if line.startswith('#'): continue
            line = line.strip()
            if line:
                phrase, category = line.split('\t')
                if category == 'Deleted': continue
                if category not in detection_map: detection_map[category] = []
                phrase = self.nlp.tokenizer(phrase)
                detection_map[category].append(phrase)
        fp.close()
        return detection_map

def prepare(doc, lexicons):
    doc.user_data['entries'] = set([])
    doc.user_data['en2scope'] = {}
    for lex in lexicons: doc.user_data['%s_matched' % lex] = False
    
def store_match(doc, lex, iid, start, end, index=-1, accepted=False):
    doc.user_data['entries'].add((lex, iid, start, end, index, accepted))

def set_matched(doc, lex):
    doc.user_data['%s_matched' % lex] = True

def is_matched(doc, lex):
    return doc.user_data['%s_matched' % lex]

def get_entries(doc):
    return doc.user_data['entries']    

def set_entries(doc, entries):
    doc.user_data['entries'] = set(entries)

def get_en2scope(doc):
    return doc.user_data['en2scope']

def extend_token():
    Token.set_extension('lex', default=DEFAULT_LEX)
    Token.set_extension('index', default=-1)
    Token.set_extension('accepted', default=False)
    Token.set_extension('category', default='?')
    Token.set_extension('direction', default='=')
    Token.set_extension('influence', default='=')
    Token.set_extension('lps', default='?')
    Token.set_extension('entity', default='?')
    Token.set_extension('sentiment', default='=')

