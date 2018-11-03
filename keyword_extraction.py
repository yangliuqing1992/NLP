# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:46:20 2018

@author: yangl

This a script for keyword extraction
"""

from nltk import tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel


# construct a corpus with 6 documents text2
text2=['The movie was about a spaceship and aliens.',
              'I really liked the movie!',
              'Awesome action scenes, but boring characters.',
              'The movie was awful! I hate alien films.',
              'Space is cool! I liked the movie.',
              'More space films, please!']

#function preprocess will tokenize a document and remove punctuations, stopwords.
def preprocess(sentence):
	sentence = sentence.lower()
	tokenizer = tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	filtered_words = [w for w in tokens if not w in stopwords.words('english')]
	return filtered_words

#function keyword will rank the words, in each ducuments, from most important to the least
def keywords(corpus):
    docs=[preprocess(doc) for doc in corpus]
    dictionary = Dictionary(docs)
    c = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = TfidfModel(c)
    result=[]    
    for s in c:
        tfidf_weights = tfidf[s]
        r=[]
        sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        for term_id, weight in sorted_tfidf_weights:
            r.append([dictionary.get(term_id), weight])
        result.append(r)
    return result

r=keywords(text2)
#print the keywords of the 1st doc in text2, from the most important to the least
print(r[0]) 
