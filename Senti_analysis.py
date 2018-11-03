# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:43:30 2018

@author: yangl
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:17:02 2018

@author: yangl

This a script for Sentiment Analysis with either vader or sentiwordnet
"""

import pandas as pd
import nltk
from nltk import tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

#construct text1 with seven documents/sentences
text1="It was one of the worst movies I've seen, despite good reviews. \
 Unbelievably Good acting!! Poor direction. VERY poor production. \
 The movie was perfect. Very bad movie. VERY bad movie."
 
# function sort_by_sentiment will sort the sentences in text from positive to negative
def sort_by_sentiment(text,desc=True):
    sia = SIA()
    results = []
    text_s=tokenize.sent_tokenize(text)

    for s in text_s:
      pol_score = sia.polarity_scores(s)
      pol_score['headline'] = s
      results.append(pol_score)

    df = pd.DataFrame.from_records(results)
    df.head()

    sort_df=df.sort_values('compound',ascending= not desc)
    return(sort_df)

tt=sort_by_sentiment(text1)
tt.head()



doc="Nice and friendly place with excellent food and friendly and helpful staff.\
 You need a car though.\
 I love this house very much.\
 Playground and animals entertained them and they felt like at home.\
 I also recommend the dinner! \
 Great value for the price!\
 This movies is bad, I will not recommend it to my friends"

def wordnet_sentAnalyzer(doc):
    sentences = nltk.sent_tokenize(doc.lower())
    stokens = [nltk.word_tokenize(sent) for sent in sentences]
    taggedlist=[]
    for stoken in stokens:        
         taggedlist.append(nltk.pos_tag(stoken))
    wnl = nltk.WordNetLemmatizer()

    score_list=[]
    for ids,taggedsent in enumerate(taggedlist):
        score_list.append([])
        for idw,w in enumerate(taggedsent):
            newtag=''
            lemmatized=wnl.lemmatize(w[0])
            #tag noun
            if w[1].startswith('NN'):
                newtag='n'
            #tag adj.
            elif w[1].startswith('JJ'):
                newtag='a'
            #tag verb
            elif w[1].startswith('V'):
                newtag='v'
            #tag adv.
            elif w[1].startswith('R'):
                newtag='r'
            else:
                newtag=''       
            if(newtag!=''):    
                synsets = list(swn.senti_synsets(lemmatized, newtag))
                #Getting average of all possible sentiments, as you requested        
                obj=0; pos=0; neg=0
                if(len(synsets)>0):
                    for syn in synsets:
                    #objective score
                        obj+=syn.obj_score()
                    #positive score
                        pos+=syn.pos_score()
                    #negative score
                        neg+=syn.neg_score()
                    score_list[ids].append(list([obj/len(synsets),pos/len(synsets),neg/len(synsets)]))

    sent_sentiment=[]; sentences = nltk.sent_tokenize(doc)

    for idx,s in enumerate(score_list):
        o=0; p=0; n=0
        for w in s:
            o+=w[0]; p+=w[1]; n+=w[2]
        o=o/len(s); p=p/len(s); n=n/len(s)
        sent_sentiment.append([sentences[idx],o,p,n])
    #configure a pandas dataframe with each row contains a sentence and its sentiment socres
    df=pd.DataFrame.from_records(sent_sentiment, columns=['Sentences', 'Objective','Positive','Negative'])
    return(df)

df=wordnet_sentAnalyzer(doc)
print(df)