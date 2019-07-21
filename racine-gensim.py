# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 03:13:33 2019

@author: a179415
"""

import pandas as pd
import numpy as np



from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import models
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint

import os
import codecs

os.chdir("C:/users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216")

STOP_WORDS="stop-words.csv"
stop_words_file = codecs.open(STOP_WORDS, "r", "utf-8")
stop_words = [ word.strip("\n") for word in stop_words_file.readlines()]


INPUT_FILE="stanford-nlp.txt"
text = codecs.open(INPUT_FILE, "r", "utf-8")
documents=[]
document=""
add_paragraph = False
for line in text.readlines():
    if add_paragraph:
        documents.append( document )
        add_paragraph = False
        document = ""
    tokens = line.strip(" \r\n").split(" ")
    filtered_tokens = [ token for token in tokens if ( len ( token) > 2 ) ]
    paragraph_separator = False
    for token in filtered_tokens:
       if token.upper() == token:
           paragraph_separator = True
           break
    if ( len( filtered_tokens ) == 0 ):
        paragraph_separator = True
    if paragraph_separator:
        if document != "":
            add_paragraph = True
    else:
        for token in filtered_tokens:
            if not (token in stop_words):
                document = document + " " + token
            
                

# Tokenize(split) the sentences into words
texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes( no_below = 11 )

# Tokenize the docs
tokenized_list = [simple_preprocess(doc) for doc in documents]

corpus = [dictionary.doc2bow(doc) for doc in tokenized_list]

# Create the TF-IDF model
tfidf = models.TfidfModel(corpus, smartirs='ntc')

lda_model = LdaMulticore(corpus=corpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=10,
                         passes=10,
                         chunksize=50,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=500,
                         gamma_threshold=0.001,
                         per_word_topics=True)

cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')

result = pd.DataFrame( columns=("topic","u_mass"))
for topic in range(5,60):
    print("topic " + str(topic))
    lda_model = LdaMulticore(corpus=corpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=topic,
                         passes=100,
                         chunksize=50,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=500,
                         gamma_threshold=0.001,
                         per_word_topics=True)
    cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
    result=result.append({"topic": topic, "u_mass":cm.get_coherence()}, ignore_index = True)
result.plot(x="topic",y="u_mass")






