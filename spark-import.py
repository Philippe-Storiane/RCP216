# -*- coding: utf-8 -*-
"""
Created on Mon May 27 05:28:49 2019

@author: a179415
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("C:/users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216")

lsa_eigen_values = "lsaEigenValues.csv"
data = pd.read_csv( lsa_eigen_values,header = None)
plt.plot( data)

## Fréquence des mots dans les documents
data2= pd.read_csv("full-vocab-freq.tsv", sep='\t' , encoding='latin1', header = None)
plt.hist(data2.iloc[:,1], range=[0,25], bins=20)


word2vecWSSE = pd.read_csv("word2vec-wsse.csv", sep='\t' , encoding='latin1', header = None)

ldaLogit = pd.read_csv("lda-log.csv", sep='\t' , encoding='latin1')
plt.plot( ldaLogit.loc[:,"topic"], ldaLogit.loc[:,"UMass"])
plt.title("Topic coherence WSSE measure")
plt.xlabel("topic")
plt.ylabel("Word2Vec")

topicMap = pd.read_csv("lda-topicMap.csv", sep='\t' , encoding='latin1')

colorMap = [
        (165,0,38),
        (215, 48,39),
        (244,109,67),
        (253,174,97),
        (254,224,144),
        (224,243,248),
        (171,217,233),
        (116,173,209),
        (69,117,180),
        (49,54,149)]
 

for index, document in topicMap.iterrows():
    max_index = 0
    min = document["topic_weight_1"] * 0.1
    sum = 0
    for i in range(1,5):
        max_index += 1
        weight = document["topic_weight_" + str( i ) ]
        if ( weight >= min):
            sum = sum + weight
        else:
            break
    print(sum, max_index)
        