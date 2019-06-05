# -*- coding: utf-8 -*-
"""
Created on Mon May 27 05:28:49 2019

@author: a179415
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib.colors as c
from matplotlib import cm
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
 
colorMapHex = [
        "#a50026",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695"
        ]


WIDTH=400
topicImage = np.zeros((topicMap.shape[0] , WIDTH,3), dtype='int32')
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
    current_pixel = 0
    for i in range(1, max_index):
        topic_weight = document["topic_weight_" + str(i)]
        topic_index = int(document["topic_index_" + str(i)])
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * ( topic_weight / sum))
            topicImage[ index , current_pixel: additional_pixel, 0] = colorMap[ topic_index][0]
            topicImage[ index , current_pixel: additional_pixel, 1] = colorMap[ topic_index][1]
            topicImage[ index , current_pixel: additional_pixel, 2] = colorMap[ topic_index][2]
            current_pixel += additional_pixel
labels = []
pictures = []
for index in range( len(colorMap)):
    pictures.append(ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=(0.4549019607843137, 0.6784313725490196, 0.8196078431372549)))
    labels.append( "topic" + str(index))
    
plt.imshow(topicImage)
plt.legend( pictures, labels,ncol=2, bbox_to_anchor= (0,0, 0.5, 0.5))