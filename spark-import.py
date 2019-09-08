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
from operator import itemgetter
# from matplotlib import cm
import os

os.chdir("C:/users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216")

rawTopic2Topic=[
        (0,"Liens familiaux"),
        (-1,"Unknown"),
        (1, "Divin"),
        (2, "Pouvoir"),
        (3,"Victoire"),
        (-1,"Unknown"),
        (-1,"Unknown"),
        (4,"Bonheur"),
        (5,"Malheur"),
        (6,"Jugement"),
        (-1,"Unknown"),        
        (5, "Malheur"),
        (7,"Trahison"),
        (-1, "Inconnu"),
        (-1, "Inconnu"),
        (8, "Vengeance"),
        (-1, "Unknown")
        ]
lsa_eigen_values = pd.read_csv( "lsa-eigenValues.csv" )
lsa_eigen_values.plot()

## Fréquence des mots dans les documents
docfreqs= pd.read_csv("docfreqs.tsv", sep='\t' , encoding='latin1')
docfreqs.hist( range=[0,25], bins=20)


word2vec = pd.read_csv("word2vec-measures-tst.csv", sep='\t' , encoding='latin1')
word2vec.plot(x="topic", y="UMass", title="word2vec",marker="o")
word2vec_UMass = word2vec.plot(x="topic", y="UMass", title="kMeans sur word2vec")
word2vec_UMass.axvline(x=11.08, linestyle="--", color="black")
word2vec_UMass.text( x=5, y = -0.54, s="11 topics", bbox=dict(boxstyle="round", fc="white",ec="grey"))
lda = pd.read_csv("lda-measures-tst.csv", sep='\t' , encoding='latin1')

lsa = pd.read_csv("lsa-measures-tst.csv", sep='\t' , encoding='latin1')
lsa_UMass = lsa.plot(x="topic", y="UMass",title="lsa", marker="o")
lsa_UMass.axvline(x=11.08, linestyle="--", color="black")
lsa_UMass.text( x=5, y = -0.54, s="11 topics", bbox=dict(boxstyle="round", fc="white",ec="grey"))

lsa_word2vec = lsa.plot(x="topic", y="word2vect",title="lsa", marker="o")
lsa_word2vec.axvline(x=17, linestyle="--", color="black")
lsa_word2vec.text( x=18, y = 0.0255, s="17 topics", bbox=dict(boxstyle="round", fc="white",ec="grey"))


lda = pd.read_csv("lda-measures-tst.csv", sep='\t' , encoding='latin1')
lda_UMass = lda.plot(x="topic", y="UMass",title="lda", marker="o")


ldaLogit = pd.read_csv("lda-log.csv", sep='\t' , encoding='latin1')
plt.plot( ldaLogit.loc[:,"topic"], ldaLogit.loc[:,"UMass"])
plt.title("Topic coherence WSSE measure")
plt.xlabel("topic")
plt.ylabel("Word2Vec")


# topic doc visualization
topicMap = pd.read_csv("word2vec-topicMap-tst.csv", sep='\t' , encoding='latin1')
topicWords= pd.read_csv("word2vec-topicWords-tst.csv", sep='\t' , encoding='latin1' )

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
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a"
        ]



to_hex = lambda color: c.to_rgb( color )

colorMapRGB = list(map( to_hex, colorMapHex ))
colorMapRGB.append( c.to_rgb("grey"))
colorMapRGB.append( c.to_rgb("white"))


colorMap = c.ListedColormap( colorMapRGB)

REAL_TOPIC_NB=len( colorMapHex )
RAW_TOPIC_NB = topicWords.shape[0]
MAX_TOPIC = 2
labels = []
labels.append("Top " + str(len( colorMapHex )) + " topics")
labels.append("")
pictures = []
empty_rectangle = ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=c.to_rgb("white"))
pictures.append( empty_rectangle )
pictures.append( empty_rectangle )
realTopics = []
for index in range( topicWords.shape[0]):
    real_topic_index = rawTopic2Topic[index][0]
    real_topic_label = rawTopic2Topic[index][1]
    if  ( real_topic_index != -1 ) and ( not ( real_topic_index in realTopics)):
        realTopics.append( real_topic_index )
        pictures.append(ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=colorMapRGB[ real_topic_index ]))
        label = "["+ real_topic_label + "] "
        for term in range(1,6):
            term_name = "term_name_"+ str(term)
            label = label + " " + topicWords.loc[ index, term_name ]
        labels.append( label) 
pictures.append(ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=colorMapRGB[ REAL_TOPIC_NB ]))
labels.append("Unknown")
#
# Plain Topic with pcolormesh
#
WIDTH=40


topicImage = np.zeros((topicMap.shape[0] , WIDTH), dtype='int32')
topic_max_index = topicMap.shape[0]
for index, document in topicMap.iterrows():
    max_index = 1
    # min = document["topic_weight_1"] * 0.1
    sum = 0
    realTopics = []

    for i in range(1,RAW_TOPIC_NB):
        max_index += 1
        raw_topic_index = int(document["topic_index_" + str( i ) ])
        real_topic_index = rawTopic2Topic[raw_topic_index][0]
        real_topic_label = rawTopic2Topic[raw_topic_index][1]        
        raw_topic_weight = document["topic_weight_" + str( i ) ]
        if  ( not ( real_topic_index in realTopics)):            
            realTopics.append( real_topic_index )
            sum = sum + raw_topic_weight 
            if len( realTopics) == MAX_TOPIC:
                break
    print( realTopics)
    print( "sum " + str(sum))

    realTopics = []
    topicPixelSize = []
    max_topic_weight = document["topic_weight_1" ]
    for i in range(1, max_index):
        raw_topic_index = int(document["topic_index_" + str( i ) ])
        real_topic_index = rawTopic2Topic[raw_topic_index][0]
        real_topic_label = rawTopic2Topic[raw_topic_index][1]
        raw_topic_weight = document["topic_weight_" + str( i ) ]
        if ( sum != 0):            
            if  ( not ( real_topic_index in realTopics)):                
                realTopics.append( real_topic_index)
                additional_pixel = int(( WIDTH * max_topic_weight) * ( raw_topic_weight / sum))
                topicPixelSize.append(( real_topic_index, additional_pixel))
                print("Index " + str( real_topic_index) + " pixel " + str(additional_pixel))
    current_pixel = 0
    real_index = topic_max_index - index - 1
    for topic_index, additional_pixel in sorted(topicPixelSize, key=itemgetter(0)):
        if  topic_index != -1:
            topicImage[ real_index , current_pixel: current_pixel + additional_pixel] =  topic_index
        else:
            topicImage[ real_index , current_pixel: current_pixel + additional_pixel] =  REAL_TOPIC_NB
        current_pixel += additional_pixel
    topicImage[ real_index, current_pixel: WIDTH ] = REAL_TOPIC_NB + 2
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
# fig.suptitle("Plain Topic through pcolormesh")
axs.set_axis_off()
documentAx = fig.add_subplot(121)
documentAx.set_axis_off()
documentAx.pcolormesh(topicImage, vmin=0, vmax=REAL_TOPIC_NB + 2, rasterized=True, cmap=colorMap) #, shading='gouraud')
legendAx = fig.add_subplot(624)
legendAx.set_axis_off()
legendAx.legend( pictures, labels,ncol=1, loc='center right')
# axs.axis("off")
# axs.pcolormesh(topicImage[0:50,:], vmin=0, vmax=TOPIC_NB + 1, rasterized=True, cmap=colorMap, shading="gouraud")
# axs.pcolormesh(topicImage[0:40,:], vmin=0, vmax=TOPIC_NB + 1, rasterized=True, cmap=colorMap, shading ="gouraud")
# axs.imshow(topicImage[0:40:], cmap=colorMap, vmin=0, vmax= TOPIC_NB + 1)

# legend
   




# deeper analysis
colorMapRGBABig = []
for color in colorMapHex:
    rgba = c.to_rgb( color )
    for index in range(10):
        colorMapRGBABig.append( c.to_rgba( (rgba[0], rgba[1], rgba[2],  1.0)))
# colorMapRGBABig.append( c.to_rgba("white"))
colorMapBig = c.ListedColormap( colorMapRGBABig)
    
WIDTH=40
TOPIC_NB=10
topicImage = np.zeros((topicMap.shape[0] , WIDTH), dtype='int32')
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
        topic_index = int(document["topic_index" + str(i)])
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * topic_weight )
            topicImage[ index , current_pixel: additional_pixel] =  topic_index * 10 + topic_weight *10
            current_pixel += additional_pixel
    topicImage[ index, current_pixel: WIDTH ] = REAL_TOPIC_NB * 10
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
axs.set_axis_off()
documentAx = fig.add_subplot(121)
documentAx.set_axis_off()
documentAx.pcolormesh(topicImage, vmin=0, vmax=TOPIC_NB * 10, rasterized=True, cmap=colorMap, shading ="gouraud")
legendAx = fig.add_subplot(624)
legendAx.set_axis_off()
legendAx.legend( pictures, labels,ncol=1)

test=np.zeros(110).reshape(11,10)
for topicId in range(10):
    for column in range(10):
        strength = ( 10 - column ) / 10.0
        test[ 9 - topicId, column] = topicId +  (1 - strength)

test[10,:]=11
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
axs.legend( pictures, labels)
axs.pcolormesh(test, cmap=colorMapBig, shading="flat")


#
# plain topic wih imgshow
#
WIDTH=200
TOPIC_NB=10
topicImage = np.zeros((topicMap.shape[0] , WIDTH, 4), dtype='float64')
topicCount = np.zeros(10)
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
        topicCount[ topic_index ] +=1
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * topic_weight )
            rgb = colorMapRGB[ topic_index]
            topicImage[ index , current_pixel: additional_pixel] =  c.to_rgba( ( rgb[0], rgb[1], rgb[2], 1.0)) #topic_weight))
            current_pixel += additional_pixel
    topicImage[ index, current_pixel: WIDTH ] = c.to_rgba("white")
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
documentAx = fig.add_subplot(121)
documentAx.imshow(topicImage)
documentAx.set_axis_off()
legendAx = fig.add_subplot(6,2,10)
legendAx.legend( pictures, labels,ncol=1)
legendAx.set_axis_off()
axs.set_axis_off()
fig.suptitle("Plain topic (imshow implementation)")
histo=fig.add_subplot(6,2,4)
histo.bar( np.arange(TOPIC_NB), topicCount, color=colorMapHex)


#
# topic with importance with imshow
#
WIDTH=200
TOPIC_NB=10
topicImage = np.zeros((topicMap.shape[0] , WIDTH, 4), dtype='float64')
topicCount = np.zeros(10)
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
        topicCount[ topic_index ] +=1
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * topic_weight )
            rgb = colorMapRGB[ topic_index]
            topicImage[ index , current_pixel: additional_pixel] =  c.to_rgba( ( rgb[0], rgb[1], rgb[2], topic_weight))
            current_pixel += additional_pixel
    topicImage[ index, current_pixel: WIDTH ] = c.to_rgba("white")
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
documentAx = fig.add_subplot(121)
documentAx.imshow(topicImage)
documentAx.set_axis_off()
legendAx = fig.add_subplot(6,2,10)
legendAx.legend( pictures, labels,ncol=1)
legendAx.set_axis_off()
axs.set_axis_off()
fig.suptitle("Topic with importance (imshow implementation)")
histo=fig.add_subplot(6,2,4)
histo.bar( np.arange(TOPIC_NB), topicCount, color=colorMapHex)

