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
# from matplotlib import cm
import os

os.chdir("C:/users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216")

lsa_eigen_values = "lsaEigenValues.csv"
data = pd.read_csv( lsa_eigen_values,header = None)
plt.plot( data)

## Fréquence des mots dans les documents
data2= pd.read_csv("docfreqs.tsv", sep='\t' , encoding='latin1', header = None)
plt.hist(data2.iloc[:,1], range=[0,25], bins=20)


word2vecWSSE = pd.read_csv("word2vec-wsse.csv", sep='\t' , encoding='latin1', header = None)

ldaLogit = pd.read_csv("lda-log.csv", sep='\t' , encoding='latin1')
plt.plot( ldaLogit.loc[:,"topic"], ldaLogit.loc[:,"UMass"])
plt.title("Topic coherence WSSE measure")
plt.xlabel("topic")
plt.ylabel("Word2Vec")


# topic doc visualization
topicMap = pd.read_csv("lda-topicMap.csv", sep='\t' , encoding='latin1')
topicWords= pd.read_csv("lda-topicWords.csv", sep='\t' , encoding='latin1' )
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
colorMapRGB.append( c.to_rgb("white"))

colorMap = c.ListedColormap( colorMapRGB)

labels = []
labels.append("Top " + str(topicWords.shape[0]) + " topics")
labels.append("")
pictures = []
empty_rectangle = ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=c.to_rgb("white"))
pictures.append( empty_rectangle )
pictures.append( empty_rectangle )
for index in range(10):
    pictures.append(ptc.Rectangle( xy=(0,0), width=5, height=5, facecolor=colorMapRGB[ index ]))
    label = ""    
    for term in range(1,6):
        term_name = "terme_name_"+ str(term)
        label = label + " " + topicWords.loc[ index, term_name ]
    labels.append( label) 

#
# Plain Topic with pcolormesh
#
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
        topic_index = int(document["topic_index_" + str(i)])
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * topic_weight )
            topicImage[ index , current_pixel: additional_pixel] =  topic_index
            current_pixel += additional_pixel
    topicImage[ index, current_pixel: WIDTH ] = TOPIC_NB + 1
fig, axs = plt.subplots(figsize=(9, 9), constrained_layout=True)
# fig.suptitle("Plain Topic through pcolormesh")
axs.set_axis_off()
documentAx = fig.add_subplot(121)
documentAx.set_axis_off()
documentAx.pcolormesh(topicImage, vmin=0, vmax=TOPIC_NB + 1, rasterized=True, cmap=colorMap) #, shading='gouraud')
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
        topic_index = int(document["topic_index_" + str(i)])
        if ( sum != 0):
            additional_pixel = int(current_pixel + WIDTH * topic_weight )
            topicImage[ index , current_pixel: additional_pixel] =  topic_index * 10 + topic_weight *10
            current_pixel += additional_pixel
    topicImage[ index, current_pixel: WIDTH ] = TOPIC_NB * 10
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

