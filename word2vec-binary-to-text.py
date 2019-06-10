# coding: utf-8
from __future__ import division

import os
import codecs
# import gzip

os.chdir("C:/users/a179415/OneDrive - Alliance/Personal/CNAM/RCP 216")

FILE_NAME="frWac_no_postag_no_phrase_500_skip_cut100.bin"
output_file_name = FILE_NAME + ".txt"


from gensim.models import KeyedVectors
output = codecs.open(output_file_name, 'w' , 'utf-8')
model_bin = KeyedVectors.load_word2vec_format( FILE_NAME , binary = True)
vocab = model_bin.vocab
for mid in vocab:
    vector = list()
    for dimension in model_bin[mid]:
        vector.append(str(dimension))
    vector_str = ",".join(vector)
    line = mid + "\t"  + vector_str
    output.write(line + "\n")
output.close()