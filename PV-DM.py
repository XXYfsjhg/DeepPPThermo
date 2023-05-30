import multiprocessing
import os
import pickle
import random

import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from embeddings_reproduction import embedding_tools


def train(X, sequence00, k, window, len):
    name_list = [X, str(k), str(window)]

    kmer_hypers = {'k': k,
                   'overlap': False,
                   'merge': False}
    model_hypers = {'vector_size': 32,
                    'min_count': 0,
                    'epochs': 5,
                    'window': window,
                    'workers': 4
                    }
    sequence00 = sequence00.reshape(len, 1)
    sequence00 = pd.DataFrame(sequence00)
    sequence_dict = {}
    sequence_dict['small'] = sequence00
    sequence_dict['small'].columns = ['sequence']

    documents = embedding_tools.Corpus(sequence_dict['small'], kmer_hypers)

    model = Doc2Vec(**model_hypers)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=25)
    model.save('../output/docvec_models/mpl32.model')


length = 3200
sequence_mpl = np.genfromtxt("../input/all_positive.fasta", dtype=str)
sequence_nmpl = np.genfromtxt("../input/all_negative.fasta", dtype=str)

size1 = sequence_mpl.shape[0]
size2 = sequence_nmpl.shape[0]

sequence_mpl = sequence_mpl[0:size1, ]
sequence_mpl = sequence_mpl.reshape(size1, 1)

sequence_nmpl = sequence_nmpl[0:size2, ]
sequence_nmpl = sequence_nmpl.reshape(size2, 1)

sequence00 = np.vstack((sequence_mpl, sequence_nmpl))    #竖直拼接

X = 'sequence00'
print(np.shape(sequence00))
train(X, sequence00, 3, 4, size2 + size1)

