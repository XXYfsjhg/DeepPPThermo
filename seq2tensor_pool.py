import numpy as np
import torch
import gensim
from torch import nn
import math

device = torch.device('cuda:0')
'''
doc2vec k-mer向量拼接，动态池化:nn.AdaptiveAvgPool
'''


def seq2tensor(seq, model, k):
    seq_tensor = []
    seq_length = len(seq)
    for i in range(seq_length-2):
        a = model.wv[str(seq[i:i+k])]
        if len(seq_tensor) == 0:
            seq_tensor = a
        else:
            seq_tensor = np.vstack((seq_tensor, a))
    seq_tensor = torch.Tensor(seq_tensor).float()
    seq_tensor = torch.unsqueeze(seq_tensor, 0)
    pool = nn.AdaptiveAvgPool2d((100, 32))  # (池化后的长度,特征维度)
    seq_tensor = pool(seq_tensor)
    return seq_tensor


k = 3
sequence_mpl = np.genfromtxt("../input/positive.txt", dtype=str)
sequence_nmpl = np.genfromtxt("../input/all_negative.fasta", dtype=str)


size1 = sequence_mpl.shape[0]
size2 = sequence_nmpl.shape[0]


sequence_nmpl = sequence_nmpl[0:size1, ]

# sequence = np.vstack((sequence_mpl, sequence_nmpl))    #竖直拼接
# sequence = sequence.reshape(1, size * 2)[0]

model = gensim.models.Doc2Vec.load('../output/docvec_models/mpl32.model')

data_tensor = seq2tensor(sequence_mpl[0], model, k)
for i in range(1, size1):
    seqi_tensor = seq2tensor(sequence_mpl[i], model, k)
    data_tensor = torch.cat((data_tensor, seqi_tensor), dim=0)

data_tensor1 = seq2tensor(sequence_nmpl[0], model, k)
for j in range(1, size1):
    seqi_tensor = seq2tensor(sequence_nmpl[j], model, k)
    data_tensor1 = torch.cat((data_tensor1, seqi_tensor), dim=0)

print(np.shape(data_tensor), np.shape(data_tensor1))
torch.save(data_tensor, "./Final_positive.pt")
torch.save(data_tensor1, "./Final_negative.pt")
