# *DeepPPThermo*：a deep learning framework for predicting protein thermostability combining the protein level and amino acid level features.

## Introduction
DeepPPThermo fuse features of classical sequence features and deep learning representation features for classifying thermophilic and mesophilic proteins. In this model, DNN and BiLSTM are used, respectively, to mine hidden features. Further, local attention and global attention mechanism give different importance to the multi-view features. The fused features are fed to a fully connected network classifier to distinguish thermophilic and mesophilic proteins. 

## Guidance to use:
# dataset
The Fasta Sequence folder contains the original Fasta sequence
The SSFs folder contains features calculated by ilearnplus

## PV-DM
K-mer embedding is obtained based on PV-DM model

## seq2tensor.py
Using encoded k-mer represented fasta sequences in the dataset.

# DeepPPThermo.py
The subject of the algorithm. Use features in the dataset to predict and classify proteins.

# BiLSTM.py 
# SSFs+attention
An algorithm that uses only one type of feature

# robust_test.py
It is used to investigate the robustness of the algorithm after deleting an SSFs.
