import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import os


dataset = load_dataset("snli")
train_data = dataset['train']
print("Train length:", len(train_data))
print(dataset)


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens

train_data = train_data.map(lambda x: {
    'premise_tokens': preprocess_text(x['premise']),
    'hypothesis_tokens': preprocess_text(x['hypothesis'])
})

def build_vocab(tokens_list):
    vocab = Counter()
    for tokens in tokens_list:
        vocab.update(tokens)
    return vocab

tokens_list = train_data['premise_tokens'] + train_data['hypothesis_tokens']
vocab = build_vocab(tokens_list)

glove_vectors = {}
embedding_dim = 300

with open("/home/scur1410/glove.840B.300d.txt", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        if len(values) != 301:
            continue
        word = values[0]
        try:
            vector = np.array(values[1:], dtype='float32')
            glove_vectors[word] = vector
        except ValueError:
            continue

word_to_idx = {'<pad>': 0}
idx = 1
for word in vocab:
    if word in glove_vectors:
        word_to_idx[word] = idx
        idx += 1

embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
for word, i in word_to_idx.items():
    if word in glove_vectors:
        embedding_matrix[i] = glove_vectors[word]



def encode(tokens):
    return [word_to_idx.get(t, 0) for t in tokens]


premise_encoded = [torch.tensor(encode(t)) for t in train_data['premise_tokens']]
hypothesis_encoded = [torch.tensor(encode(t)) for t in train_data['hypothesis_tokens']]
labels = torch.tensor([l if l != -1 else 0 for l in train_data['label']])

premise_padded = pad_sequence(premise_encoded, batch_first=True, padding_value=0)
hypothesis_padded = pad_sequence(hypothesis_encoded, batch_first=True, padding_value=0)





val_data = dataset['validation']
val_data = val_data.map(lambda x: {
    'premise_tokens': preprocess_text(x['premise']),
    'hypothesis_tokens': preprocess_text(x['hypothesis'])
})

val_premise_encoded = [torch.tensor(encode(t)) for t in val_data['premise_tokens']]
val_hypothesis_encoded = [torch.tensor(encode(t)) for t in val_data['hypothesis_tokens']]
val_labels = torch.tensor([l if l != -1 else 0 for l in val_data['label']])

val_premise_padded = pad_sequence(val_premise_encoded, batch_first=True, padding_value=0)
val_hypothesis_padded = pad_sequence(val_hypothesis_encoded, batch_first=True, padding_value=0)

train_dataset = TensorDataset(premise_padded, hypothesis_padded, labels)
val_dataset = TensorDataset(val_premise_padded, val_hypothesis_padded, val_labels)

torch.save(train_dataset, "/home/scur1410/Practical_1/train_dataset_saved.pt")
torch.save(val_dataset, "/home/scur1410/Practical_1/val_dataset_saved.pt")
torch.save(embedding_matrix, "/home/scur1410/Practical_1/embedding_matrix.pt")
torch.save(word_to_idx, "/home/scur1410/Practical_1/word_to_idx.pt")