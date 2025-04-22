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
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import senteval


word_to_idx = torch.load("/home/scur1410/Practical_1/word_to_idx.pt")
def encode(tokens):
    return [word_to_idx.get(t, 0) for t in tokens]

embedding_matrix = torch.load("/home/scur1410/Practical_1/embedding_matrix.pt",weights_only=False)
embedding_layer_saved = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True, padding_idx=0)

class SentenceEncoder_v3(nn.Module):
    def __init__(self, embedding_layer, hidden_dim=300):
        super().__init__()
        self.embedding = embedding_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        embedded = self.embedding(x)  
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()

        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        sentence_rep, _ = torch.max(lstm_out, dim=1)  
        return sentence_rep



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceEncoder_v3(embedding_layer_saved).to(device)
model.eval()


def prepare(params, samples):
    return

def batcher(params, batch):
    
    encoded = [torch.tensor(encode(tokens), dtype=torch.long) for tokens in batch]
    padded_batch = pad_sequence(encoded, batch_first=True, padding_value=0).to(device)

    with torch.no_grad():
        embeddings = model(padded_batch)

    return embeddings.cpu().numpy()

params = {
    'task_path': '/home/scur1410/SentEval/data',
    'usepytorch': True,
    'kfold': 5,
    'classifier': {
        'nhid': 0,
        'optim': 'adam',
        'batch_size': 64,
        'tenacity': 3,
        'epoch_size': 4
    }
}

def compute_macro_micro(results):
    total_acc = 0
    total_samples = 0
    macro_scores = []
    
    for task, val in results.items():
        if 'devacc' in val:
            acc = val['devacc']
            print("Task:",task)
            n_samples = val.get('ndev', 0)
            print("Accuracy:",acc)
            print("Dev samples:",n_samples)

            total_acc += acc * n_samples
            total_samples += n_samples
            macro_scores.append(acc)
    
    macro = sum(macro_scores) / len(macro_scores) if macro_scores else 0
    micro = total_acc / total_samples if total_samples else 0
    
    return macro, micro





se = senteval.engine.SE(params, batcher, prepare)
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
results = se.eval(transfer_tasks)
macro_acc, micro_acc = compute_macro_micro(results)
print(f"Macro accuracy: {macro_acc:.2f}")
print(f"Micro accuracy: {micro_acc:.2f}")



