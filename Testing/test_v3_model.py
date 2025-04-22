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
from torch.utils.data import TensorDataset, DataLoader
nltk.download('punkt')
word_to_idx = torch.load("/home/scur1410/Practical_1/word_to_idx.pt")


pad_idx = word_to_idx.get('<pad>', None)
unk_idx = word_to_idx.get('<unk>', None)

print(f"<pad> index: {pad_idx}")
print(f"<unk> index: {unk_idx}")

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens

dataset = load_dataset("snli")
test_data = dataset['test']
test_data = test_data.map(lambda x: {
    'premise_tokens': preprocess_text(x['premise']),
    'hypothesis_tokens': preprocess_text(x['hypothesis'])
})

def encode(tokens):
    return [word_to_idx.get(t, 0) for t in tokens]


test_premise_encoded = [torch.tensor(encode(t)) for t in test_data['premise_tokens']]
test_hypothesis_encoded = [torch.tensor(encode(t)) for t in test_data['hypothesis_tokens']]
test_labels = torch.tensor([l if l != -1 else 0 for l in test_data['label']])
test_premise_padded = pad_sequence(test_premise_encoded, batch_first=True, padding_value=0)
test_hypothesis_padded = pad_sequence(test_hypothesis_encoded, batch_first=True, padding_value=0)
test_dataset = TensorDataset(test_premise_padded, test_hypothesis_padded, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64)

embedding_matrix = torch.load("/home/scur1410/Practical_1/embedding_matrix.pt",weights_only=False)
embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True, padding_idx=0)

def combine(u, v):
    return torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

class SentenceEncoder(nn.Module):
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



class NLIClassifier(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim=512, output_dim=3):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(4 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, premise, hypothesis):
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        combined = combine(u, v)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)

hidden_dim = 300
model = NLIClassifier(SentenceEncoder(embedding_layer, hidden_dim), input_dim=2 * hidden_dim)

model.load_state_dict(torch.load("/scratch-shared/chatty_atcs/v3/snli_v3_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for premise, hypothesis, label in test_loader:
        outputs = model(premise, hypothesis)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

