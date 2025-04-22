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

log_dir = os.path.join("/home/scur1410/Practical_1/runs", "snli_v1")
writer = SummaryWriter(log_dir)
nltk.download('punkt')
embedding_matrix = torch.load("/home/scur1410/Practical_1/embedding_matrix.pt",weights_only=False)
embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True, padding_idx=0)
train_dataset = torch.load("/home/scur1410/Practical_1/train_dataset.pt",weights_only=False)
val_dataset = torch.load("/home/scur1410/Practical_1/val_dataset.pt",weights_only=False)

print("Total train samples:",len(train_dataset))
print("Total val sampeles:",len(val_dataset))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

class SentenceEncoder(nn.Module):
    def __init__(self, embedding_layer, hidden_dim=300):
        super().__init__()
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        embedded = self.embedding(x)  
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()  

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)  
        return h_n.squeeze(0)  


def combine(u, v):
    return torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

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
model = NLIClassifier(SentenceEncoder(embedding_layer), input_dim=hidden_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.99)
loss_fn = CrossEntropyLoss(ignore_index=-1)
num_epochs = 50
current_lr = 0.1
best_val_acc = 0.0
print("Training started")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, (premise, hypothesis, label) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(premise, hypothesis)
        loss = loss_fn(outputs, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar("Loss/Batch", loss.item(), epoch * len(train_loader) + step)

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/Epoch", avg_loss, epoch)

    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for premise, hypothesis, label in val_loader:
            outputs = model(premise, hypothesis)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    val_acc = correct / total
    writer.add_scalar("Accuracy/Val", val_acc, epoch)
    print("Accuracy/Val", val_acc, epoch)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")


    if val_acc < best_val_acc:
        current_lr /= 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"LR reduced to {current_lr:.6f}")
    else:
        best_val_acc = val_acc


    if current_lr < 1e-5:
        print("Stopping early: learning rate below 1e-5")
        break
    torch.save(model.state_dict(), "/scratch-shared/chatty_atcs/v1/snli_v1_model.pth")

writer.close()
torch.save(model.state_dict(), "/scratch-shared/chatty_atcs/v1/snli_v1_model.pth")
