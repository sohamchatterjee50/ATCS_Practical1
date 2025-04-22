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




log_dir = os.path.join("/home/scur1410/Practical_1/runs", "snli_baseline")
writer = SummaryWriter(log_dir)

nltk.download('punkt')


# dataset = load_dataset("snli")
# train_data = dataset['train']
# print("Train length:", len(train_data))
# print(dataset)


# def preprocess_text(text):
#     text = text.lower()
#     tokens = word_tokenize(text)
#     return tokens

# train_data = train_data.map(lambda x: {
#     'premise_tokens': preprocess_text(x['premise']),
#     'hypothesis_tokens': preprocess_text(x['hypothesis'])
# })


# def build_vocab(tokens_list):
#     vocab = Counter()
#     for tokens in tokens_list:
#         vocab.update(tokens)
#     return vocab

# tokens_list = train_data['premise_tokens'] + train_data['hypothesis_tokens']
# vocab = build_vocab(tokens_list)


# glove_vectors = {}
# embedding_dim = 300

# with open("/home/scur1410/glove.840B.300d.txt", 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.strip().split()
#         if len(values) != 301:
#             continue
#         word = values[0]
#         try:
#             vector = np.array(values[1:], dtype='float32')
#             glove_vectors[word] = vector
#         except ValueError:
#             continue

# word_to_idx = {'<pad>': 0}
# idx = 1
# for word in vocab:
#     if word in glove_vectors:
#         word_to_idx[word] = idx
#         idx += 1

# embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
# for word, i in word_to_idx.items():
#     if word in glove_vectors:
#         embedding_matrix[i] = glove_vectors[word]


word_to_idx = torch.load("/home/scur1410/Practical_1/word_to_idx.pt")
embedding_matrix = torch.load("/home/scur1410/Practical_1/embedding_matrix.pt",weights_only=False)
embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True, padding_idx=0)

def encode(tokens):
    return [word_to_idx.get(t, 0) for t in tokens]

# premise_encoded = [torch.tensor(encode(t)) for t in train_data['premise_tokens']]
# hypothesis_encoded = [torch.tensor(encode(t)) for t in train_data['hypothesis_tokens']]
# labels = torch.tensor([l if l != -1 else 0 for l in train_data['label']])

# premise_padded = pad_sequence(premise_encoded, batch_first=True, padding_value=0)
# hypothesis_padded = pad_sequence(hypothesis_encoded, batch_first=True, padding_value=0)





# val_data = dataset['validation']
# val_data = val_data.map(lambda x: {
#     'premise_tokens': preprocess_text(x['premise']),
#     'hypothesis_tokens': preprocess_text(x['hypothesis'])
# })

# val_premise_encoded = [torch.tensor(encode(t)) for t in val_data['premise_tokens']]
# val_hypothesis_encoded = [torch.tensor(encode(t)) for t in val_data['hypothesis_tokens']]
# val_labels = torch.tensor([l if l != -1 else 0 for l in val_data['label']])

# val_premise_padded = pad_sequence(val_premise_encoded, batch_first=True, padding_value=0)
# val_hypothesis_padded = pad_sequence(val_hypothesis_encoded, batch_first=True, padding_value=0)

# train_dataset = TensorDataset(premise_padded, hypothesis_padded, labels)
# val_dataset = TensorDataset(val_premise_padded, val_hypothesis_padded, val_labels)


train_dataset = torch.load("/home/scur1410/Practical_1/train_dataset.pt",weights_only=False)
val_dataset = torch.load("/home/scur1410/Practical_1/val_dataset.pt",weights_only=False)

print("Total train samples:",len(train_dataset))
print("Total val sampeles:",len(val_dataset))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding = embedding_layer

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        summed = torch.sum(embedded * mask, dim=1)
        lengths = torch.sum(mask, dim=1)
        return summed / lengths.clamp(min=1)

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
num_epochs = 10
current_lr = 0.1
best_val_acc = 0.0
print("Traiing started")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, (premise, hypothesis, label) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(premise, hypothesis)
        # print("Outputs:",outputs)
        # print("Labels:",label)
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
        print(f"↓ LR reduced to {current_lr:.6f}")
    else:
        best_val_acc = val_acc

 
    if current_lr < 1e-5:
        print("Stopping early: learning rate below 1e-5")
        break
    
    torch.save(model.state_dict(), "/scratch-shared/chatty_atcs/Baseline/snli_baseline_model.pth")

writer.close()
torch.save(model.state_dict(), "/scratch-shared/chatty_atcs/Baseline/snli_baseline_model.pth")
