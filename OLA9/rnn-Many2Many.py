# rnn-Many2Many.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

num_sequences = 1000
seq_length = 50
input_dim = 1
num_classes = 2
hidden_size = 128
batch_size = 40
num_epochs = 300
learning_rate = 0.001

# Generate data with cumulative parity labels for each time step
def generate_parity_data_many2many(num_sequences, seq_length):
    X = np.random.randint(0, 2, size=(num_sequences, seq_length)).astype(np.float32)
    # Compute cumulative parity (mod 2) along the sequence
    y = np.mod(np.cumsum(X, axis=1), 2).astype(np.int64)
    X = np.expand_dims(X, axis=2)
    return X, y

X, y = generate_parity_data_many2many(num_sequences, seq_length)
split = int(0.8 * num_sequences)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val, dtype=torch.long)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define a Many-to-Many RNN model
class RNN_Many2Many(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(RNN_Many2Many, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_size, 
                          batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        out, _ = self.rnn(x)  # out: (batch, seq_length, hidden_size)
        logits = self.fc(out) # (batch, seq_length, num_classes)
        return logits

model = RNN_Many2Many(input_dim, hidden_size, num_classes)
print("RNN Many-to-Many Model:")
print(model)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameters:", count_parameters(model))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs shape: (batch, seq_length, num_classes)
        # Reshape outputs and labels for loss computation
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 2)  # shape: (batch, seq_length)
        total += labels.numel()
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 2)
            total_val += labels.numel()
            correct_val += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_dataset)
    val_acc = correct_val / total_val

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

os.makedirs("logs", exist_ok=True)
torch.save(history, "logs/history_rnn_m2m.pt")
print("Training complete. History saved to logs/history_rnn_m2m.pt")
