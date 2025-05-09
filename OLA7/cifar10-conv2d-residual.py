#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import csv
import os

# Define a Residual Conv2d Block using two 3x3 conv layers with a skip connection
class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# Define the Residual Conv2d Network
class ResidualConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResidualConvNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ResidualConvBlock(32)
        self.pool1 = nn.MaxPool2d(2)  # from 32x32 -> 16x16
        self.block2 = ResidualConvBlock(32)
        self.pool2 = nn.MaxPool2d(2)  # 16x16 -> 8x8
        self.block3 = ResidualConvBlock(32)
        self.block4 = ResidualConvBlock(32)
        self.pool3 = nn.MaxPool2d(2)  # 8x8 -> 4x4
        self.block5 = ResidualConvBlock(32)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.block1(out)
        out = self.pool1(out)
        out = self.block2(out)
        out = self.pool2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool3(out)
        out = self.block5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / total, correct / total

def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, correct / total

def main():
    num_epochs = 50
    batch_size = 128
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=False, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResidualConvNet(num_classes=10).to(device)
    print("Residual Conv2d Network Architecture:")
    print(model)
    print("Total trainable parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_dir = "logs/conv2d_residual"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    csv_file = "cifar10-metrics-conv2d-residual.csv"
    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_conv2d_residual_model.pth")
    writer.close()

if __name__ == '__main__':
    main()
