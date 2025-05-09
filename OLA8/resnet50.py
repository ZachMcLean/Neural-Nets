#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import sys
import os
import csv

def get_dataloaders(use_augmentation, batch_size=100):
    # ResNet50 preprocessing: resize images to 224x224 and normalize using ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Download and load CIFAR10 dataset
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Split training dataset into training and validation sets (80-20 split)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        total += target.size(0)
        correct += (preds == target).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def main():
    if len(sys.argv) < 3:
        print("Usage: python resnet50.py [augmented|nonaugmented] [pretrained|nonpretrained]")
        sys.exit(1)
        
    aug_arg = sys.argv[1]
    pretrain_arg = sys.argv[2]
    use_augmentation = True if aug_arg == "augmented" else False
    use_pretrained = True if pretrain_arg == "pretrained" else False

    # Create logger directory based on configuration
    logger_dir = f"logs_{aug_arg}_{pretrain_arg}"
    os.makedirs(logger_dir, exist_ok=True)
    
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(use_augmentation, batch_size=100)
    
    # Load ResNet50 and modify the final fully connected layer for 10 classes
    model = models.resnet50(pretrained=use_pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()  # works as sparse categorical cross-entropy in PyTorch
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 50
    metrics = []  # List to store epoch metrics: [epoch, train_loss, train_acc, val_loss, val_acc]
    
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        metrics.append([epoch, train_loss, train_acc, val_loss, val_acc])
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Save training history metrics to CSV file
    metrics_csv = f"resnet50-metrics-{aug_arg}-{pretrain_arg}.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(metrics)
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    
    # Save test results to CSV file
    test_csv = f"resnet50-test-{aug_arg}-{pretrain_arg}.csv"
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["test_loss", "test_acc"])
        writer.writerow([test_loss, test_acc])

if __name__ == "__main__":
    main()
