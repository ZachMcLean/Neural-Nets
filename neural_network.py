# Module: neural_network.py

import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics

class NeuralNetwork(pl.LightningModule):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 num_hidden_layers, 
                 num_hidden_units_per_layer,
                 residual):
        """
        Neural network module capable of building single-layer, multi-layer, or residual networks.
        
        Parameters:
        - input_size (int): Size of input features
        - output_size (int): Number of output classes
        - num_hidden_layers (int): Number of hidden layers
        - num_hidden_units_per_layer (int): Number of hidden units per layer
        - residual (bool): Whether to use residual connections
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Always start with a flatten layer
        layers = [torch.nn.Flatten()]
        
        # Single-layer network (no hidden layers)
        if num_hidden_layers == 0:
            layers.append(torch.nn.Linear(input_size, output_size))
        
        # Multi-layer network (with or without residual connections)
        else:
            # First layer (input to first hidden)
            layers.append(torch.nn.Linear(input_size, num_hidden_units_per_layer))
            layers.append(torch.nn.ReLU())
            
            # Additional hidden layers
            for i in range(1, num_hidden_layers):
                if residual and i % 2 == 0:  # Create residual blocks (every 2 layers)
                    # Create a residual block
                    residual_block = ResidualBlock(num_hidden_units_per_layer)
                    layers.append(residual_block)
                else:
                    # Standard layer
                    layers.append(torch.nn.Linear(num_hidden_units_per_layer, num_hidden_units_per_layer))
                    layers.append(torch.nn.ReLU())
            
            # Output layer
            layers.append(torch.nn.Linear(num_hidden_units_per_layer, output_size))
        
        # Combine all layers
        self.model = torch.nn.Sequential(*layers)
        
        # Loss function and metrics
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', 
                                                           num_classes=output_size)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Predict with softmax activation."""
        logits = self(x)
        return torch.nn.functional.softmax(logits, dim=-1)
    
    def configure_optimizers(self):
        """Configure the Adam optimizer as required."""
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        """
        Residual block with two linear layers and ReLU activations.
        
        Parameters:
        - hidden_size (int): Number of hidden units
        """
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.layers(x)
        out = out + identity  # Add residual connection
        out = self.relu(out)
        return out