# Script: cifar10-residual.py

import torch
import lightning.pytorch as pl
from neural_network import NeuralNetwork
from cifar10_data import CIFAR10DataModule
from torchinfo import summary

def main():
    # Create data module
    data_module = CIFAR10DataModule()
    data_module.setup()
    
    # Create model with same hidden layers and units as deep network
    # but with residual connections
    model = NeuralNetwork(
        input_size=data_module.input_shape,
        output_size=data_module.output_shape,
        num_hidden_layers=3,
        num_hidden_units_per_layer=350,
        residual=True
    )
    
    # Print model summary
    print("Residual network summary:")
    summary(model, input_size=(1, 32, 32, 3))
    
    # Create logger
    logger = pl.loggers.CSVLogger("logs", name="cifar10_residual")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=50,  # 50 epochs as required
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=0,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=20)]
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Residual model parameter count: {param_count}")
    print(f"Logger directory: {logger.log_dir}")

if __name__ == "__main__":
    main()