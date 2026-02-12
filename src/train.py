"""
Training pipeline for the pneumonia detection model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from src.utils import (
    get_device, set_seed, save_checkpoint, 
    compute_metrics, EarlyStopping
)


class PneumoniaTrainer:
    """
    Trainer class for pneumonia detection model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            device (torch.device): Device to use
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training dataloader
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation dataloader
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        checkpoint_dir: str = './checkpoints',
        patience: int = 5
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader (DataLoader): Training dataloader
            val_loader (DataLoader): Validation dataloader
            num_epochs (int): Number of epochs
            checkpoint_dir (str): Directory to save checkpoints
            patience (int): Patience for early stopping
            
        Returns:
            Dict: Training history
        """
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_dir,
                    'best_model.pt'
                )
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        return history


def train_model(
    model_name: str = 'resnet18',
    data_dir: str = './data/chest_xray',
    batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = './checkpoints'
):
    """
    Complete training script.
    
    Args:
        model_name (str): Model name
        data_dir (str): Path to dataset
        batch_size (int): Batch size
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        checkpoint_dir (str): Checkpoint directory
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load dataloaders
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir,
        batch_size=batch_size
    )
    
    # Create model
    from src.model import get_model
    model = get_model(model_name)
    
    # Create trainer
    trainer = PneumoniaTrainer(
        model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    return history


if __name__ == '__main__':
    # Example usage
    history = train_model(
        model_name='resnet18',
        data_dir='./data/chest_xray',
        batch_size=32,
        num_epochs=20,
        learning_rate=1e-4
    )
