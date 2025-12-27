import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Src.helper.configs import get_config
from typing import Tuple
from tqdm import tqdm
Config = get_config('B1')


class Trainer:
    
    def __init__(self, model: nn.Module, device: torch.device):

        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.LR_STEP_SIZE,
            gamma=Config.LR_GAMMA
        )
        
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
       
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self) -> None:
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"Model saved: {Config.MODEL_SAVE_PATH}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """
        Full training loop with validation and early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Trained model
        """
        print(f"\n Starting training on {self.device}")
        print(f" Total epochs: {Config.NUM_EPOCHS}")
        print(f" Early stopping patience: {Config.PATIENCE}\n")
        
        for epoch in range(Config.NUM_EPOCHS):
            print(f"{'='*50}")
            print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}]")
            print(f"{'='*50}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            
        
            val_loss, val_acc = self.validate_epoch(val_loader)
            
           
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f" Train | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f" Val   | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
            print(f" LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self.save_checkpoint()
                print(f" New best validation accuracy: {self.best_val_acc*100:.2f}%")
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s)")
            
            # Early stopping
            if self.epochs_no_improve >= Config.PATIENCE:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        print(f"{'='*50}")
        print(f" Training completed!")
        print(f" Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"{'='*50}\n")
        
        return self.model


