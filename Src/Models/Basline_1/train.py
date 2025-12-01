"""
Volleyball Action Recognition Training Pipeline
================================================
Professional training script for video action recognition using B1ResNet50
"""


import sys
import os
from pathlib import Path
from typing import Tuple, List 
from tqdm import tqdm
import random 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.volleyball_annot_loader import annotante, VideoFrameDataset
from model import B1ResNet50
from config import Config


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class DataTransforms:
    """Data augmentation and preprocessing transforms"""
    
    @staticmethod
    def get_train_transform() -> transforms.Compose:
        """Returns training data transform pipeline with augmentation"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ])
    
    @staticmethod
    def get_val_transform() -> transforms.Compose:
        """Returns validation/test data transform pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ])


class DatasetBuilder:
    """Handles dataset preparation and loading"""
    
    @staticmethod
    def prepare_dataset(video_ids: List[int], dataset_path: str) -> List:
       
        dataset = []
        for video_id in video_ids:
            images_and_labels = annotante(video_id, dataset_path)
            dataset.extend(images_and_labels)
        return dataset
    
    @staticmethod
    def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
       
        print("📦 Preparing datasets...")
        
        # Prepare datasets
        train_data = DatasetBuilder.prepare_dataset(Config.TRAIN_VIDEOS, Config.DATASET_PATH)
        val_data = DatasetBuilder.prepare_dataset(Config.VAL_VIDEOS, Config.DATASET_PATH)
        test_data = DatasetBuilder.prepare_dataset(Config.TEST_VIDEOS, Config.DATASET_PATH)
        
        # Create dataset objects
        train_dataset = VideoFrameDataset(train_data, transform=DataTransforms.get_train_transform())
        val_dataset = VideoFrameDataset(val_data, transform=DataTransforms.get_val_transform())
        test_dataset = VideoFrameDataset(test_data, transform=DataTransforms.get_val_transform())
        
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        return train_loader, val_loader, test_loader


class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            device: Training device (CPU/GPU)
        """
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
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
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
        print(f" Model saved: {Config.MODEL_SAVE_PATH}")
    
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
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f" Train | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f" Val   | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
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


def main():
    """Main execution function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    

    train_loader, val_loader, test_loader = DatasetBuilder.create_dataloaders()
    
    print(f"\n  Initializing B1ResNet50 model...")
    model = B1ResNet50(num_classes=Config.NUM_CLASSES)
    print(f"✓ Model created with {Config.NUM_CLASSES} classes")
    
    # Initialize trainer and train
    trainer = Trainer(model, device)
    trained_model = trainer.train(train_loader, val_loader)
    
    print(" Training pipeline completed successfully!")


if __name__ == "__main__":
    main()          