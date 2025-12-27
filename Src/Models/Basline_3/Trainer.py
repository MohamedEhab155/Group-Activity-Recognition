from collections import Counter
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, config, train_data, use_weighted_loss=False, 
                 num_classes=9, lr=1e-4, weight_decay=1e-3, step_size=2, gamma=0.1,
                 save_dir='./checkpoints', model_name='best_model'):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.model_name = model_name
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Compute class weights - FIXED to match friend's logic
        label_counts = Counter([player['action_class'] 
                                for frame in train_data 
                                for player in frame['players']])
        total_samples = sum(label_counts.values())
        class_weights = [total_samples / label_counts[cls] if cls in label_counts else 1e-3 
                         for cls in range(num_classes)]
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        if use_weighted_loss:
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def custom_loss(self, outputs, labels):
        mask = labels != -1
        outputs = outputs[mask]
        labels = labels[mask]
        if len(labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        return self.criterion(outputs, labels)

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for imgs_batch, lbls_batch in tqdm(dataloader, desc="Training", leave=False):
            imgs = torch.stack([img for frame in imgs_batch for img in frame]).to(self.device)
            labels = torch.stack([lbl for frame in lbls_batch for lbl in frame]).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.custom_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy and collect predictions
            mask = labels != -1
            _, preds = torch.max(outputs, 1)
            total_samples += mask.sum().item()
            total_correct += ((preds == labels) & mask).sum().item()
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100. * total_correct / total_samples
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_accuracy, epoch_f1

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs_batch, lbls_batch in tqdm(dataloader, desc="Validation", leave=False):
                imgs = torch.stack([img for frame in imgs_batch for img in frame]).to(self.device)
                labels = torch.stack([lbl for frame in lbls_batch for lbl in frame]).to(self.device)

                outputs = self.model(imgs)
                loss = self.custom_loss(outputs, labels)
                running_loss += loss.item()

                # Calculate accuracy and collect predictions
                mask = labels != -1
                _, preds = torch.max(outputs, 1)
                total_samples += mask.sum().item()
                total_correct += ((preds == labels) & mask).sum().item()
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        accuracy = 100. * total_correct / total_samples
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, epoch_f1

    def save_checkpoint(self, epoch, val_f1, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'best_val_f1': self.best_val_f1
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, f'{self.model_name}.pth')
            torch.save(checkpoint, save_path)
            print(f" Best model saved! F1: {val_f1:.4f} -> {save_path}")

    def train(self, trainloader, valloader, num_epochs=10, patience=5):
        """
        Main training loop with early stopping
        
        Args:
            trainloader: Training data loader
            valloader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience (default: 5)
        """
        print("\n" + "="*70)
        print(f" Starting Training")
        print("="*70)
        print(f"Total Epochs: {num_epochs}")
        print(f"Early Stopping Patience: {patience}")
        print(f"Save Directory: {self.save_dir}")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f" Epoch [{epoch+1}/{num_epochs}]")
            print(f"{'='*70}")
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(trainloader)
            
            # Validation phase
            val_loss, val_acc, val_f1 = self.validate_epoch(valloader)
            
            # Step the scheduler
            self.scheduler.step()
            
            # Print metrics in a nice table format
            print(f"\n{'Metric':<20} {'Training':<15} {'Validation':<15}")
            print(f"{'-'*50}")
            print(f"{'Loss':<20} {train_loss:<15.4f} {val_loss:<15.4f}")
            print(f"{'Accuracy (%)':<20} {train_acc:<15.2f} {val_acc:<15.2f}")
            print(f"{'F1 Score':<20} {train_f1:<15.4f} {val_f1:<15.4f}")
            print(f"{'-'*50}")
            
            # Check if this is the best model
            if val_f1 > self.best_val_f1:
                print(f" New best F1 score! {self.best_val_f1:.4f} -> {val_f1:.4f}")
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, val_f1, is_best=True)
            else:
                self.patience_counter += 1
                print(f" No improvement. Patience: {self.patience_counter}/{patience}")
            
            # Early stopping check
            if self.patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"  Early stopping triggered!")
                print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                print(f"{'='*70}\n")
                break
        
        # Training complete summary
        print("\n" + "="*70)
        print(f" Training Complete!")
        print("="*70)
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Model saved at: {os.path.join(self.save_dir, f'{self.model_name}.pth')}")
        print("="*70 + "\n")
        
        return self.best_val_f1, self.best_epoch
