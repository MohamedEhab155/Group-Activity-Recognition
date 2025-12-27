import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


class MetricsSaver:
    """
    Handles metrics calculation, printing, and saving only.
    No data loading or model evaluation here.
    """
    
    def __init__(self, class_names, model_name, output_dir="/kaggle/working"):
        """
        Initialize metrics saver.
        
        Args:
            class_names: List of class names
            model_name: String identifier for the model
            output_dir: Directory to save results
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
    
    def calculate_metrics(self, all_preds, all_labels):
        """
        Calculate all metrics from predictions and labels.
        
        Args:
            all_preds: List or array of predictions
            all_labels: List or array of true labels
        """
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        
        # Overall metrics
        self.metrics['accuracy'] = 100 * (preds == labels).mean()
        self.metrics['f1_macro'] = f1_score(labels, preds, average='macro')
        self.metrics['f1_weighted'] = f1_score(labels, preds, average='weighted')
        self.metrics['f1_per_class'] = f1_score(labels, preds, average=None)
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # Classification report
        self.metrics['classification_report'] = classification_report(
            labels, preds, 
            target_names=self.class_names, 
            output_dict=True,
            digits=4
        )
        
        # Store for later use
        self.all_preds = preds
        self.all_labels = labels
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics."""
        print(f"\n{'='*70}")
        print(f"  {self.model_name} - Test Evaluation Results")
        print(f"{'='*70}")
        print(f"  Accuracy:       {self.metrics['accuracy']:.2f}%")
        print(f"  F1 (Macro):     {self.metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted):  {self.metrics['f1_weighted']:.4f}")
        print(f"{'='*70}\n")
        
        print("Per-Class F1 Scores:")
        print(f"{'-'*40}")
        for class_name, f1 in zip(self.class_names, self.metrics['f1_per_class']):
            print(f"  {class_name:20s}: {f1:.4f}")
        print(f"{'-'*40}\n")
        
        # Print classification report
        report_text = classification_report(
            self.all_labels, self.all_preds,
            target_names=self.class_names,
            digits=4
        )
        print("Classification Report:")
        print(report_text)
    
    def save_results(self):
        """Save all evaluation results to files."""
        # Save classification report as CSV
        df_report = pd.DataFrame(self.metrics['classification_report']).transpose()
        csv_path = self.output_dir / f"{self.model_name}_classification_report.csv"
        df_report.to_csv(csv_path, index=True)
        print(f"✓ Classification report saved: {csv_path}")
        
        # Save summary text file
        summary_path = self.output_dir / f"{self.model_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"{self.model_name} - Test Evaluation Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Accuracy:       {self.metrics['accuracy']:.2f}%\n")
            f.write(f"F1 (Macro):     {self.metrics['f1_macro']:.4f}\n")
            f.write(f"F1 (Weighted):  {self.metrics['f1_weighted']:.4f}\n\n")
            
            f.write("Per-Class F1 Scores:\n")
            f.write("-"*40 + "\n")
            for class_name, f1 in zip(self.class_names, self.metrics['f1_per_class']):
                f.write(f"  {class_name:20s}: {f1:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Detailed Classification Report:\n")
            f.write("="*70 + "\n\n")
            
            report_text = classification_report(
                self.all_labels, self.all_preds,
                target_names=self.class_names,
                digits=4
            )
            f.write(report_text)
        
        print(f"✓ Summary saved: {summary_path}")
        
        # Save visualizations
        self.save_confusion_matrix()
        self.save_per_class_f1_plot()
    
    def save_confusion_matrix(self):
        """Save confusion matrix heatmap."""
        cm = self.metrics['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f"Confusion Matrix - {self.model_name}", fontsize=14, pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = self.output_dir / f"{self.model_name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✓ Confusion matrix saved: {cm_path}")
    
    def save_per_class_f1_plot(self):
        """Save per-class F1 scores bar plot."""
        f1_scores = self.metrics['f1_per_class']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(self.class_names)), f1_scores, 
                       color='steelblue', alpha=0.8)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.title(f"Per-Class F1 Scores - {self.model_name}", fontsize=14, pad=20)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        f1_plot_path = self.output_dir / f"{self.model_name}_f1_scores.png"
        plt.savefig(f1_plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"✓ F1 scores plot saved: {f1_plot_path}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example 1: Using validate_epoch from Trainer
    """
    from train import DatasetBuilder, Trainer
    from model import B3ResNet50
    from config import get_config
    from player import player
    from dataset import VideoFrameDataset
    
    # Setup
    config = get_config('B3')
    players = player()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = [
        'waiting', 'setting', 'digging', 'falling',
        'spiking', 'blocking', 'jumping', 'moving', 'standing'
    ]
    
    # Load test dataloader
    print("Loading test data...")
    _, _, test_loader, train_data = DatasetBuilder.create_dataloaders(
        config,
        players.extract_target_image_and_label,
        VideoFrameDataset,
        players.collate_fn
    )
    
    # Load model
    model = B3ResNet50(num_class=9)
    checkpoint = torch.load("/kaggle/working/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create a temporary trainer just to use validate_epoch
    trainer = Trainer(
        model=model,
        device=device,
        config=config,
        train_data=train_data,
        num_classes=9
    )
    
    # Use validate_epoch from Trainer to get predictions
    print("\nRunning evaluation using Trainer.validate_epoch...")
    test_loss, test_acc, test_f1 = trainer.validate_epoch(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1 (Weighted): {test_f1:.4f}")
    
    # Now use MetricsSaver to calculate and save all metrics
    metrics_saver = MetricsSaver(
        class_names=class_names,
        model_name="B3ResNet50",
        output_dir="/kaggle/working"
    )
    
    # Calculate metrics from trainer's predictions
    metrics_saver.calculate_metrics(trainer.all_preds, trainer.all_labels)
    
    # Print and save
    metrics_saver.print_metrics()
    metrics_saver.save_results()
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)


"""
Example 2: Manual evaluation loop in main script
"""
def example_manual_evaluation():
    from train import DatasetBuilder
    from model import B3ResNet50
    from config import get_config
    from player import player
    from dataset import VideoFrameDataset
    from tqdm import tqdm
    
    # Setup
    config = get_config('B3')
    players = player()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = [
        'waiting', 'setting', 'digging', 'falling',
        'spiking', 'blocking', 'jumping', 'moving', 'standing'
    ]
    
    # Load test dataloader
    print("Loading test data...")
    _, _, test_loader, _ = DatasetBuilder.create_dataloaders(
        config,
        players.extract_target_image_and_label,
        VideoFrameDataset,
        players.collate_fn
    )
    
    # Load model
    model = B3ResNet50(num_class=9)
    checkpoint = torch.load("/kaggle/working/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Manual evaluation loop
    all_preds = []
    all_labels = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for imgs_batch, lbls_batch in tqdm(test_loader, desc="Testing"):
            imgs = torch.stack([img for frame in imgs_batch for img in frame]).to(device)
            labels = torch.stack([lbl for frame in lbls_batch for lbl in frame]).to(device)
            
            outputs = model(imgs)
            
            # Filter -1 labels
            mask = labels != -1
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    # Use MetricsSaver to calculate and save all metrics
    metrics_saver = MetricsSaver(
        class_names=class_names,
        model_name="B3ResNet50",
        output_dir="/kaggle/working"
    )
    
    metrics_saver.calculate_metrics(all_preds, all_labels)
    metrics_saver.print_metrics()
    metrics_saver.save_results()
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)