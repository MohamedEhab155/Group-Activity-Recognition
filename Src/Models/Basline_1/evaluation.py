from train import DatasetBuilder  
from model import B1ResNet50 
import torch 
import torch.nn as nn 
from sklearn.metrics import f1_score, confusion_matrix, classification_report 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from tqdm import tqdm 
import os

# --------------------------
# Load Test DataLoader
# --------------------------
_, _, test_loader = DatasetBuilder.create_dataloaders()

# --------------------------
# Class names
# --------------------------
class_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',  
               'l_winpoint', 'l-pass', 'l-spike', 'l_set']

# --------------------------
# Load Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = B1ResNet50(num_class=8)
model.load_state_dict(torch.load("/kaggle/working/best_model_B1ResNet50.pth2", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Run Evaluation
# --------------------------
all_preds = []
all_labels = []

with torch.no_grad():  
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------
# Metrics
# --------------------------
acc = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_per_class = f1_score(all_labels, all_preds, average=None)

print(f"\n Accuracy: {acc:.2f}%")
print(f" F1 (Macro): {f1_macro:.4f}\n")
print("Classification Report:")
report_text = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report_text)

# --------------------------
# Save Classification Report as CSV
# --------------------------
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
csv_path = "/kaggle/working/B1ResNet50_classification_report.csv"
df_report.to_csv(csv_path, index=True)
print(f" Classification report saved as CSV at: {csv_path}")

# --------------------------
# Save Summary TXT
# --------------------------
summary_path = "/kaggle/working/B1ResNet50_summary.txt"
with open(summary_path, "w") as f:
    f.write("B1 ResNet50 Test Summary\n")
    f.write("="*40 + "\n")
    f.write(f"Accuracy: {acc:.2f}%\n")
    f.write(f"F1 (Macro): {f1_macro:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_text)

print(f"✅ Summary text file saved at: {summary_path}")

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - B1ResNet50")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save Confusion Matrix as Image
cm_path = "/kaggle/working/B1ResNet50_confusion_matrix.png"
plt.savefig(cm_path, bbox_inches="tight", dpi=300)
plt.show()
print(f" Confusion matrix saved as image at: {cm_path}")
