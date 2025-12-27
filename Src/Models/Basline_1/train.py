

import sys
import os
import random
import numpy as np
import torch
# Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.volleyball_annot_loader import annotante, VideoFrameDataset
from model import B1ResNet50
from Src.helper.configs import get_config

Config = get_config('B1')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from .Trainer import Trainer
import torchvision.transforms as transforms
from Src.helper.configs import get_config
Config = get_config('B1')
from Src.utils.DatasetBuilder import DatasetBuilder
from .DataTransforms import DataTransforms


def main():
    """Main execution function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    

    train_loader, val_loader, _ = DatasetBuilder.create_dataloaders(Config, annotante, DataTransforms, VideoFrameDataset)
    
    print(f"\n  Initializing B1ResNet50 model...")
    model = B1ResNet50(num_classes=Config.NUM_CLASSES)
    print(f"✓ Model created with {Config.NUM_CLASSES} classes")
    
    trainer = Trainer(model, device, Config)
    trained_model = trainer.train(train_loader, val_loader)
    
    print(" Training pipeline completed successfully!")


if __name__ == "__main__":
    main()          