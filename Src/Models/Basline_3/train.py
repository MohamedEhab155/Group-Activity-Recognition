
from .model import B3ResNet50
from Src.helper import get_config

# Use central configs.py so changes affect all baselines from one place
Config = get_config('B3')

import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from Src.utils.PlayerClass import player, VideoFrameDataset

from Src.utils.DatasetBuilder import DatasetBuilder
from.DataTransforms import DataTransforms
from .Trainer import Trainer

def main():

    

    """Main execution function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    players=player()
    train_loader, val_loader, _ ,train_data= DatasetBuilder.create_dataloaders(Config,players.extract_target_image_and_label,
                                                                             VideoFrameDataset,DataTransforms,players.collate_fn)

    print(f"\n  Initializing B3ResNet50 model...")
    model = B3ResNet50(num_class=Config.NUM_CLASSES)
    print(f"✓ Model created with {Config.NUM_CLASSES} classes")
    
    trainer = Trainer(model, device,config=Config,use_weighted_loss=True, train_data=train_data,num_classes=9 )
    trained_model = trainer.train(train_loader, val_loader)
    
    print(" Training pipeline completed successfully!")

if __name__ == "__main__":
    main()