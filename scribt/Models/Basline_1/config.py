from typing import Tuple, List 
class Config:
    """Training configuration parameters"""
    
    # Paths
    DATASET_PATH: str = r'E:\deeblearning\volleyball-dataset\videos_sample'
    MODEL_SAVE_PATH: str = "best_model_B1ResNet50.pth"
    
    # Dataset splits
    TRAIN_VIDEOS: List[int] = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 
                                36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    VAL_VIDEOS: List[int] = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    TEST_VIDEOS: List[int] = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    
    # Model parameters
    NUM_CLASSES: int = 8
    
    # Training hyperparameters
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 35
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5
    
    # Learning rate scheduler
    LR_STEP_SIZE: int = 20
    LR_GAMMA: float = 0.1
    
    # Early stopping
    PATIENCE: int = 10
    
    # DataLoader settings
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    # Image normalization (ImageNet stats)
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)