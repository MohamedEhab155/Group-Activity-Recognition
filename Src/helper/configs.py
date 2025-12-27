from typing import Dict, Tuple, List


class SharedConfig:
    DATASET_PATH: str = ""
    MODEL_SAVE_PATH: str = "best_model.pth"

    TRAIN_VIDEOS = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32,
                    36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
    VAL_VIDEOS = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
    TEST_VIDEOS = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class Baseline1Config(SharedConfig):
    
    
    # Paths
    MODEL_SAVE_PATH: str = "/kaggle/working/best_model_B1ResNet50.pth2"
    OUTPUT_PATH: str = "/kaggle/working/"
    
    # Dataset / Transform Settings
    RESIZE_SIZE: int = 256
    CROP_SIZE: int = 224
    ROTATION_DEGREES: int = 5
    CLASS_NAMES: list = [
        'r_set', 'r_spike', 'r-pass', 'r_winpoint',  
        'l_winpoint', 'l-pass', 'l-spike', 'l_set'
    ]
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 35
    LEARNING_RATE: float = 1e-4
    NUM_CLASSES: int = 8
    WEIGHT_DECAY: float = 1e-5
    LR_STEP_SIZE: int = 5
    LR_GAMMA: float = 0.1
    PATIENCE: int = 5
    
    # DataLoader Settings
    NUM_WORKERS: int = 0  # For Kaggle set to 0, else 4
    PIN_MEMORY: bool = True
    
    # Model Info
    MODEL_NAME: str = "B1ResNet50"

class Baseline3Config(SharedConfig):
    # Dataset / Paths
    DATASET_PATH: str = r'E:\deeblearning\volleyball-dataset\videos_sample'
    MODEL_SAVE_PATH: str = "best_model_B3.pth"
    OUTPUT_PATH: str = "/kaggle/working/"

    # Dataset / Transform Settings
    RESIZE_SIZE: tuple[int, int] = (224, 224)
    RANDOM_FLIP_PROB: float = 0.4
    ROTATION_DEGREES: int = 15

    CLASS_NAMES: list[str] = [
        'waiting', 'setting', 'digging', 'falling',
        'spiking', 'blocking', 'jumping', 'moving', 'standing'
    ]
    BATCH_SIZE: int = 4
    NUM_CLASSES: int = 9

    # Training Settings
    NUM_EPOCHS: int = 5
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-3

    LR_STEP_SIZE: int = 2
    LR_GAMMA: float = 0.1
    PATIENCE: int = 10

    NUM_WORKERS: int = 0  # For Kaggle set 0, else 4
    PIN_MEMORY: bool = True

    # Model Info
    MODEL_NAME: str = "B3ResNet50"


_MAP: Dict[str, SharedConfig] = {
    'B1': Baseline1Config,
    'B3': Baseline3Config,
}


def get_config(name: str = 'B1') -> SharedConfig:
    """Return the Config class for given baseline name (defaults to 'B1')."""
    return _MAP.get(name, Baseline1Config)
