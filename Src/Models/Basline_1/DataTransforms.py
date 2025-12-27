import torchvision.transforms as transforms
from Src.helper.configs import get_config
Config = get_config('B1')

class DataTransforms:
    """Data augmentation and preprocessing transforms"""
    
    @staticmethod
    def get_train_transform() -> transforms.Compose:
        """Returns training data transform pipeline with augmentation"""
        return transforms.Compose([
            transforms.Resize(Config.RESIZE_SIZE),
            transforms.RandomResizedCrop(Config.CROP_SIZE),
            transforms.RandomRotation(degrees=Config.ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ])
    
    @staticmethod
    def get_val_transform() -> transforms.Compose:
        """Returns validation/test data transform pipeline"""
        return transforms.Compose([
            transforms.Resize(Config.RESIZE_SIZE),
            transforms.CenterCrop(Config.CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ])
