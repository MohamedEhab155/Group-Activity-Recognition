import torchvision.transforms as transforms
from Src.helper.configs import get_config
Config = get_config('B3')

class DataTransforms:

    @staticmethod
    def get_train_transform(config) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((Config.RESIZE_SIZE)),
            transforms.RandomHorizontalFlip(p=Config.RANDOM_FLIP_PROB),
            transforms.RandomRotation(degrees=Config.ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])
    
    @staticmethod
    def get_val_transform(config) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((Config.RESIZE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ])