import random
from typing import List, Tuple, Callable
from torch.utils.data import DataLoader
import os

class DatasetBuilder:
    """Handles dataset preparation and loading in a config-agnostic way."""

    @staticmethod
    def prepare_dataset(video_ids: List[int], dataset_path: str, annotator: Callable, shuffle: bool = True) -> List:
            dataset = []
            for video_id in video_ids:
                video_annotations = annotator(video_id, dataset_path)
                dataset.extend(video_annotations)

            if shuffle:
                random.shuffle(dataset)
            
            return dataset
    @staticmethod
    def create_dataloaders(config, annotator: Callable, dataset_class,DataTransforms ,collate_fn) -> Tuple[DataLoader, DataLoader, DataLoader, List]:
        print("Preparing datasets...")

        train_data = DatasetBuilder.prepare_dataset(config.TRAIN_VIDEOS, config.DATASET_PATH, annotator)
        val_data = DatasetBuilder.prepare_dataset(config.VAL_VIDEOS, config.DATASET_PATH, annotator, shuffle=False)
        test_data = DatasetBuilder.prepare_dataset(config.TEST_VIDEOS, config.DATASET_PATH, annotator, shuffle=False)

        print(f"✓ Raw train samples: {len(train_data)}")
        print(f"✓ Raw val samples: {len(val_data)}")
        print(f"✓ Raw test samples: {len(test_data)}")

        train_dataset = dataset_class(train_data, transform=DataTransforms.get_train_transform(config))
        val_dataset = dataset_class(val_data, transform=DataTransforms.get_val_transform(config))
        test_dataset = dataset_class(test_data, transform=DataTransforms.get_val_transform(config))

        print(f"✓ Train dataset length: {len(train_dataset)}")
        print(f"✓ Val dataset length: {len(val_dataset)}")
        print(f"✓ Test dataset length: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader, train_data
