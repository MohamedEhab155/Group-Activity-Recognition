import os 
import logging
from PIL import Image 
from torch.utils.data import Dataset
import torch


class Player: 
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def crop_player(self, image_path, X, Y, W, H):
        """Crop a player region from the target frame."""
        image = Image.open(image_path)
        width, height = image.size

        x1 = max(0, X - W // 2)
        y1 = max(0, Y - H // 2)
        x2 = min(width, X + W // 2)
        y2 = min(height, Y + H // 2)

        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image
    
    def extract_target_image_and_label(self, video_id, dataset_path):
        """Extract target frames and player labels for a specific video."""
        video_path = os.path.join(dataset_path, str(video_id))
        annotation_file = os.path.join(video_path, 'annotations.txt')

        frame_annotations = []

        if not os.path.exists(annotation_file):
            return frame_annotations

        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_image = parts[0]
                frame_id = os.path.splitext(frame_image)[0]
                target_image_path = os.path.join(video_path, str(frame_id), frame_image)

                players = []
                for i in range(2, len(parts), 5):
                    if i + 4 < len(parts):
                        X, Y, W, H = map(int, parts[i:i+4])
                        action_class = parts[i + 4]
                        players.append({
                            'action_class': action_class,
                            'X': X,
                            'Y': Y,
                            'W': W,
                            'H': H
                        })

                frame_annotations.append({
                    'frame_id': frame_id,
                    'frame_image_path': target_image_path,
                    'players': players
                })

        return frame_annotations

    def collate_fn(self,batch):
            player_images_batch = []
            player_labels_batch = []

            for imgs, lbls in batch:
                player_images_batch.append(imgs)
                player_labels_batch.append(lbls)

            return player_images_batch, player_labels_batch

class VideoFrameDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.player_class_mapping = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3,
            'spiking': 4, 'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8
        }
        self.player = Player()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_data = self.data[idx]
        frame_image_path = frame_data['frame_image_path']
        players = frame_data['players']

        player_images = []
        player_labels = []

        for player in players:
            cropped_image = self.player.crop_player(
                frame_image_path, player['X'], player['Y'], player['W'], player['H']
            )

            if self.transform:
                cropped_image = self.transform(cropped_image)

            label = torch.tensor(self.player_class_mapping[player['action_class']], dtype=torch.long)

            player_images.append(cropped_image)
            player_labels.append(label)

        while len(player_images) < 12:
            dummy_image = torch.zeros((3, 224, 224))
            dummy_label = torch.tensor(-1, dtype=torch.long)
            player_images.append(dummy_image)
            player_labels.append(dummy_label)

        player_images = player_images[:12]
        player_labels = player_labels[:12]

        return player_images, player_labels


  
