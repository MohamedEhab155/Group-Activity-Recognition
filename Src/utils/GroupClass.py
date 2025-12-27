from torch.utils.data import Dataset
import os
import cv2
from PIL import Image   


class VolleyballDataset(Dataset):
    def __init__(self, videos_root, labels_map, transform=None):
        self.samples = []
        self.transform = transform
        self.labels_map = labels_map

        videos_dirs = sorted(os.listdir(videos_root))
        for video_dir in videos_dirs:
            video_path = os.path.join(videos_root, video_dir)
            if not os.path.isdir(video_path):
                continue

            clips = sorted(os.listdir(video_path))
            for clip_dir in clips:
                clip_path = os.path.join(video_path, clip_dir)
                if not os.path.isdir(clip_path):
                    continue

                frames = sorted(os.listdir(clip_path))
                if len(frames) == 0:
                    continue

                # middle frame
                mid_idx = len(frames) // 2
                mid_frame = os.path.join(clip_path, frames[mid_idx])

              
                label = 0  
                self.samples.append((mid_frame, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR → RGB
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, label


def annotante(vidoe_id,Datasetpath):

    video_path=os.path.join(Datasetpath,str(vidoe_id))
    annotation_file=os.path.join(video_path,'annotations.txt')

    images_and_labels=[]

    with open(annotation_file) as f :
        for  line in f :
            parts=line.strip().split()

            frame_image=parts[0]

            frame_activity_class=parts[1]
            frame_id=os.path.splitext(frame_image)[0]
            target_image_path = os.path.join(video_path, str(frame_id), frame_image)

            if os.path.exists(target_image_path):
                images_and_labels.append((frame_id,frame_activity_class))
            else : 
                print ("is not found")

        return images_and_labels