import cv2
import os
import pickle
from typing import List
#from boxinfo import BoxInfo
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torch

def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        # let's create view from frame to boxes
        for player_ID, boxes_info in player_boxes.items():
            # let's keep the middle 9 frames only (enough for this task empirically)
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct


import os
import cv2

def vis_clip(annot_path, video_dir, save_dir="output_frames"):
    os.makedirs(save_dir, exist_ok=True)
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f"{frame_id}.jpg")
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Warning: Image not found at {img_path}")
            continue  # Skip this frame

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, box_info.category, (x1, y1 - 10),
                        font, 0.5, (0, 255, 0), 2)

        save_path = os.path.join(save_dir, f"{frame_id}.jpg")
        cv2.imwrite(save_path, image)

    print(f"Annotated frames saved in: {save_dir}")




def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            #print(f'\t{clip_dir_path}')
            assert clip_dir in clip_category_dct

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)
            #vis_clip(annot_file, clip_dir_path)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot
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


class VideoFrameDataset(Dataset):
    def __init__(self, data, transform=None):

        self.data=data
        self.transform=transform 

        self.class_mapping = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
                              'l_winpoint': 4, 'l-pass': 5, 'l-spike': 6, 'l_set':7} 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path,label=self.data[index]

        image=Image.open(image_path)

        if self.transform :
            image=self.transform(image)
        

        label=torch.tensor(self.class_mapping[label],dtype=torch.long)

        return image, label


if __name__ == '__main__':
    Datasetpath = r"E:\deeblearning\volleyball-dataset\videos_sample"

    print(annotante(7, Datasetpath))