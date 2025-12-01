import os
import numpy as np
import cv2
import torch
from PIL import __version__ as PILLOW_VERSION

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from volleyball_annot_loader import load_tracking_annot

dataset_root = "E:/deeblearning/Group-Activity-Recognition/volleyball-dataset"


def check():
    print('torch: version', torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")

        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Get the name of the current device
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")




def prepare_model(image_level = False):
    if image_level:
        # image has a lot of space around objects. Let's crop around
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization values from ImageNet (for pretrained ResNet).

        ])
    else:
        # already croped box. Don't crop more
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)  

    model = nn.Sequential(*(list(model.children())[:-1]))

    model.to(device)

    model.eval()

    return model, preprocess


def extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level=False):
    frame_boxes = load_tracking_annot(annot_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)   

    with torch.no_grad():
        for frame_id, boxes_info in frame_boxes.items():
            try:
                img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')

                if image_level:
                    preprocessed_image = preprocess(image).unsqueeze(0).to(device)  
                    dnn_repr = model(preprocessed_image)
                    dnn_repr = dnn_repr.view(1, -1)
                else:
                    preprocessed_images = []
                    for box_info in boxes_info:
                        x1, y1, x2, y2 = box_info.box
                        cropped_image = image.crop((x1, y1, x2, y2))
                        preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))

                    preprocessed_images = torch.cat(preprocessed_images).to(device) 
                    dnn_repr = model(preprocessed_images)
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1)

                np.save(output_file, dnn_repr.cpu().numpy())  
            except Exception as e:
                print(f"An error occurred: {e}")



def temp():
    categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]


    val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]


if __name__ == '__main__':
    check()

    # image_level: extract features for the whole image or just a crop
    image_level = False
    model, preprocess = prepare_model(image_level)

    videos_root = f'{dataset_root}/videos_sample'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation'
    output_root = f'{dataset_root}/features/image-level/resnet'

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            print(f'\t{clip_dir_path}')

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            output_file = os.path.join(output_root, video_dir)

            if not os.path.exists(output_file):
                os.makedirs(output_file)

            output_file = os.path.join(output_file, f'{clip_dir}.npy')
            extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level = image_level)
