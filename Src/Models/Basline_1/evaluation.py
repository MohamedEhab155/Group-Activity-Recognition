from train import DatasetBuilder  
from model import B1ResNet50 
import torch 
from tqdm import tqdm 
from Src.helper.configs import get_config
from utils.volleyball_annot_loader import annotante, VideoFrameDataset
from .DataTransforms import DataTransforms

from Src.utils.eval_utlize import MetricsSaver

Config = get_config('B1')
_, _, test_loader = DatasetBuilder.create_dataloaders(Config, annotante, DataTransforms, VideoFrameDataset)


class_names = Config.CLASS_NAMES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = B1ResNet50(num_class=8)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():  
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

metrics_saver = MetricsSaver(class_names, model_name=Config.MODEL_NAME, output_path=Config.OUTPUT_PATH)
metrics_saver.calculate_metrics(all_preds, all_labels)
metrics_saver.print_metrics()
metrics_saver.save_results()