# Load your model with weights
import torch
from Src.utils.eval_utlize import MetricsSaver 
from Src.Models.Basline_3.model import  B3ResNet50
from Src.utils.DatasetBuilder import DatasetBuilder
from Src.helper import get_config
from tqdm import tqdm
from Src.utils.PlayerClass import Player, VideoFrameDataset


Config = get_config('B3')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = B3ResNet50(num_class=9)
model.load_state_dict(torch.load("E:\deeblearning\best_model_B3.pth", map_location=device))



# Load test loader
players=Player()
_, _, test_loader, _ = DatasetBuilder.create_dataloaders(Config,players.extract_target_image_and_label, VideoFrameDataset,players.collate_fn)

# Load model and evaluate
model.eval()
all_preds = []
all_labels = []
class_names = Config.CLASS_NAMES
with torch.no_grad():
    for imgs_batch, lbls_batch in tqdm(test_loader):
        imgs = torch.stack([img for frame in imgs_batch for img in frame]).to(device)
        labels = torch.stack([lbl for frame in lbls_batch for lbl in frame]).to(device)
        outputs = model(imgs)
        mask = labels != -1
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())

# Save metrics
metrics_saver = MetricsSaver(class_names, Config.MODEL_NAME, Config.OUTPUT_PATH)
metrics_saver.calculate_metrics(all_preds, all_labels)
metrics_saver.print_metrics()
metrics_saver.save_results()