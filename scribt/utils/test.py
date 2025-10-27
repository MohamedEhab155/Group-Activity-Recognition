from torchvision import transforms
from torchvision import models
import torch.nn as nn 

def prepare_model(image_level = False):
        if image_level:
                procecces= transforms.Compose(
                        transforms.Resize(256,256),
                        transforms.CenterCrop(224,224)
                        transforms.ToTensor()

                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization values from ImageNet (for pretrained ResNet).

                )

        model=models.resnet50(pretrained=True)

        model=nn.Sequential(*(list(model.children())[:-1]))




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
             preprocessed_imagge =preprocess (image).unsqueeze(0).to(device)
             dnn_image=model(preprocessed_imagge)
             dnn_image=dnn_image.view(1,-1)
        else:
             prossecced_image=[]

             for box in boxes_info: 
                  x1,x2,x3,x4=boxes_info.box

                  croped_image=image.crop(x1,x2,x3,x4)

                  preprocessed_imagge.append(preprocess.croped_image.unsqeenze(0))
                
             preprocessed_images = torch.cat(preprocessed_images).to(device)

             dnn_image=model(prossecced_image)
             dnn_image=dnn_image.view(len(prossecced_image),1) 

             

