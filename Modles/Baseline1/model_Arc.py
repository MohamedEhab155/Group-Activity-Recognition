from torchvision import models
import torch.nn as nn


class B1ResNet50(nn.Module):
    def __init__(self,num_class=8): 
        super(B1ResNet50,self).__init__()
        self.base=models.resnet50(pretrained=True)
        in_features=self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_class)   
    def forward(self,x):
        return self.base(x)
