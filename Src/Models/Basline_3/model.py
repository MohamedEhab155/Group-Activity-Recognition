from torchvision import models
import torch.nn as nn



class B3ResNet50(nn.Module):
    def __init__(self, num_class=9, freeze_layers=True):
        super(B3ResNet50, self).__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_class)

        if freeze_layers:
            for layer in [self.base.layer1, self.base.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.base(x)
