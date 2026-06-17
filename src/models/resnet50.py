
import torch.nn as nn
import torchvision.models as models

def build_resnet50(num_classes=5, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
