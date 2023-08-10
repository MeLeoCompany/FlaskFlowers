import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def model_init():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def param_init(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 10
    return num_epochs, optimizer, lr_scheduler