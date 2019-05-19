import torch
import torch.nn as nn
from torchvision import models


def get_model(name, pre_trained=True):
    if name == 'resnet':
        net = models.resnet18(pre_trained)
        net.fc.out_features = 200
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        return net
    elif name == 'squeeze':
        net = models.squeezenet1_1(pre_trained)
        net.classifier._modules["1"] = nn.Conv2d(512, 200, kernel_size=(1, 1))
        net.num_classes = 200
        return net
    elif name == 'alexnet':
        net = models.alexnet(pre_trained)
        #net.classifier[1] = nn.Linear(256 * 1 * 1, 4096)
        #net.classifier[6] = nn.Linear(4096, 200)
        return net
    else:
        raise Exception("Model not available!")


def get_pre_loaded_model(name, filename):
    net = get_model(name, False)
    net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s' % filename)
    return net
